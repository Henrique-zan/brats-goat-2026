import numpy as np
import torch
from scipy.ndimage import gaussian_filter, map_coordinates, binary_dilation, center_of_mass
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from batchgeneratorsv2.transforms.base.basic_transform import BasicTransform


# ----------------------- helpers (numpy/scipy, rodam SO no ROI do tumor) -----------------------

def _rand32(shape, rng):
    return rng.standard_normal(shape, dtype=np.float32)


def _smooth_disp(shape, sigma, max_disp, rng):
    nd = len(shape)
    disp = np.empty((nd, *shape), dtype=np.float32)
    for d in range(nd):
        f = gaussian_filter(_rand32(shape, rng), sigma)
        disp[d] = f * np.float32(max_disp / (np.abs(f).max() + 1e-8))
    return disp


def _tumor_weight(seg_bin, dil, ssig):
    w = binary_dilation(seg_bin, iterations=int(dil)).astype(np.float32)
    w = gaussian_filter(w, ssig)
    w /= (w.max() + 1e-8)
    return w


def _tumor_elastic_disp(shape, seg_bin, sigma, max_disp, dil, ssig, rng):
    disp = _smooth_disp(shape, sigma, max_disp, rng)
    disp *= _tumor_weight(seg_bin, dil, ssig)[None]
    return disp


def _tumor_scale_disp(shape, seg_bin, scale, dil, ssig):
    w = _tumor_weight(seg_bin, dil, ssig)
    com = center_of_mass(seg_bin)
    nd = len(shape)
    fac = np.float32(1.0 / scale - 1.0)
    disp = np.empty((nd, *shape), dtype=np.float32)
    for d in range(nd):
        g = (np.arange(shape[d], dtype=np.float32) - np.float32(com[d])) * fac
        sh = [1] * nd; sh[d] = shape[d]
        disp[d] = g.reshape(sh)
    disp *= w[None]
    return disp


def _disp_to_coords(disp):
    # disp (nd, *shape) -> coordenadas absolutas NO SISTEMA DO RECORTE (in-place)
    nd = disp.shape[0]
    shape = disp.shape[1:]
    for d in range(nd):
        g = np.arange(shape[d], dtype=np.float32)
        sh = [1] * nd; sh[d] = shape[d]
        disp[d] += g.reshape(sh)
    return disp


def _tumor_bbox(seg_bin, spatial, margin):
    # bounding-box do tumor (seg>0) + margem, clampado ao patch
    nz = np.where(seg_bin)
    roi = []
    for d in range(len(spatial)):
        lo = max(0, int(nz[d].min()) - margin)
        hi = min(spatial[d], int(nz[d].max()) + 1 + margin)
        roi.append((lo, hi))
    return tuple(roi)


def _edge_fade_mask(shape, fade):
    # janela separavel que e 1 no interior e cai a 0 EXATO nos `fade` voxels de cada borda.
    # forca deslocamento=0 na borda do ROI -> warp do recorte == warp do patch inteiro (sem seam).
    nd = len(shape)
    mask = np.ones(shape, dtype=np.float32)
    for d in range(nd):
        n = shape[d]
        ramp = np.ones(n, dtype=np.float32)
        f = int(min(fade, n // 2))
        if f > 0:
            w = np.linspace(0.0, 1.0, f, dtype=np.float32)  # w[0]=0
            ramp[:f] = w
            ramp[-f:] = w[::-1]
        sh = [1] * nd; sh[d] = n
        mask *= ramp.reshape(sh)
    return mask


# ----------------------- transform -----------------------

class DeformAugTransform(BasicTransform):
    # Deformacao LOCAL ao tumor (tumor_elastic | tumor_scale), restrita ao
    # bounding-box do tumor + margem. Imagem e mascara recebem o MESMO campo.
    # Fora do bbox nada muda (o peso do tumor ja zera o deslocamento la),
    # entao recortar e exatamente equivalente a warpar o patch inteiro.
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self._rng = None

    @property
    def rng(self):
        # lazy: 1 rng por worker (entropia distinta apos o fork; nao recria por amostra)
        if self._rng is None:
            self._rng = np.random.default_rng()
        return self._rng

    def get_parameters(self, **data_dict):
        seg = data_dict["segmentation"]
        spatial = tuple(int(x) for x in seg.shape[1:])
        c = self.cfg
        rng = self.rng

        seg_bin = (seg[0].detach().cpu().numpy() > 0)
        if seg_bin.sum() < c["min_tumor_voxels"] or rng.random() >= c["p_geom"]:
            return {"roi": None, "local_coords": None}

        roi = _tumor_bbox(seg_bin, spatial, c["roi_margin"])
        sl = tuple(slice(a, b) for a, b in roi)
        seg_roi = np.ascontiguousarray(seg_bin[sl])
        shape = seg_roi.shape

        if rng.random() < 0.5:
            disp = _tumor_elastic_disp(shape, seg_roi, c["tumor_elastic_sigma"],
                                       c["tumor_elastic_max_disp"], c["tumor_dilate_iter"],
                                       c["tumor_smooth_sigma"], rng)
        else:
            disp = _tumor_scale_disp(shape, seg_roi, rng.uniform(*c["tumor_scale_range"]),
                                     c["tumor_dilate_iter"], c["tumor_smooth_sigma"])

        disp *= _edge_fade_mask(shape, c["roi_fade"])[None]   # desloc=0 na borda do ROI
        local_coords = _disp_to_coords(disp)
        return {"roi": roi, "local_coords": local_coords}

    def _warp(self, arr, roi, local_coords, order):
        out = arr.copy()
        sl = tuple(slice(a, b) for a, b in roi)
        for ch in range(arr.shape[0]):
            sub = np.ascontiguousarray(arr[(ch, *sl)])
            out[(ch, *sl)] = map_coordinates(sub, local_coords, order=order, mode="nearest")
        return out

    def _apply_to_image(self, img, roi=None, local_coords=None, **params):
        if local_coords is None:
            return img
        arr = img.detach().cpu().numpy()
        out = self._warp(arr, roi, local_coords, order=1)
        return torch.from_numpy(out).to(dtype=img.dtype, device=img.device)

    def _apply_to_segmentation(self, segmentation, roi=None, local_coords=None, **params):
        if local_coords is None:
            return segmentation
        arr = segmentation.detach().cpu().numpy()
        out = self._warp(arr, roi, local_coords, order=0)
        return torch.from_numpy(out).to(dtype=segmentation.dtype, device=segmentation.device)


class DeformAugWrapper(BasicTransform):
    def __init__(self, deform, base):
        super().__init__()
        self.deform = deform
        self.base = base

    def __call__(self, **data_dict):
        data_dict = self.deform(**data_dict)
        return self.base(**data_dict)


_CFG = {
    "p_geom": 0.50,                  # prob. de aplicar deformacao do tumor por amostra
    "min_tumor_voxels": 50,          # abaixo disso, nao deforma
    "roi_margin": 20,                # margem (voxels) alem do bbox do tumor
    "roi_fade": 10,                  # faixa de fade na borda do ROI (>= tumor_elastic_max_disp)
    "tumor_elastic_sigma": 4.0,
    "tumor_elastic_max_disp": 9.0,
    "tumor_dilate_iter": 8,
    "tumor_smooth_sigma": 5.0,
    "tumor_scale_range": (0.65, 1.45),
}


class nnUNetTrainer_deformAug_agressive(nnUNetTrainer):
    @staticmethod
    def get_training_transforms(*args, **kwargs):
        base = nnUNetTrainer.get_training_transforms(*args, **kwargs)
        return DeformAugWrapper(DeformAugTransform(_CFG), base)

