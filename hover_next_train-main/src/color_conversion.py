import torch
import numpy as np
from src.constants import RGB_FROM_HD, HD_FROM_RGB
from torchvision.transforms.transforms import (
    ColorJitter,
    RandomApply,
    GaussianBlur,
    Normalize,
)


# ── Stain Space Dönüşümleri ────────────────────────────────────────────────

def torch_rgb2hd(img, hd_t, e):
    """RGB tensörü → HD (Hematoxylin-DAB) stain uzayına çevir."""
    img = img.movedim(-3, -1)
    img = torch.clamp(img, min=e)
    img = torch.log(img) / torch.log(e)
    img = torch.matmul(img, hd_t)
    return img.movedim(-1, -3)


def torch_hd2rgb(img, rgb_t, e):
    """HD stain uzayından RGB tensörüne çevir."""
    e = -torch.log(e)
    img = img.movedim(-3, -1)
    img = torch.matmul(-(img * e), rgb_t)
    img = torch.exp(img)
    img = torch.clamp(img, 0, 1)
    return img.movedim(-1, -3)


# Geriye dönük uyumluluk alias'ları
torch_rgb2hed = torch_rgb2hd
torch_hed2rgb = torch_hd2rgb


class Hd2Rgb(torch.nn.Module):
    def __init__(self, rank):
        super().__init__()
        self.e = torch.tensor(1e-6).to(rank)
        self.rgb_t = torch.from_numpy(RGB_FROM_HD).to(rank)

    def forward(self, img):
        return torch_hd2rgb(img, self.rgb_t, self.e)


class Rgb2Hd(torch.nn.Module):
    def __init__(self, rank):
        super().__init__()
        self.e = torch.tensor(1e-6).to(rank)
        self.hd_t = torch.from_numpy(HD_FROM_RGB).to(rank)

    def forward(self, img):
        return torch_rgb2hd(img, self.hd_t, self.e)


# Geriye dönük uyumluluk alias'ları
Hed2Rgb = Hd2Rgb
Rgb2Hed = Rgb2Hd


# ── IHC Stain Augmentation ─────────────────────────────────────────────────

class IHCNormalizeTorch(torch.nn.Module):
    """
    IHC-spesifik stain augmentation.

    Hematoxylin (kanal 0) ve DAB (kanal 1) kanalları bağımsız olarak
    varyasyona uğratılır. DAB yoğunluğu marker expression seviyesini
    yansıttığından daha geniş bir sigma ile augmente edilir.

    Parameters
    ----------
    sigma_h : float
        Hematoxylin kanalı ölçek varyasyonu (önerilen: 0.05)
    sigma_dab : float
        DAB kanalı ölçek varyasyonu (önerilen: 0.15 — daha geniş)
    bias_h : float
        Hematoxylin kanalı bias varyasyonu (önerilen: 0.03)
    bias_dab : float
        DAB kanalı bias varyasyonu (önerilen: 0.08)
    rank : int or str or torch.device
    """

    def __init__(self, sigma_h=0.05, sigma_dab=0.15,
                 bias_h=0.03, bias_dab=0.08, rank=0):
        super().__init__()
        self.sigma_h = sigma_h
        self.sigma_dab = sigma_dab
        self.bias_h = bias_h
        self.bias_dab = bias_dab
        self.rank = rank
        self.rgb2hd = Rgb2Hd(rank=rank)
        self.hd2rgb = Hd2Rgb(rank=rank)

    def _augment_channels(self, hd):
        B = hd.shape[0]

        sigma_h   = torch.empty(B).uniform_(-self.sigma_h,   self.sigma_h).to(self.rank)
        bias_h    = torch.empty(B).uniform_(-self.bias_h,    self.bias_h).to(self.rank)
        sigma_dab = torch.empty(B).uniform_(-self.sigma_dab, self.sigma_dab).to(self.rank)
        bias_dab  = torch.empty(B).uniform_(-self.bias_dab,  self.bias_dab).to(self.rank)
        # Residual kanal: H kadar varyasyon
        sigma_r = torch.empty(B).uniform_(-self.sigma_h, self.sigma_h).to(self.rank)
        bias_r  = torch.empty(B).uniform_(-self.bias_h,  self.bias_h).to(self.rank)

        sigmas = torch.stack([sigma_h, sigma_dab, sigma_r], dim=1)
        biases = torch.stack([bias_h,  bias_dab,  bias_r],  dim=1)

        return (hd * (1 + sigmas[..., None, None])) + biases[..., None, None]

    def forward(self, img):
        if img.dim() == 3:
            img = img.unsqueeze(0)
        hd = self.rgb2hd(img)
        hd = self._augment_channels(hd)
        return self.hd2rgb(hd)


# Geriye dönük uyumluluk alias'ı
HED_normalize_torch = IHCNormalizeTorch


class GaussianNoise(torch.nn.Module):
    def __init__(self, sigma, rank):
        super().__init__()
        self.sigma = sigma
        self.rank = rank

    def forward(self, img):
        noise = torch.empty(img.shape).uniform_(-self.sigma, self.sigma).to(self.rank)
        return img + noise


class ScannerBrightnessShift(torch.nn.Module):
    """
    Farklı WSI tarayıcıların global parlaklık farkını simüle eder.
    IHC görüntülerde tarayıcı bazında önemli renk kayması olabilir.
    """
    def __init__(self, max_shift=0.08, rank=0):
        super().__init__()
        self.max_shift = max_shift
        self.rank = rank

    def forward(self, img):
        if img.dim() == 3:
            img = img.unsqueeze(0)
        B = img.shape[0]
        shift = torch.empty(B, 3, 1, 1).uniform_(-self.max_shift, self.max_shift).to(self.rank)
        return torch.clamp(img + shift, 0, 1)


# ── Ana Augmentation Pipeline ──────────────────────────────────────────────

def color_augmentations(train, sigma_h=0.05, sigma_dab=0.15,
                        bias_h=0.03, bias_dab=0.08, s=0.15, rank=0,
                        # geriye dönük uyumluluk için eski parametreler
                        sigma=None, bias=None):
    """
    IHC-spesifik renk augmentation pipeline.

    Eğitimde:
      - IHC stain augmentation (H ve DAB bağımsız varyasyon)
      - Kısıtlı ColorJitter (saturation=0, hue=0.03)
      - Gaussian noise
      - Gaussian blur (scanner optik artefakt)
      - Scanner bazlı global parlaklık kayması

    Validasyonda:
      - Augmentation uygulanmaz (kimlik dönüşümü)
    """
    # Eski API uyumluluğu: sigma/bias verilmişse her iki kanal için kullan
    if sigma is not None:
        sigma_h = sigma_dab = sigma
    if bias is not None:
        bias_h = bias_dab = bias

    if train:
        color_jitter = ColorJitter(
            brightness=0.8 * s,
            contrast=0.8 * s,
            saturation=0.0,   # IHC'de saturation değiştirme — DAB rengi biyolojik sinyal
            hue=0.03,         # çok minimal hue
        )
        data_transforms = torch.nn.Sequential(
            RandomApply([IHCNormalizeTorch(sigma_h, sigma_dab, bias_h, bias_dab, rank)], p=0.85),
            RandomApply([color_jitter], p=0.3),
            RandomApply([GaussianNoise(0.03, rank)], p=0.3),
            RandomApply([GaussianBlur(kernel_size=15, sigma=(0.1, 1.5))], p=0.2),
            RandomApply([ScannerBrightnessShift(max_shift=0.08, rank=rank)], p=0.25),
        )
    else:
        data_transforms = torch.nn.Sequential(torch.nn.Identity())
    return data_transforms


def get_normalize(use_norm=True):
    """
    IHC (DAB+Hematoxylin) için normalize parametreleri.
    ImageNet değerleri yerine IHC doku istatistikleri kullanılır.
    """
    if use_norm:
        return Normalize(
            mean=[0.740, 0.513, 0.670],
            std=[0.171, 0.177, 0.133],
        )
    else:
        return lambda x: x
