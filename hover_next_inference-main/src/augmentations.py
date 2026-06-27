import torch
import numpy as np
from torchvision.transforms.transforms import ColorJitter, RandomApply, GaussianBlur

# ── IHC Stain Matrix (Hematoxylin + DAB) ──────────────────────────────────
# Kaynak: Ruifrok & Johnston 2001
rgb_from_hd = np.array(
    [
        [0.650, 0.704, 0.286],   # Hematoxylin
        [0.268, 0.570, 0.776],   # DAB
        [0.700, 0.423, 0.576],   # Residual
    ],
    dtype=np.float32,
)
hd_from_rgb = np.linalg.inv(rgb_from_hd)

# Geriye dönük uyumluluk alias'ları
rgb_from_hed = rgb_from_hd
hed_from_rgb = hd_from_rgb


# ── Stain Space Dönüşümleri ────────────────────────────────────────────────

def torch_rgb2hd(img: torch.Tensor, hd_t: torch.Tensor, e: float):
    """RGB tensörü → HD (Hematoxylin-DAB) stain uzayına çevir."""
    img = img.movedim(-3, -1)
    img = torch.clamp(img, min=e)
    img = torch.log(img) / torch.log(e)
    img = torch.matmul(img, hd_t)
    return img.movedim(-1, -3)


def torch_hd2rgb(img: torch.Tensor, rgb_t: torch.Tensor, e: float):
    """HD stain uzayından RGB tensörüne çevir."""
    e = -torch.log(e)
    img = img.movedim(-3, -1)
    img = torch.matmul(-(img * e), rgb_t)
    img = torch.exp(img)
    img = torch.clamp(img, 0, 1)
    return img.movedim(-1, -3)


# Geriye dönük uyumluluk
torch_rgb2hed = torch_rgb2hd
torch_hed2rgb = torch_hd2rgb


class Hd2Rgb(torch.nn.Module):
    def __init__(self, rank):
        super().__init__()
        self.e = torch.tensor(1e-6).to(rank)
        self.rgb_t = torch.from_numpy(rgb_from_hd).to(rank)
        self.rank = rank

    def forward(self, img):
        return torch_hd2rgb(img, self.rgb_t, self.e)


class Rgb2Hd(torch.nn.Module):
    def __init__(self, rank):
        super().__init__()
        self.e = torch.tensor(1e-6).to(rank)
        self.hd_t = torch.from_numpy(hd_from_rgb).to(rank)
        self.rank = rank

    def forward(self, img):
        return torch_rgb2hd(img, self.hd_t, self.e)


# Geriye dönük uyumluluk alias'ları
Hed2Rgb = Hd2Rgb
Rgb2Hed = Rgb2Hd


# ── IHC Stain Augmentation ─────────────────────────────────────────────────

class IHCNormalizeTorch(torch.nn.Module):
    """
    IHC-spesifik stain augmentation.

    Hematoxylin ve DAB kanalları bağımsız varyasyona uğratılır.
    Test-time augmentation (TTA) için train=True ile çağrılabilir.

    Parameters
    ----------
    sigma_h : float
        Hematoxylin kanalı ölçek varyasyonu
    sigma_dab : float
        DAB kanalı ölçek varyasyonu (daha geniş — marker expression değişimi)
    bias_h : float
        Hematoxylin bias varyasyonu
    bias_dab : float
        DAB bias varyasyonu
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
        sigma_r   = torch.empty(B).uniform_(-self.sigma_h,   self.sigma_h).to(self.rank)
        bias_r    = torch.empty(B).uniform_(-self.bias_h,    self.bias_h).to(self.rank)
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
HedNormalizeTorch = IHCNormalizeTorch


class GaussianNoise(torch.nn.Module):
    """
    Parameters
    ----------
    sigma : float
    rank : str or int or torch.device
    """

    def __init__(self, sigma, rank):
        super().__init__()
        self.sigma = sigma
        self.rank = rank

    def forward(self, img):
        noise = torch.empty(img.shape).uniform_(-self.sigma, self.sigma).to(self.rank)
        return img + noise


# ── Inference Augmentation Pipeline ───────────────────────────────────────

def color_augmentations(train, sigma_h=0.05, sigma_dab=0.15,
                        bias_h=0.03, bias_dab=0.08, s=0.15, rank=0,
                        sigma=None, bias=None):
    """
    IHC inference augmentation pipeline.

    train=True → TTA (test-time augmentation) için daha fazla varyasyon.
    train=False → augmentation yok (saf inference).

    Parameters
    ----------
    train : bool
        TTA modunda True; standart inference'ta False
    sigma_h, sigma_dab, bias_h, bias_dab : float
        IHC stain augmentation parametreleri
    s : float
        ColorJitter ölçeği (düşük tutulur)
    rank : int or str or torch.device
    sigma, bias : float, optional
        Eski API uyumluluğu — verilirse her iki kanal için kullanılır
    """
    if sigma is not None:
        sigma_h = sigma_dab = sigma
    if bias is not None:
        bias_h = bias_dab = bias

    if train:
        color_jitter = ColorJitter(
            brightness=0.8 * s,
            contrast=0.8 * s,
            saturation=0.0,   # IHC'de saturation değiştirme
            hue=0.03,
        )
        data_transforms = torch.nn.Sequential(
            RandomApply([IHCNormalizeTorch(sigma_h, sigma_dab, bias_h, bias_dab, rank)], p=0.85),
            RandomApply([color_jitter], p=0.3),
            RandomApply([GaussianNoise(0.02, rank)], p=0.3),
            RandomApply([GaussianBlur(kernel_size=15, sigma=(0.1, 1.5))], p=0.2),
        )
    else:
        # Inference: augmentation uygulanmaz
        data_transforms = torch.nn.Sequential(torch.nn.Identity())
    return data_transforms
