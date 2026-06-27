from torch.utils.data import Dataset
import numpy as np
import torch
from typing import Optional, List, Tuple, Callable
from torch.utils.data import Dataset
from tqdm import tqdm
import mahotas as mh
import os
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from torch.utils.data.distributed import DistributedSampler
from src.constants import (
    PANNUKE_FOLDS,
    CLASS_NAMES,
    CLASS_NAMES_IHC,
    IHC_MARKERS,
    MARKER_TO_ID,
)


def make_cpvs(gt_inst):
    device = gt_inst.device
    gt_inst = gt_inst.squeeze()
    cpvs = torch.zeros((2,) + gt_inst.shape, dtype=torch.float).to(device)
    ind_x, ind_y = gt_inst.nonzero(as_tuple=True)
    val = gt_inst[ind_x, ind_y]
    labels = val.unique()
    for label in labels:
        sel = val == label
        x = ind_x[sel]
        y = ind_y[sel]
        cpvs[0, x, y] = -x + x.float().mean()
        cpvs[1, x, y] = -y + y.float().mean()
    return cpvs.unsqueeze(0)


@torch.jit.script
def jit_cpvs(gt_inst):
    device = gt_inst.device
    gt_inst = gt_inst.squeeze()
    cpvs = torch.zeros((2,) + gt_inst.shape, dtype=torch.float, device=device)
    ind = gt_inst.nonzero().T
    val = gt_inst[ind[0], ind[1]]
    labels = torch.unique(val)
    for label in labels:
        sel = val == label
        x = ind[0][sel]
        y = ind[1][sel]
        cpvs[0, x, y] = -x + x.float().mean()
        cpvs[1, x, y] = -y + y.float().mean()
    return cpvs.unsqueeze(0)


@torch.jit.script
def parallel_cpvs(gt_inst):
    futures: List[torch.jit.Future[torch.Tensor]] = []
    for i in range(gt_inst.shape[0]):
        futures.append(torch.jit.fork(jit_cpvs, gt_inst[i]))
    results = []
    for future in futures:
        results.append(torch.jit.wait(future))
    return torch.cat(results, dim=0)


def normalize_percentile(
    x, pmin=3, pmax=99.8, axis=None, clip=False, eps=1e-8, dtype=np.float32
):
    mi = np.percentile(x, pmin, axis=axis, keepdims=True)
    ma = np.percentile(x, pmax, axis=axis, keepdims=True)
    return normalize_min_max(x, mi, ma, clip=clip, eps=eps, dtype=dtype)


def normalize_min_max(x, mi, ma, clip=False, eps=1e-20, dtype=np.float32):
    if mi is None:
        mi = np.min(x)
    if ma is None:
        ma = np.max(x)
    if dtype is not None:
        x = x.astype(dtype, copy=False)
        mi = dtype(mi) if np.isscalar(mi) else mi.astype(dtype, copy=False)
        ma = dtype(ma) if np.isscalar(ma) else ma.astype(dtype, copy=False)
        eps = dtype(eps)
    x = (x - mi) / (ma - mi + eps)
    if clip:
        x = np.clip(x, 0, 1)
    return x


# ── Dataset Sınıfları ──────────────────────────────────────────────────────

class SliceDataset(Dataset):
    """Orijinal PanNuke numpy array dataset'i (geriye dönük uyumluluk)."""

    def __init__(self, raw, labels, norm=True):
        self.raw = raw
        self.labels = labels
        self.norm = norm

    def __len__(self):
        return self.raw.shape[0]

    def __getitem__(self, idx):
        raw_tmp = self.raw[idx].astype(np.float32)
        if self.norm:
            raw_tmp = normalize_min_max(raw_tmp, 0, 255)
        if self.labels is not None:
            return raw_tmp, self.labels[idx].astype(np.float32)
        else:
            return raw_tmp, False


class IHCSliceDataset(Dataset):
    """
    IHC PanNuke dataset'i — UNIStainNet/MIST çıktı klasör yapısından beslenir.

    Her sample ile birlikte marker_id döndürür:
        ER=0, PR=1, HER2=2, Ki67=3

    Parameters
    ----------
    images : np.ndarray  [N, H, W, 3]
    labels : np.ndarray  [N, H, W, C]  PanNuke mask formatı
    marker_ids : np.ndarray  [N]
    norm : bool
    """

    def __init__(self, images, labels, marker_ids, norm=True):
        self.images = images
        self.labels = labels
        self.marker_ids = marker_ids
        self.norm = norm

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx].astype(np.float32)
        if self.norm:
            img = normalize_min_max(img, 0, 255)
        marker_id = int(self.marker_ids[idx])
        if self.labels is not None:
            return img, self.labels[idx].astype(np.float32), marker_id
        return img, False, marker_id


class GaussianNoise(torch.nn.Module):
    def __init__(self, sigma, rank):
        super().__init__()
        self.sigma = sigma
        self.rank = rank

    def forward(self, img):
        noise = torch.randn(img.shape).to(self.rank) * self.sigma
        return img + noise


def center_crop(t, croph, cropw):
    h, w = t.shape[-2:]
    startw = w // 2 - (cropw // 2)
    starth = h // 2 - (croph // 2)
    return t[..., starth: starth + croph, startw: startw + cropw]


def inst_to_3c(gt_labels):
    borders = mh.labeled.borders(gt_labels, Bc=np.ones((3, 3)))
    mask = gt_labels > 0
    return (((borders & mask) * 1) + (mask * 1))[np.newaxis, :]


def add_3c_gt_fast(Y):
    print("adding 3-class ground truth...")
    instances = Y[..., 0]
    gt_3c_list = []
    for inst in tqdm(instances):
        gt_3c = inst_to_3c(inst)
        gt_3c_list.append(gt_3c)
    gt_3c = np.transpose(np.stack(gt_3c_list, 0), [0, 2, 3, 1])
    Y = np.concatenate([Y, gt_3c], -1)
    return Y


# ── IHC Veri Yükleme ───────────────────────────────────────────────────────

def _load_marker_data(data_root, marker, split="train"):
    """
    Tek bir marker için görüntü ve mask yükler.

    Desteklenen klasör yapıları (öncelik sırasıyla):

    1. Düz NPY:
        data_root/<marker>/images.npy
        data_root/<marker>/labels.npy

    2. Split'li NPY:
        data_root/<marker>/images/<split>/images.npy
        data_root/<marker>/masks/<split>/labels.npy

    3. PNG (UNIStainNet/MIST çıktısı):
        data_root/<marker>/images/<split>/*.png
        data_root/<marker>/masks/<split>/labels.npy
    """
    marker_path = os.path.join(data_root, marker)

    # 1. Düz NPY — tüm veri tek dosyada, split burada yapılır (%80 train / %20 val)
    images_npy = os.path.join(marker_path, "images.npy")
    labels_npy = os.path.join(marker_path, "labels.npy")
    if os.path.exists(images_npy) and os.path.exists(labels_npy):
        images = np.load(images_npy, mmap_mode="r")
        labels = np.load(labels_npy, mmap_mode="r")
        n = len(images)
        cut = int(n * 0.8)
        if split == "val" or split == "test":
            return images[cut:], labels[cut:]
        else:  # train
            return images[:cut], labels[:cut]

    # 2. Split'li NPY
    images_split = os.path.join(marker_path, "images", split, "images.npy")
    labels_split = os.path.join(marker_path, "masks",  split, "labels.npy")
    if os.path.exists(images_split) and os.path.exists(labels_split):
        return np.load(images_split, mmap_mode="r"), np.load(labels_split, mmap_mode="r")

    # 3. PNG klasör yapısı
    img_dir = os.path.join(marker_path, "images", split)
    msk_dir = os.path.join(marker_path, "masks",  split)
    if os.path.isdir(img_dir):
        img_files = sorted([
            f for f in os.listdir(img_dir)
            if f.lower().endswith((".png", ".jpg", ".tif"))
        ])
        images = np.stack([
            np.array(Image.open(os.path.join(img_dir, f)).convert("RGB"))
            for f in img_files
        ])
        msk_npy = os.path.join(msk_dir, "labels.npy")
        if os.path.exists(msk_npy):
            labels = np.load(msk_npy, mmap_mode="r")
        else:
            msk_files = sorted([f for f in os.listdir(msk_dir) if f.endswith(".npy")])
            labels = np.stack([np.load(os.path.join(msk_dir, f)) for f in msk_files])
        return images, labels

    raise FileNotFoundError(
        f"'{marker}' için veri bulunamadı: {marker_path}\n"
        f"Beklenen yapı:\n"
        f"  {marker_path}/images.npy + labels.npy\n"
        f"  veya {marker_path}/images/<split>/ + masks/<split>/"
    )


def get_ihc_pannuke(params):
    """
    UNIStainNet/MIST çıktı klasör yapısından IHC eğitim dataseti oluşturur.

    Beklenen klasör yapısı:
        <data_path>/
        ├── ER/
        ├── PR/
        ├── HER2/
        └── Ki67/

    Her marker ayrı yüklenir, marker_id etiketi eklenir,
    tüm markerlar birleştirilir → toplam 4× PanNuke büyüklüğünde dataset.

    Config parametreleri
    --------------------
    data_path : str
    markers : list[str]   (varsayılan: ["ER","PR","HER2","Ki67"])
    train_split : str     (varsayılan: "train")
    val_split : str       (varsayılan: "val")
    batch_size, validation_batch_size, num_workers : int
    use_weighted_sampling : bool
    """
    data_path   = params["data_path"]
    markers     = params.get("markers", IHC_MARKERS)
    train_split = params.get("train_split", "train")
    val_split   = params.get("val_split",   "val")

    print(f"IHC dataset yükleniyor: {markers}")

    # ── Eğitim ────────────────────────────────────────────────────────────
    tr_imgs, tr_lbls, tr_ids = [], [], []
    for marker in markers:
        mid = MARKER_TO_ID[marker]
        try:
            imgs, lbls = _load_marker_data(data_path, marker, split=train_split)
        except FileNotFoundError as e:
            print(f"[UYARI] {e}")
            continue
        lbls = add_3c_gt_fast(lbls)
        tr_imgs.append(imgs)
        tr_lbls.append(lbls)
        tr_ids.append(np.full(len(imgs), mid, dtype=np.int64))
        print(f"  {marker}: {len(imgs)} patch (id={mid})")

    x_train      = np.concatenate(tr_imgs)
    y_train      = np.concatenate(tr_lbls)
    marker_train = np.concatenate(tr_ids)
    print(f"Toplam eğitim: {len(x_train)} patch")

    # ── Validasyon ────────────────────────────────────────────────────────
    va_imgs, va_lbls, va_ids = [], [], []
    for marker in markers:
        mid = MARKER_TO_ID[marker]
        try:
            imgs, lbls = _load_marker_data(data_path, marker, split=val_split)
        except FileNotFoundError:
            continue
        lbls = add_3c_gt_fast(lbls)
        va_imgs.append(imgs)
        va_lbls.append(lbls)
        va_ids.append(np.full(len(imgs), mid, dtype=np.int64))

    x_val      = np.concatenate(va_imgs)
    y_val      = np.concatenate(va_lbls)
    marker_val = np.concatenate(va_ids)
    print(f"Toplam validasyon: {len(x_val)} patch")

    # ── DataLoader ────────────────────────────────────────────────────────
    labeled_dataset    = IHCSliceDataset(x_train, y_train, marker_train)
    validation_dataset = IHCSliceDataset(x_val,   y_val,   marker_val)

    # DÜZELTME: önceden `y_train.shape[-1] - 1` kullanılıyordu, bu Y'nin kanal sayısından
    # (instance, type, 3c => 3) geliyordu ve range(2) = [0,1] veriyordu — yani weighted
    # sampler SADECE class id 0 ve 1'i sayıyor, negative/background/dead/stromal (2,3,4,5)
    # tamamen GÖZ ARDI ediliyordu. Asıl olması gereken: type kanalındaki tüm class id'leri
    # (0..out_channels_cls-1), yani out_channels_cls=6 için [0,1,2,3,4,5].
    classes = list(range(params.get("out_channels_cls", 6)))
    sampler = (
        get_weighted_sampler(labeled_dataset, classes)
        if params.get("use_weighted_sampling", True)
        else None
    )

    labeled_dataloader = DataLoader(
        labeled_dataset,
        batch_size=params["batch_size"],
        prefetch_factor=2,
        sampler=sampler,
        shuffle=(sampler is None),
        num_workers=params["num_workers"],
        pin_memory=True,
    )

    dist_samp = DistributedSampler(validation_dataset, shuffle=True, drop_last=True)
    validation_dataloader = DataLoader(
        validation_dataset,
        sampler=dist_samp,
        batch_size=params["validation_batch_size"],
        num_workers=params["num_workers"],
        pin_memory=True,
    )

    sz = int(len(x_train) / params["batch_size"])
    return [labeled_dataloader], validation_dataloader, sz, dist_samp, CLASS_NAMES_IHC


# ── Orijinal PanNuke H&E (geriye dönük uyumluluk) ─────────────────────────

def get_pannuke(params):
    from src.constants import CLASS_NAMES_PANNUKE

    fold = params["fold"] - 1
    im_folds = [
        np.load(
            os.path.join(params["data_path"], "images", "fold" + str(i), "images.npy"),
            mmap_mode="r",
        )
        for i in range(1, 4)
    ]
    im_types = [
        np.load(
            os.path.join(params["data_path"], "images", "fold" + str(i), "types.npy"),
            mmap_mode="r",
        )
        for i in range(1, 4)
    ]
    gt_folds = [
        np.load(
            os.path.join(params["data_path"], "masks", "fold" + str(i), "labels.npy"),
            mmap_mode="r",
        )
        for i in range(1, 4)
    ]
    val_f, test_f = PANNUKE_FOLDS[fold]
    if params["test_as_val"]:
        x_train = np.concatenate([im_folds[fold], im_folds[val_f]])
        train_types = np.concatenate([im_types[fold], im_types[val_f]])
        y_train = np.concatenate([gt_folds[fold], gt_folds[val_f]])
        x_val = im_folds[test_f]
        y_val = gt_folds[test_f]
    else:
        x_train = im_folds[fold]
        y_train = gt_folds[fold]
        train_types = im_types[fold]
        x_val = im_folds[val_f]
        y_val = gt_folds[val_f]

    labeled_dataset    = SliceDataset(raw=x_train, labels=add_3c_gt_fast(y_train))
    validation_dataset = SliceDataset(raw=x_val,   labels=add_3c_gt_fast(y_val))

    labeled_dataloader = DataLoader(
        labeled_dataset,
        batch_size=params["batch_size"],
        prefetch_factor=2,
        sampler=get_weighted_sampler(labeled_dataset, [0, 1, 2, 3, 4, 5]),
        num_workers=params["num_workers"],
        pin_memory=True,
    )

    dist_samp = DistributedSampler(validation_dataset, shuffle=True, drop_last=True)
    validation_dataloader = DataLoader(
        validation_dataset,
        sampler=dist_samp,
        batch_size=params["validation_batch_size"],
        num_workers=params["num_workers"],
        pin_memory=True,
    )
    sz = int(x_train.shape[0] / params["batch_size"])
    return [labeled_dataloader], validation_dataloader, sz, dist_samp, CLASS_NAMES_PANNUKE


def get_data(params):
    if params["dataset"] == "ihc_pannuke":
        return get_ihc_pannuke(params)
    elif params["dataset"] == "pannuke":
        return get_pannuke(params)
    elif params["dataset"] == "lizard":
        return get_lizard(params)
    else:
        raise NotImplementedError(
            f"Bilinmeyen dataset: '{params['dataset']}'. "
            "Seçenekler: 'ihc_pannuke', 'pannuke', 'lizard'"
        )


def get_lizard(params):
    fold_path = os.path.join(params["data_path_liz"], "fold_" + str(params["fold"]))
    if params["test_as_val"]:
        Liz_X_train = np.concatenate([
            np.load(os.path.join(fold_path, "train_img.npy")),
            np.load(os.path.join(fold_path, "valid_img.npy")),
        ])
        Liz_Y_train = add_3c_gt_fast(np.concatenate([
            np.load(os.path.join(fold_path, "train_lab.npy")),
            np.load(os.path.join(fold_path, "valid_lab.npy")),
        ]))
        Liz_X_val = np.load(os.path.join(params["data_path_liz"], "test_images.npy"))
        Liz_Y_val = add_3c_gt_fast(np.load(os.path.join(params["data_path_liz"], "test_labels.npy")))
    else:
        Liz_X_train = np.load(os.path.join(fold_path, "train_img.npy"))
        Liz_Y_train = add_3c_gt_fast(np.load(os.path.join(fold_path, "train_lab.npy")))
        Liz_X_val   = np.load(os.path.join(fold_path, "valid_img.npy"))
        Liz_Y_val   = add_3c_gt_fast(np.load(os.path.join(fold_path, "valid_lab.npy")))

    Mit_X_train = np.load(os.path.join(params["data_path_mit"], "train_full_img.npy"))
    Mit_X_val   = np.load(os.path.join(params["data_path_mit"], "valid_full_img.npy"))
    Mit_Y_train = add_3c_gt_fast(np.load(os.path.join(params["data_path_mit"], "train_full_lab.npy")))
    Mit_Y_val   = add_3c_gt_fast(np.load(os.path.join(params["data_path_mit"], "valid_full_lab.npy")))

    X_val = np.concatenate([Liz_X_val, Mit_X_val])
    Y_val = np.concatenate([Liz_Y_val, Mit_Y_val])

    labeled_dataset    = SliceDataset(raw=Liz_X_train, labels=Liz_Y_train)
    validation_dataset = SliceDataset(raw=X_val, labels=Y_val)
    mit_dataset        = SliceDataset(raw=Mit_X_train, labels=Mit_Y_train)

    lab_sampler = get_weighted_sampler(labeled_dataset, [0,1,2,3,4,5,6,7]) if params["use_weighted_sampling"] else None
    mit_sampler = get_weighted_sampler(mit_dataset,     [0,1,2,3,4,5,6,7]) if params["use_weighted_sampling"] else None

    labeled_dataloader = DataLoader(
        labeled_dataset, batch_size=params["batch_size"], prefetch_factor=2,
        sampler=lab_sampler, shuffle=(lab_sampler is None),
        num_workers=params["num_workers"], pin_memory=True,
    )
    mit_labeled_dataloader = DataLoader(
        mit_dataset, batch_size=params["batch_size"], prefetch_factor=2,
        sampler=mit_sampler, shuffle=(mit_sampler is None),
        num_workers=params["num_workers"], pin_memory=True,
    )
    dist_samp = DistributedSampler(validation_dataset, shuffle=True, drop_last=True)
    validation_dataloader = DataLoader(
        validation_dataset, sampler=dist_samp,
        batch_size=params["validation_batch_size"],
        num_workers=params["num_workers"], pin_memory=True,
    )
    sz = int(min(Liz_X_train.shape[0], Mit_X_train.shape[0]) / params["batch_size"])
    return [labeled_dataloader, mit_labeled_dataloader], validation_dataloader, sz, dist_samp, CLASS_NAMES


# ── Weighted Sampler ───────────────────────────────────────────────────────

def get_weighted_sampler(ds, classes=None):
    """
    Sınıf dengesizliğini gidermek için ağırlıklı örnekleyici.
    IHC'de pozitif hücre oranı çok düşük olabilir (Ki67 %5 gibi) —
    weighted sampler bu dengesizliği telafi eder.
    """
    if classes is None:
        classes = [0, 1, 2, 3, 4]

    count_list = []
    for sample in ds:
        gt = sample[1] if isinstance(sample, (tuple, list)) else sample
        if hasattr(gt, 'numpy'):
            gt = gt.numpy()
        gt_classes = gt[..., 1].squeeze() if gt is not False else np.zeros(1)
        tmp_list = [np.count_nonzero(gt_classes == c) for c in classes]
        count_list.append(np.stack(tmp_list))

    counts = np.stack(count_list)
    sampling_weights = np.divide(
        counts,
        counts.sum(0)[np.newaxis, ...],
        where=counts.sum(0)[np.newaxis, ...] != 0,
    )
    sampling_weights = sampling_weights.sum(1)
    return torch.utils.data.WeightedRandomSampler(
        torch.from_numpy(sampling_weights),
        num_samples=len(sampling_weights),
        replacement=True,
    )
