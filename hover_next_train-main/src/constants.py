import numpy as np

# ── IHC Stain Matrix (Hematoxylin + DAB, H&E yok) ──────────────────────────
# Kaynak: Ruifrok & Johnston 2001, standart IHC dekonvolüsyon vektörleri
RGB_FROM_HD = np.array(
    [
        [0.650, 0.704, 0.286],   # Hematoxylin (mavi counterstain)
        [0.268, 0.570, 0.776],   # DAB (kahverengi, marker pozitifliği)
        [0.700, 0.423, 0.576],   # Residual (HD'ye ortogonal)
    ],
    dtype=np.float32,
)
HD_FROM_RGB = np.linalg.inv(RGB_FROM_HD)

# Geriye dönük uyumluluk için alias
RGB_FROM_HED = RGB_FROM_HD
HED_FROM_RGB = HD_FROM_RGB

# ── IHC Marker Tanımları ───────────────────────────────────────────────────
IHC_MARKERS = ["ER", "PR", "HER2", "Ki67"]           # klasör adlarıyla birebir
MARKER_TO_ID = {m: i for i, m in enumerate(IHC_MARKERS)}  # {"ER":0, "PR":1, ...}

# ── PanNuke ────────────────────────────────────────────────────────────────
PANNUKE_FOLDS = [[1, 2], [0, 2], [1, 0]]
PANNUKE_TISSUES = [
    "Adrenal_gland",
    "Bile-duct",
    "Bladder",
    "Breast",
    "Cervix",
    "Colon",
    "Esophagus",
    "HeadNeck",
    "Kidney",
    "Liver",
    "Lung",
    "Ovarian",
    "Pancreatic",
    "Prostate",
    "Skin",
    "Stomach",
    "Testis",
    "Thyroid",
    "Uterus",
]

# ── Class Names ────────────────────────────────────────────────────────────
# IHC için hücre tipi değil pozitiflik sınıflandırması
CLASS_NAMES_IHC = [
    "positive",    # DAB pozitif çekirdek / membran (ER+, PR+, HER2+, Ki67+)
    "negative",    # boyasız çekirdek
    "background",  # hücre dışı alan
    "dead",        # nekrotik hücre
    "stromal",     # stromal / bağ doku hücresi
]

# Geriye dönük uyumluluk
CLASS_NAMES_PANNUKE = CLASS_NAMES_IHC
CLASS_NAMES = [
    "neutrophil",
    "epithelial-cell",
    "lymphocyte",
    "plasma-cell",
    "eosinophil",
    "connective-tissue-cell",
    "mitosis",
]

# ── Eşik Değerleri ─────────────────────────────────────────────────────────
BEST_MIN_THRESHS = [30, 30, 20, 20, 30, 30, 15]
BEST_MAX_THRESHS = [5000, 5000, 5000, 5000, 5000, 5000, 5000]

MIN_THRESHS_PANNUKE = [10, 10, 10, 10, 10]
MAX_THRESHS_PANNUKE = [20000, 20000, 20000, 3000, 10000]
