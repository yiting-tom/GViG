from pathlib import Path

ROOT: Path = Path(__file__).parent.parent
DATASETS_DIR: Path = ROOT / "datasets"
IMAGESS_DIR: Path = ROOT / "datasets" / "images"
RESULTS_DIR: Path = ROOT / "results"

# The directory where the WSDM dataset is stored
WSDM_CSV_DIR: Path = DATASETS_DIR / "official"
WSDM_IMAGES_DIR: Path = DATASETS_DIR / "images"

# The data provided by WSDM
TRAIN_CSV: Path = WSDM_CSV_DIR / "train.csv"
TRAIN_SAMPLE_CSV: Path = WSDM_CSV_DIR / "train_sample.csv"
TEST_PUBLIC_CSV: Path = WSDM_CSV_DIR / "test_public.csv"
TEST_PRIVATE_CSV: Path = WSDM_CSV_DIR / "test_private.csv"

# The images provided by WSDM
TRAIN_IMAGES_DIR: Path = WSDM_IMAGES_DIR / "train"
TRAIN_SAMPLE_IMAGES_DIR: Path = WSDM_IMAGES_DIR / "train_sample"
TEST_PUBLIC_IMAGES_DIR: Path = WSDM_IMAGES_DIR / "test_public"
TEST_PRIVATE_IMAGES_DIR: Path = WSDM_IMAGES_DIR / "test_private"
