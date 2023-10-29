OFFICIAL_COLUMNS = [
    "image",
    "width",
    "height",
    "left",
    "top",
    "right",
    "bottom",
    "question",
]

VQA_PREDICT_COLUMNS = ["id", "answer"]

OFA_ARCH_ENC_LAYERS_MAPPING = {
    "tiny": 4,
    "medium": 4,
    "base": 6,
    "large": 12,
    "huge": 24,
}

OFA_ARCH_HIDDEN_SIZE_MAPPING = {
    "tiny": 256,
    "medium": 512,
    "base": 768,
    "large": 1024,
    "huge": 1280,
}

TRAIN_SIZE: int = 38_990
TRAIN_SAMPLE_SIZE: int = 1_000
TEST_PUBLIC_SIZE: int = 1_705
TEST_PRIVATE_SIZE: int = 4_504
