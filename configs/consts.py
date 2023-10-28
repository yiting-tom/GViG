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

TRAIN_SIZE: int = 38_990
TRAIN_SAMPLE_SIZE: int = 1_000
TEST_PUBLIC_SIZE: int = 1_705
TEST_PRIVATE_SIZE: int = 4_504
