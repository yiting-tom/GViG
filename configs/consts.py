from typing import Dict

VG_COLUMNS = ["unique_id", "image_id", "text", "bbox", "image"]
VQA_COLUMNS = ["question_id", "image_id", "question", "answer", "candidate", "image"]

TRAIN_SIZE: int = 38_990
TRAIN_SAMPLE_SIZE: int = 1_000
TEST_PUBLIC_SIZE: int = 1_705
TEST_PRIVATE_SIZE: int = 4_504

VG_EXAMPLE: Dict[str, str] = {
    "image_id": "237367",
    "text": "A woman in a white blouse holding a glass of wine.",
    "bbox": "230.79,121.75,423.66,463.06",
}

VQA_EXAMPLE: Dict[str, str] = {
    "question_id": "79459",
    "image_id": "79459",
    "question": "is this person wearing shorts?",
    "answer": "0.6|!+no",
    "candidate": "house&&short&&...&&sky",
    "image": "9j/4AAQ...1pAz/9k=",
}

VG_EXAMPLE: Dict[str, str] = {
    "unique_id": "79_1",
    "image_id": "237367",
    "text": "A woman in a white blouse holding a glass of wine.",
    "bbox": "230.79,121.75,423.66,463.06",
    "image": "9j/4AAQ...1pAz/9k=",
}
