import os
from pathlib import Path

from huggingface_hub import hf_hub_download
from PIL import Image
from ultralytics import YOLO

HF_REPO_ID = "do1ng/few_shot2"
HF_FILENAME = "best.pt"

CACHE_DIR = Path(os.getenv("YOLO_MODEL_DIR", Path(__file__).resolve().parent.parent / "models"))

DEFAULT_CONF = 0.25
DEFAULT_IOU = 0.45


class YOLOPredictor:
    def __init__(self):
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        model_path = hf_hub_download(repo_id=HF_REPO_ID, filename=HF_FILENAME, local_dir=str(CACHE_DIR))
        self.model = YOLO(model_path)

    def predict(self, image: Image.Image) -> list[dict]:
        results = self.model.predict(image, conf=DEFAULT_CONF, iou=DEFAULT_IOU, verbose=False)
        result = results[0]

        detections = []
        for box in result.boxes:
            detections.append(
                {
                    "class_id": int(box.cls),
                    "class_name": result.names[int(box.cls)],
                    "confidence": round(float(box.conf), 4),
                    "bbox": {
                        "x1": round(float(box.xyxy[0][0]), 2),
                        "y1": round(float(box.xyxy[0][1]), 2),
                        "x2": round(float(box.xyxy[0][2]), 2),
                        "y2": round(float(box.xyxy[0][3]), 2),
                    },
                }
            )
        return detections
