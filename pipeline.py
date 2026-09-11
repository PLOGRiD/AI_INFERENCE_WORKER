from PIL import Image


class PipelineError(Exception):
    def __init__(self, message: str, status_code: int = 400):
        super().__init__(message)
        self.status_code = status_code


def classify_material(image: Image.Image, yolo_predictor) -> dict:
    """이미지로 YOLO 객체 탐지를 수행하고 최고 신뢰도 검출 클래스를 최종 결과로 확정한다."""
    detections = yolo_predictor.predict(image)
    if not detections:
        raise PipelineError("탐지된 객체가 없음", status_code=422)

    top_detection = max(detections, key=lambda d: d["confidence"])
    yolo_class = top_detection["class_name"]

    return {
        "final_label": yolo_class,
        "material_verified": False,
        "label_overridden": False,
        "yolo_class": yolo_class,
        "yolo_confidence": top_detection["confidence"],
    }
