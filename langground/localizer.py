from collections import defaultdict
from pathlib import Path
import torch
from transformers import Owlv2Processor, Owlv2ForObjectDetection
from PIL import Image
from pathlib import Path
from ultralytics import YOLO


def build_localizer(model_name):
    if "owl" in model_name :
        return OWL()
    elif "yolo" in model_name :
        return YOLO11(model_name)
    else:
        raise ValueError(f"Unknown model name: {model_name}")


class OWL:

    def __init__(self):
        model_name = "google/owlv2-large-patch14-ensemble"
        self.processor = Owlv2Processor.from_pretrained(model_name)
        self.model = Owlv2ForObjectDetection.from_pretrained(model_name).to("cuda")
        self.model.eval()
        self.objects_f = Path(__file__).parent / "objs" / "owl.txt"
        self.objects = [line.strip() for line in self.objects_f.open().readlines()]
        self.device = "cuda"

    def __call__(self, image, threshold=0.5):
        return self.localize(image, threshold=threshold)

    def localize(self, image, threshold=0.5):
        image = Image.fromarray(image)
        final = defaultdict(list)
        with torch.inference_mode():
            inputs = self.processor(text=self.objects, images=[image], return_tensors="pt").to(self.device)
            outputs = self.model(**inputs)
            target_sizes = torch.Tensor([image.size[::-1]]).to(self.device)
            result = self.processor.post_process_object_detection(outputs=outputs, target_sizes=target_sizes, threshold=threshold)[0]

            boxes, scores, labels = result["boxes"], result["scores"], result["labels"]
            for box, score, label in zip(boxes, scores, labels):
                final[self.objects[label]].append(box)
        return final


class YOLO11:
    def __init__(self, model_name):
        model_name = model_name + ".pt"
        self.model = YOLO(model_name)
        self.objects_f = Path(__file__).parent / "objs" / "yolo.txt"
        self.objects = [line.strip() for line in self.objects_f.open().readlines()]

    def localize(self, image, threshold=0.5):
        result = self.model(image, conf=threshold)[0]
        boxes = result.boxes
        bbox_ids = boxes.cls.cpu().numpy().astype(int)
        boxes_xyxy = boxes.xyxy.cpu().numpy()
        final = defaultdict(list)
        for label, box in zip(bbox_ids, boxes_xyxy):
            final[self.objects[label]].append(box)
        return final
