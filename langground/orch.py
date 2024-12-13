from .localizer import OWL
from .llm import LLM
from .utils import image_w_box, yolo_class, image_w_box_cv2
import numpy as np

class LangGround:

    def __init__(self, loc_model="owl", llm_model="Qwen/Qwen2.5-7B-Instruct"):
        if "owl" in loc_model:
            self.loc = OWL()
            self.loc_name = "owl"
        elif "yolo" in loc_model:
            self.loc = YOLO("yolo11m.pt")
            self.loc_name = "yolo"
        self.llm = LLM(llm_model)

    def localize(self, frame, question, **kwargs):
        if self.loc_name == "owl":
            frame = np.array(frame)
            objxbox = self.loc.localize(frame, kwargs.get("threshold", 0.5))
            locobjs = self.llm.answer(question, objxbox.keys())
            locobjxbox = {k:v for k,v in objxbox.items() if k in locobjs}
            all_box_image = image_w_box(frame,objxbox)
            llm_box_image = image_w_box(frame,locobjxbox)
            return ", ".join(locobjs), all_box_image, llm_box_image
        elif self.loc_name == "yolo":
            frame = np.array(frame)
            result = self.loc(frame, conf=kwargs.get("threshold", 0.5))[0]
            boxes = result.boxes
            bbox_ids = boxes.cls.cpu().numpy().astype(int) 
            boxes_xyxy = boxes.xyxy.cpu().numpy()
            yolo_detected_classes = [yolo_class.get(cls_id, "unknown") for cls_id in bbox_ids]
            labels_bboxes = {label: [box] for label, box in zip(yolo_detected_classes, boxes_xyxy)}
            llm_objs = self.llm.answer(question, yolo_detected_classes)
            llm_labels_bboxes = {label: box for label, box in labels_bboxes.items() if label in llm_objs}
            all_box_image = image_w_box_cv2(frame,labels_bboxes)
            llm_box_image = image_w_box_cv2(frame,llm_labels_bboxes)
            return llm_objs, all_box_image, llm_box_image
