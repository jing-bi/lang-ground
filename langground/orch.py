from .localizer import build_localizer
from .tracker import SAM2StreamPredictor
from .llm import LLM
from .utils import image_w_box, anno_frame
import numpy as np

class LangGround:

    def __init__(self, loc_model="owl", llm_model="Qwen/Qwen2.5-7B-Instruct", track=False):

        self.loc = build_localizer(loc_model)
        self.llm = LLM(llm_model)
        self.tracker = None
        if track:
            print("Tracking enabled")
            self.tracker = SAM2StreamPredictor.from_pretrained("facebook/sam2.1-hiera-small")
            self.status = None
            self.video_segments = {}
            self.questions = set()
            self.global_boxes = []
            self.obj_to_question = {}

    def localize(self, frame, question, **kwargs):
        frame = np.array(frame)
        objxbox = self.loc.localize(frame, kwargs.get("threshold", 0.5))
        locobjs = self.llm.answer(question, objxbox.keys())
        locobjxbox = {k: v for k, v in objxbox.items() if k in locobjs}
        all_box_image = image_w_box(frame, objxbox)
        llm_box_image = image_w_box(frame, locobjxbox)
        texts = [(text, str(idx)) for idx, text in enumerate(locobjs)]
        if self.tracker is not None:
            self.status = "tracking"
            self.questions.add(question)
            self.tracker.initialize(frame)
            qidx = len(self.questions)
            for obj in locobjs:
                for box in objxbox[obj]:
                    self.global_boxes.append(box)
                    obj_id = len(self.global_boxes)
                    frame_idx, obj_ids, video_res_masks = self.tracker.add_new_points_or_box(
                        inference_state=self.tracker.state, frame_idx=0, obj_id=obj_id, box=box.cpu().numpy()
                    )
                    print(f"Adding {obj} to tracking with id {obj_id}")
                    self.obj_to_question[obj_id] = qidx
        return texts, all_box_image, llm_box_image

    def track(self, fidx, frame, question, **kwargs):
        if self.status == "tracking":
            obj_ids, mask_logits = self.tracker.track(frame)
            self.video_segments[fidx] = {obj_id: (mask_logits > 0.0).cpu().numpy() for obj_id, mask_logits in zip(obj_ids, mask_logits)}

        segments = self.video_segments.get(fidx, {})
        return anno_frame(frame, segments, question, self.obj_to_question)
