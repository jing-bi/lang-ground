from .localizer import build_localizer
from .tracker import SAM2StreamPredictor
from .llm import LLM
from .utils import image_w_box, image_w_mask
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

    def localize(self, frame, question, return_type="vis", **kwargs):
        frame = np.array(frame)
        objxbox = self.loc.localize(frame, kwargs.get("threshold", 0.5))
        locobjs = self.llm.answer(question, objxbox.keys())
        locobjs = sorted(locobjs)
        locobjxbox = {k: v for k, v in objxbox.items() if k in locobjs}
        loc_objsxids = [(text, str(idx)) for idx, text in enumerate(locobjs)]

        if return_type == "vis":
            all_box_image = image_w_box(frame, objxbox)
            llm_box_image = image_w_box(frame, locobjxbox)
            return loc_objsxids, all_box_image, llm_box_image
        else:
            return loc_objsxids, objxbox, locobjxbox

    def track(self, frame, question=None, return_type="vis", **kwargs):
        segments = {}
        if question:
            loc_objsxids, _, locobjxbox = self.localize(frame, question, return_type="data", **kwargs)
            self.status = "tracking"
            self.questions.add(question)
            self.tracker.initialize(frame)

            for obj_id, obj in enumerate(locobjxbox):
                for box in locobjxbox[obj]:
                    _, obj_ids, mask_logits = self.tracker.add_new_points_or_box(
                        inference_state=self.tracker.state, frame_idx=0, obj_id=obj_id, box=box.cpu().numpy()
                    )
                    print(f"Adding {obj} to tracking with id {obj_id}")
            segments = {obj_id: (mask_logits > 0.0).cpu().numpy() for obj_id, mask_logits in zip(obj_ids, mask_logits)}
            return loc_objsxids, image_w_mask(frame, segments)
        elif self.status == "tracking":
            obj_ids, mask_logits = self.tracker.track(frame)
            segments = {obj_id: (mask_logits > 0.0).cpu().numpy() for obj_id, mask_logits in zip(obj_ids, mask_logits)}
            try:
                return image_w_mask(frame, segments)
            except:
                return image_w_mask(frame, segments)
        else:
            return frame
