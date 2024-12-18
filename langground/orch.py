import threading
from .localizer import build_localizer
from .tracker import SAM2StreamPredictor
from .llm import LLM
from .utils import image_w_box, image_w_mask
import numpy as np
import concurrent.futures

class LangGround:

    def __init__(self, loc="owl", llm="Qwen/Qwen2.5-7B-Instruct", track=True):
        self.llm = LLM(llm)
        self.loc = build_localizer(loc)
        self.tracker = None
        if track == "sam2":
            print("Tracking enabled")
            self.tracker = SAM2StreamPredictor.from_pretrained("facebook/sam2.1-hiera-small")
            self.status = None
            self.video_segments = {}
            self.questions = set()
            self.global_boxes = []
            self.obj_to_question = {}

        self.obj_latest = []
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)

    def select(self, question, objs, block=False):
        def _select():
            return self.llm.answer(question, objs)
        if block:
            self.obj_latest = _select()
        elif not hasattr(self, "future") or self.future.done():
            self.future = self.executor.submit(_select)
            self.obj_latest = self.future.result()

        return self.obj_latest

    def localize(self, frame, question, return_type="vis", block=True, **kwargs):
        frame = np.array(frame)
        objxbox = self.loc.localize(frame, kwargs.get("threshold", 0.5))
        locobjs = self.select(question, objxbox.keys(), block=block)
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
            return image_w_mask(frame, segments)
        else:
            return frame
