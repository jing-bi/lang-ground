from .localizer import build_localizer
from .llm import LLM
from .utils import image_w_box
import numpy as np

class LangGround:

    def __init__(self, loc_model="owl", llm_model="Qwen/Qwen2.5-7B-Instruct"):

        self.loc = build_localizer(loc_model)
        self.llm = LLM(llm_model)

    def localize(self, frame, question, **kwargs):

        frame = np.array(frame)
        objxbox = self.loc.localize(frame, kwargs.get("threshold", 0.5))
        locobjs = self.llm.answer(question, objxbox.keys())
        locobjxbox = {k: v for k, v in objxbox.items() if k in locobjs}
        all_box_image = image_w_box(frame, objxbox)
        llm_box_image = image_w_box(frame, locobjxbox)
        texts = [(text, str(idx)) for idx, text in enumerate(locobjs)]
        return texts, all_box_image, llm_box_image
