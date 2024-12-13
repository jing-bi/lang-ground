from .localizer import OWL
from .llm import LLM
from .utils import image_w_box
import numpy as np
class LangGround:

    def __init__(self):
        self.loc = OWL()
        self.llm = LLM()

    def localize(self, frame, question, **kwargs):
        frame = np.array(frame)
        objxbox = self.loc.localize(frame, kwargs.get("threshold", 0.5))
        locobjs = self.llm.answer(question, objxbox.keys())
        locobjxbox = {k:v for k,v in objxbox.items() if k in locobjs}
        all_box_image = image_w_box(frame,objxbox)
        llm_box_image = image_w_box(frame,locobjxbox)
        return ", ".join(locobjs), all_box_image, llm_box_image
