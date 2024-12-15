from collections import OrderedDict
from pathlib import Path
import cv2
from hydra import compose
from hydra.utils import instantiate
import numpy as np
from omegaconf import OmegaConf
import torch
from sam2.build_sam import _load_checkpoint, _hf_download
from sam2.sam2_video_predictor import SAM2VideoPredictor
import sys

sys.path.append(str(Path(__file__).parent.resolve()))


def build_sam2_stream_predictor(
    config_file,
    ckpt_path=None,
    device="cuda",
    mode="eval",
    hydra_overrides_extra=[],
    apply_postprocessing=True,
    **kwargs,
):
    hydra_overrides = [
        "++model._target_=tracker.SAM2StreamPredictor",
    ]
    if apply_postprocessing:
        hydra_overrides_extra = hydra_overrides_extra.copy()
        hydra_overrides_extra += [
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_via_stability=true",
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_stability_delta=0.05",
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_stability_thresh=0.98",
            "++model.binarize_mask_from_pts_for_mem_enc=true",
            "++model.fill_hole_area=8",
        ]
    hydra_overrides.extend(hydra_overrides_extra)
    cfg = compose(config_name=config_file, overrides=hydra_overrides)
    OmegaConf.resolve(cfg)
    model = instantiate(cfg.model, _recursive_=True)
    _load_checkpoint(model, ckpt_path)
    model = model.to(device)
    if mode == "eval":
        model.eval()
    return model


def build_sam2_stream_predictor_hf(model_id, **kwargs):
    config_name, ckpt_path = _hf_download(model_id)
    return build_sam2_stream_predictor(config_file=config_name, ckpt_path=ckpt_path, **kwargs)


class SAM2StreamPredictor(SAM2VideoPredictor):
    """
    A streaming predictor that inherits from SAM2VideoPredictor and overrides
    key methods to handle single incoming frames rather than a pre-loaded video.
    """

    def __init__(
        self,
        fill_hole_area=0,
        non_overlap_masks=False,
        clear_non_cond_mem_around_input=False,
        add_all_frames_to_correct_as_cond=False,
        **kwargs,
    ):
        super().__init__(
            fill_hole_area=fill_hole_area,
            non_overlap_masks=non_overlap_masks,
            clear_non_cond_mem_around_input=clear_non_cond_mem_around_input,
            add_all_frames_to_correct_as_cond=add_all_frames_to_correct_as_cond,
            **kwargs,
        )
        self.state = {}
        self.frame_idx = -1

    @torch.inference_mode()
    def initialize(self, img):
        self.state = {}
        self.state["images"] = []
        self.state["offload_video_to_cpu"] = False
        self.state["offload_state_to_cpu"] = False
        self.state["device"] = self.device
        self.state["storage_device"] = self.device
        self.state["point_inputs_per_obj"] = {}
        self.state["mask_inputs_per_obj"] = {}
        self.state["cached_features"] = {}
        self.state["constants"] = {}
        self.state["obj_id_to_idx"] = OrderedDict()
        self.state["obj_idx_to_id"] = OrderedDict()
        self.state["obj_ids"] = []
        self.state["output_dict"] = {
            "cond_frame_outputs": {},
            "non_cond_frame_outputs": {},
        }
        self.state["output_dict_per_obj"] = {}
        self.state["temp_output_dict_per_obj"] = {}
        self.state["consolidated_frame_inds"] = {
            "cond_frame_outputs": set(),
            "non_cond_frame_outputs": set(),
        }
        self.state["tracking_has_started"] = False
        self.state["frames_already_tracked"] = {}

        self.frame_idx = 0
        img, width, height = self._process_img(img)
        self.state["images"].append(img)
        self.state["num_frames"] = 1
        self.state["video_height"] = height
        self.state["video_width"] = width
        self.state["frames_tracked_per_obj"] = {}
        self._get_image_feature(self.state, frame_idx=0, batch_size=1)

    @classmethod
    def from_pretrained(cls, model_id: str, **kwargs) -> "SAM2StreamPredictor":
        return build_sam2_stream_predictor_hf(model_id, **kwargs)

    @torch.inference_mode()
    def track(self, img):
        self.frame_idx += 1
        img, _, _ = self._process_img(img)
        self.state["images"].append(img)
        self.state["num_frames"] = len(self.state["images"])
        if not self.state["tracking_has_started"]:
            self.propagate_in_video_preflight(self.state)
        output_dict = self.state["output_dict"]
        obj_ids = self.state["obj_ids"]
        batch_size = self._get_obj_num(self.state)
        compact_current_out, pred_masks_gpu = self._run_single_frame_inference(
            self.state,
            output_dict,
            self.frame_idx,
            batch_size,
            is_init_cond_frame=False,
            point_inputs=None,
            mask_inputs=None,
            reverse=False,
            run_mem_encoder=True,
            prev_sam_mask_logits=None,
        )
        output_dict["non_cond_frame_outputs"][self.frame_idx] = compact_current_out
        self._add_output_per_object(self.state, self.frame_idx, compact_current_out, "non_cond_frame_outputs")
        self.state["frames_already_tracked"][self.frame_idx] = {"reverse": False}
        _, video_res_masks = self._get_orig_video_res_output(self.state, pred_masks_gpu)
        return obj_ids, video_res_masks

    def _process_img(self, img, image_size=1024, img_mean=(0.485, 0.456, 0.406), img_std=(0.229, 0.224, 0.225)):
        if isinstance(img, np.ndarray):
            height, width = img.shape[:2]
            img_np = cv2.resize(img, (image_size, image_size)) / 255.0
        else:
            width, height = img.size
            img_np = np.array(img.convert("RGB").resize((image_size, image_size))) / 255.0

        img = torch.from_numpy(img_np).permute(2, 0, 1).float()
        img_mean = torch.tensor(img_mean, dtype=torch.float32)[:, None, None]
        img_std = torch.tensor(img_std, dtype=torch.float32)[:, None, None]
        img -= img_mean
        img /= img_std
        return img, width, height
