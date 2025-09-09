import numpy as np
import cv2
import warnings
warnings.filterwarnings("ignore")

from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.structures import BitMasks

from mask2former import add_maskformer2_config

from gwel.network import Detector


class Mask2Former(Detector):

    def __init__(self, arch_yaml: str, weights: str = None, device : str = "cpu"):
        self.cfg = get_cfg()
        self.cfg.set_new_allowed(True)
        add_deeplab_config(self.cfg)
        add_maskformer2_config(self.cfg)
        self.cfg.merge_from_file(arch_yaml)
        self.threshold = 0.9

        if weights:
            self.load_weights(weights)
    
        self.set_device(device = device)

    def set_device(self, device : str):
        self.cfg.MODEL.DEVICE = device

    def load_weights(self, weights : str):
        self.cfg.MODEL.WEIGHTS = weights


    def inference(self, image : np.ndarray):
        cfg = self.cfg
        cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON = True
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
        predictor = DefaultPredictor(cfg)

        outputs = predictor(image)

        instances = outputs["instances"].to("cpu")
        high_conf_indices = np.where(instances.scores >= self.threshold)[0]  # Only keep high confidence
        filtered_instances = instances[high_conf_indices]

        pred_masks = filtered_instances.pred_masks
        image_height, image_width = filtered_instances.image_size

        bitmasks = BitMasks(pred_masks)
        contours_list = []
        for mask in bitmasks.tensor.cpu().numpy():  # Convert tensor to NumPy array
            contours, _ = cv2.findContours(mask.astype("uint8"), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours_list.append(contours)
        
        return contours_list




