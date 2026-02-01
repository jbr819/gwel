import numpy as np
import cv2
import warnings
import os
from gwel.network import Detector
import subprocess
import sys
import torch
from torchvision.ops import nms
import sys
import cv2
import math

try:
    from ultralytics import YOLO
    import tempfile, shutil
    from ultralytics.utils import SETTINGS
    yolo_tmp = tempfile.mkdtemp(prefix="ultralytics_")
    SETTINGS["runs_dir"] = yolo_tmp

except ImportError:
    sys.exit("Ultralytics not found. Install with 'pip install ultralytics'.")



warnings.filterwarnings("ignore")

def bbox_to_polygon(bboxes):
    polygons = []
    for bbox in bboxes:
        x1, y1, x2, y2 = bbox
        polygon = [[x1, y1], [x1, y2], [x2, y2], [x2, y1]]
        polygons.append([np.array(polygon,dtype=np.int32).reshape(-1, 1, 2)])
    return polygons

class YOLOv8seg(Detector):

    def __init__(self, weights: str , device: str = "cpu", patch_size: tuple = None):
        self.threshold = 0.1
        self.patch_size = patch_size
        self.device = device
        if weights:
            self.load_weights(weights)
        #self.set_device(device)

    def set_device(self, device: str):
        self.device = device
        if hasattr(self, 'model'):
            self.model.to(self.device)

    def load_weights(self, weights: str):
        self.weights = weights
        self.model = YOLO(weights, task = 'segment')
        if torch.cuda.is_available():
            device = torch.device('cuda')
            self.model.to(device)
        #self.model.to(self.device)
    
    
    def inference(self, image: np.ndarray, smoother: int = 18, threshold :float = 0.75):
        results_list = []
        h, w = image.shape[:2]

        # Determine scale factor so longest side = 640
        if h > w:
            scale = 640 / h
            new_h = 640
            new_w = int(round(w * scale / 32) * 32)  # nearest multiple of 32
        else:
            scale = 640 / w
            new_w = 640
            new_h = int(round(h * scale / 32) * 32)  # nearest multiple of 32

        # Resize image
        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        results = self.model.predict(image, device=self.device, imgsz=640, verbose=False)

        for r in results:
            boxes = r.boxes
            masks = getattr(r, "masks", None)

            for i, cls_id in enumerate(boxes.cls.cpu().numpy()):
                score = float(boxes.conf[i])

                if masks is not None:
                    # probability mask (0–1)
                    mask = masks.data[i].cpu().numpy()
                    
                    # smooth resize
                    mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_LINEAR)

                    # light smoothing
                    mask = cv2.GaussianBlur(mask, (2*smoother+1, 2*smoother+1), 0)

                    # threshold last
                    mask_bin = (mask > threshold).astype(np.uint8)
                    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_bin, connectivity=8)

                    if num_labels > 1:
                        # skip the background label 0
                        largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
                        # create new mask with only the largest component
                        largest_mask = np.zeros_like(mask_bin)
                        largest_mask[labels == largest_label] = 1
                    else:
                        largest_mask = mask_bin

                    contours, _ = cv2.findContours(
                        largest_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                    )

                    polygons = [
                        cv2.approxPolyDP(c, 0.00005 * cv2.arcLength(c, True), True).reshape(-1, 1, 2)
                        for c in contours
                    ]
                else:
                    xyxy = boxes.xyxy[i].cpu().numpy()
                    polygons = bbox_to_polygon([xyxy])

                results_list.append((int(cls_id), polygons, score))

        return results_list




    """
    def inference(self, image: np.ndarray):
        results_list = []
        h_orig, w_orig = image.shape[:2]  # original image size

        # Run YOLOv8 prediction
        results = self.model.predict(image, verbose=False, device=self.device, save=False,imgsz=800)

        for result in results:
            boxes = result.boxes
            masks = getattr(result, "masks", None)  # safer: masks may be None

            for idx, cls_id in enumerate(boxes.cls.cpu().numpy()):
                score = boxes.conf.cpu().numpy()[idx]

                if masks is not None:
                    # get mask for this detection
                    mask = masks.data.cpu().numpy()[idx]  # mask in model input size

                    # resize mask to original image size
                    mask_resized = cv2.resize(
                        (mask * 255).astype(np.uint8), 
                        (w_orig, h_orig), 
                        interpolation=cv2.INTER_NEAREST
                    )

                    # extract polygon contours
                    contours, _ = cv2.findContours(mask_resized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    epsilon = 1 # smaller = closer to original mask, larger = smoother
                    polygons = [cv2.approxPolyDP(cnt, epsilon, True).reshape(-1, 1, 2) for cnt in contours]

                else:
                    # fallback: use bounding box if mask is missing
                    xyxy = boxes.xyxy.cpu().numpy()[idx]
                    polygons = bbox_to_polygon([xyxy])

                results_list.append((int(cls_id), polygons, score))


        # cleanup temporary folder
        shutil.rmtree(yolo_tmp, ignore_errors=True)
        return results_list
        """

