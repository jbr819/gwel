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

class YOLOv8(Detector):

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
        self.model = YOLO(weights, task = 'detect')
        if torch.cuda.is_available():
            device = torch.device('cuda')
            self.model.to(device)
        #self.model.to(self.device)
     
    def inference(self, image: np.ndarray):
        if not self.patch_size:
            results_list = []
            results = self.model.predict(image,verbose=False, device = self.device, save=False)
            boxes = results[0].boxes  # ultralytics Box object
            for xyxy, cls_id, score in zip(boxes.xyxy.cpu().numpy(), boxes.cls.cpu().numpy(), boxes.conf.cpu().numpy()):
                polygon = bbox_to_polygon([xyxy])[0]
                results_list.append((int(cls_id), polygon, score))

            #boxes = results[0].boxes.xyxy.cpu().numpy() 
            #detections = boxes.tolist()
            #detections = bbox_to_polygon(detections)
        else:
            results_list = self.inference_with_patches(patch_size = self.patch_size, image = image)

        shutil.rmtree(yolo_tmp, ignore_errors=True)
        return results_list

    def inference_with_patches(
        self,
        patch_size: tuple,
        image: np.ndarray,
        overlap_ratio: float = 0.2,
        iou_thresh: float = 0.2,
        border_margin: float = 0.05,
    ):
        h, w = image.shape[:2]
        patch_h, patch_w = patch_size

        stride_h = int(patch_h * (1 - overlap_ratio))
        stride_w = int(patch_w * (1 - overlap_ratio))

        pad_h = (patch_h - h % stride_h) % stride_h
        pad_w = (patch_w - w % stride_w) % stride_w

        padded_image = cv2.copyMakeBorder(
            image, 0, pad_h, 0, pad_w,
            cv2.BORDER_CONSTANT, value=(0, 0, 0)
        )

        detections = []
        class_ids = []

        # overlap-aware border margin
        margin_x = int(patch_w * overlap_ratio * border_margin)
        margin_y = int(patch_h * overlap_ratio * border_margin)

        for i in range(0, padded_image.shape[0] - patch_h + 1, stride_h):
            for j in range(0, padded_image.shape[1] - patch_w + 1, stride_w):
                patch = padded_image[i:i + patch_h, j:j + patch_w]
                results = self.model.predict(patch, verbose=False)

                boxes = results[0].boxes.xyxy.cpu().numpy()
                confs = results[0].boxes.conf.cpu().numpy()
                cls_ids = results[0].boxes.cls.cpu().numpy()  

                
                for bbox, conf, cls_id in zip(boxes, confs, cls_ids):
                    x1, y1, x2, y2 = bbox

                    x1 += j
                    y1 += i
                    x2 += j
                    y2 += i

                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(w, x2), min(h, y2)

                    # discard boxes touching slice border
                    if (
                        x1 <= j + margin_x or
                        y1 <= i + margin_y or
                        x2 >= j + patch_w - margin_x or
                        y2 >= i + patch_h - margin_y
                    ):
                        continue

                    detections.append([x1, y1, x2, y2, conf])
                    class_ids.append(int(cls_id))


        if len(detections) == 0:
            return []

        detections = torch.tensor(detections, dtype=torch.float32)
        boxes = detections[:, :4]
        scores = detections[:, 4]

        keep_idx = nms(boxes, scores, iou_thresh)

        class_ids = np.array(class_ids)
        boxes = boxes[keep_idx].cpu().numpy()
        class_ids = class_ids[keep_idx.cpu().numpy()].astype(int).tolist()
        scores = scores[keep_idx.cpu().numpy()].tolist()


        polygons = bbox_to_polygon(boxes)

        results_list = list(zip(class_ids,polygons,scores))

        shutil.rmtree(yolo_tmp, ignore_errors=True)

        return results_list
