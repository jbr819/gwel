from gwel.dataset import ImageDataset
from gwel.protocol import Exporter
from tqdm import tqdm
from shapely.geometry import Polygon, box
import numpy as np
import os
import cv2
from itertools import product
from tqdm import tqdm

class Slicer(Exporter):

    def __init__(self, dataset: ImageDataset):
        self.dataset = dataset
        self.object_detections = dataset.object_detections
        self.new_detections = {"class_names": self.object_detections.get("class_names", {})}

    def _clip_bbox(self, bbox, x0, y0, x1, y1):
        bx1, by1, w, h = bbox
        bx1 = max(bx1, x0)
        by1 = max(by1, y0)
        bx2 = min(bx1+w, x1)
        by2 = min(by1+h, y1)

        if bx2 <= bx1 or by2 <= by1:
            return None
        return [bx1, by1, bx2-bx1, by2-by1]

    def _shift_bbox(self, bbox, x0, y0):
        # convert global -> slice-local coords
        x, y, w, h = bbox
        return [x - x0, y - y0, w, h]


    def _clip_poly(self, poly, x0, y0, x1, y1):
        poly_geom = Polygon(poly)
        tile = box(x0, y0, x1, y1)

        clipped = poly_geom.intersection(tile)

        if clipped.is_empty:
            return []

        if clipped.geom_type == "Polygon":
            return [np.asarray(clipped.exterior.coords[:-1], dtype=np.float32)]

        elif clipped.geom_type == "MultiPolygon":
            return [
                np.asarray(p.exterior.coords[:-1], dtype=np.float32)
                for p in clipped.geoms
            ]

        return []
   

    def _shift_poly(self, poly, x0, y0):
        return poly - np.array([x0, y0], dtype=poly.dtype)


    def export(self, slice_size: int, output_dir: str):

        os.makedirs(output_dir, exist_ok=True)
        print("Slicing images...")

        for image_name in tqdm(self.dataset.images):

            img_path = os.path.join(self.dataset.directory, image_name)
            img = cv2.imread(img_path)

            if img is None:
                print(f"Failed to load {img_path}")
                continue

            h, w = img.shape[:2]

            # pad to multiple of slice_size
            new_h = ((h + slice_size - 1) // slice_size) * slice_size
            new_w = ((w + slice_size - 1) // slice_size) * slice_size

            pad_top = (new_h - h) // 2
            pad_bottom = new_h - h - pad_top
            pad_left = (new_w - w) // 2
            pad_right = new_w - w - pad_left

            img = cv2.copyMakeBorder(
                img,
                pad_top, pad_bottom,
                pad_left, pad_right,
                cv2.BORDER_CONSTANT,
                value=[0, 0, 0]
            )

            detections = self.object_detections.get(image_name, {})
            bboxes = detections.get("bbox", [])
            class_ids = detections.get("class_id", [])
            confs = detections.get("conf", [])
            polygons = detections.get("polygons", [])
            basename = os.path.splitext(image_name)[0]

            for i, j in product(range(0, new_h, slice_size),
                                range(0, new_w, slice_size)):

                slice_img = img[i:i+slice_size, j:j+slice_size]

                slice_name = f"{basename}_{i}_{j}.png"

                slice_bboxes = []
                slice_classes = []
                slice_confs = []
                slice_polys = []

                x0, y0 = j-pad_right, i-pad_top
                x1, y1 = j-pad_right + slice_size, i-pad_top + slice_size

                # --- filter detections into slice ---
                for k, bbox in enumerate(bboxes):
                    clipped = self._clip_bbox(bbox, x0, y0, x1, y1)

                    if clipped is None:
                        continue
                    
                    slice_bboxes.append(self._shift_bbox(clipped, x0, y0))
                    slice_classes.append(class_ids[k])
                    slice_confs.append(confs[k])

                    # NOTE: polygons require same logic (not fully expanded here)
                    if polygons:
                        new_poly = []

                        for poly in polygons[k]:

                            clipped_parts = self._clip_poly(
                                poly,
                                x0, y0,
                                x1, y1
                            )

                            for part in clipped_parts:
                                new_poly.append(
                                    self._shift_poly(part, x0, y0)
                                ) 
                        slice_polys.append(new_poly)     

                # save image
                cv2.imwrite(os.path.join(output_dir, slice_name), slice_img)

                # store detections
                self.new_detections[slice_name] = {
                    "bbox": slice_bboxes,
                    "class_id": slice_classes,
                    "conf": slice_confs,
                    "polygons": slice_polys,
                    "image_size": (slice_size, slice_size)
                }
            
        sliced_dataset = ImageDataset(output_dir)
        sliced_dataset.object_detections = self.new_detections

        sliced_dataset.write_object_detections()

