
from gwel.dataset import ImageDataset
from gwel.protocol import Exporter
from collections import Counter
from tqdm import tqdm
from pycocotools import mask as mask_utils
import pandas as pd
import numpy as np


class CSV(Exporter):

    def __init__(self, dataset: ImageDataset):
            self.dataset = dataset

    def export(self,path: str):
        columns = ["image_name"]
                        
        if self.dataset.object_detections:
            object_class_dict = self.dataset.object_detections['class_names']
            columns += [f"{cls_name}_count" for cls_name in object_class_dict.values()]
        if self.dataset.masks:
            mask_channels = self.dataset.masks['channels']
            columns += [f"{ch}_area" for ch in mask_channels]

        df = pd.DataFrame(columns=columns)

        image_names = self.dataset.images

        for image_name in tqdm(image_names):
            
            if self.dataset.object_detections:
                detections = self.dataset.object_detections[image_name]['class_id']
                class_counts = Counter(detections)
                row = {"image_name": image_name}
                for cls_id, cls_name in object_class_dict.items():
                    row[f"{cls_name}_count"] = class_counts.get(cls_id, 0)

            if self.dataset.masks:
                masks_dict = self.dataset.masks[image_name]
                for ch, rle in masks_dict.items():
                    decoded = mask_utils.decode(rle)
                    row[f"{ch}_area"] = int(np.sum(decoded))

            df.loc[len(df)] = row

        df.to_csv(path,index=False)


