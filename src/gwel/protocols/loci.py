from gwel.dataset import ImageDataset
from gwel.protocol import Exporter
from tqdm import tqdm
from pycocotools import mask as mask_utils
from collections import defaultdict
import numpy as np
import cv2

class LOCI(Exporter):

    def __init__(self, dataset: ImageDataset):
            self.dataset = dataset

    def export(self,path: str = None):
                        
        if self.dataset.object_detections:
            object_class_dict = self.dataset.object_detections['class_names']
        if self.dataset.masks:
            mask_channels = self.dataset.masks['channels']

        for object_class in object_class_dict.values():
            mask_channels.append(object_class + '_loci')

        print('Calculating loci masks...')
        for image in tqdm(self.dataset.images):
            boxes = self.dataset.object_detections[image]['bbox']
            classes = self.dataset.object_detections[image]['class_id']
            H, W = self.dataset.image_sizes[image]

            class_centers = defaultdict(list)

            for box, cls_id in zip(boxes, classes):
                x, y, w, h = box
                cx = x + w / 2
                cy = y + h / 2
                class_centers[cls_id].append((cx, cy))

            for cls_id in class_centers:
                coords = class_centers[cls_id]
                loci_mask =  np.zeros((H, W), dtype=np.float32)

                bandwidth = 40
                kernel_size = 10

                for (i,j) in coords:

                    cv2.circle(loci_mask, (int(i), int(j)), radius=bandwidth, color=1, thickness=-1)  
               
                
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size*2+1, kernel_size*2+1))
                loci_mask = cv2.morphologyEx(loci_mask, cv2.MORPH_CLOSE, kernel)
                
                rles_dict = self.dataset.masks[image] 
                if rles_dict.get('unhealthy',None):
                    mask_yellow = mask_utils.decode(rles_dict['unhealthy'])
     
                    loci_mask = mask_yellow * loci_mask


                rle = mask_utils.encode(np.asfortranarray(loci_mask.astype(np.uint8)))
                self.dataset.masks[image][object_class_dict[cls_id]+'_loci'] = rle
                
        
        #if not path:
            
        output_file = '.gwel/masks-loci.json'
        
        self.dataset.write_segmentation(output_file)
        


