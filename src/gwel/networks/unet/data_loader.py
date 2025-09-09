import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from pycocotools.coco import COCO
from torchvision import transforms

class CenterPad:
    def __init__(self, target_multiple=256):
       self.target_multiple = target_multiple
    
    def __call__(self, tensor):

        if len(tensor.shape) == 3:  
            _, height, width = tensor.shape
        elif len(tensor.shape) == 2: 
            height, width = tensor.shape
            
            tensor = tensor.unsqueeze(0)  
        else:
            raise ValueError("Input tensor must have 2 or 3 dimensions.")
        

        _, height, width = tensor.shape  # Assuming tensor has shape (C, H, W)
        
        pad_h = (self.target_multiple - height % self.target_multiple) % self.target_multiple
        pad_w = (self.target_multiple - width % self.target_multiple) % self.target_multiple
        
        pad_h_top = pad_h // 2
        pad_h_bottom = pad_h - pad_h_top
        pad_w_left = pad_w // 2
        pad_w_right = pad_w - pad_w_left
        
        padded_tensor = torch.nn.functional.pad(tensor, (pad_w_left, pad_w_right, pad_h_top, pad_h_bottom),
                                                value=0)
        return padded_tensor


class CocoSegmentationDataset(Dataset):
    def __init__(self, image_dir, coco_json, categories: list, transform = transforms.Compose([CenterPad(target_multiple=256)])):
        self.coco = COCO(coco_json)
        self.image_dir = image_dir
        self.image_ids = self.coco.getImgIds()
        self.categories = categories
        self.transform = transform

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        img_info = self.coco.loadImgs([img_id])[0]
        img_path = os.path.join(self.image_dir, img_info['file_name'])
        image = Image.open(img_path).convert("RGB")

        # Target mask initialized as background = 0
        target = np.zeros((image.height, image.width), dtype=np.uint8)

        # Loop over categories
        for catIdx, _ in enumerate(self.categories):  
            ann_ids = self.coco.getAnnIds(imgIds=[img_id], catIds=[catIdx+1])
            annotations = self.coco.loadAnns(ann_ids)

            mask = np.zeros((image.height, image.width), dtype=np.uint8)
            for ann in annotations:
                ann_mask = self.coco.annToMask(ann)
                mask = np.maximum(mask, ann_mask)

            target[mask == 1] = catIdx + 1  # assign class index

        # Convert to tensors
        image = torch.as_tensor(np.array(image), dtype=torch.float32).permute(2, 0, 1) / 255.0
        target = torch.as_tensor(target, dtype=torch.long)

        # Apply transforms 
        if self.transform is not None:
            image = self.transform(image)
            target = self.transform(target)
            target = target.squeeze(0).long()
        
        return image, target

