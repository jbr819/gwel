from gwel.network import Segmenter
from gwel.network import Detector
import numpy as np
import cv2
from pycocotools import mask as mask_utils


class LociSegmenter(Segmenter):
    def __init__(self, detector: Detector,
                 bandwidth: int,
                 kernel_size: int):
        self.detector = detector
        self.bandwidth = bandwidth
        self.kernel_size = kernel_size
        self.channels = ['loci']
   
    def set_device(self, device : str):
        pass 

    def load_weights(self, weights : str = None):
        pass

    def inference(self, image : np.ndarray):
 
        #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        H=image.shape[0]
        W=image.shape[1]
        objects = self.detector.inference(image)

        points = np.zeros((H, W), dtype=np.float32)
        uniform_kernel_points = points
        coords = np.array(objects)
        if len(coords) > 0:
            x = ((coords[:,0,0, 0, 0] + coords[:,0,2, 0, 0]) / 2).astype(int)
            y = ((coords[:,0,0, 0, 1] + coords[:,0,2, 0, 1]) / 2).astype(int)
        
            x = np.clip(x, 0, W - 1)
            y = np.clip(y, 0, H - 1)

            for i, j in zip(y, x):
                cv2.circle(uniform_kernel_points, (j, i), radius=self.bandwidth, color=1, thickness=-1)  
        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.kernel_size*2+1, self.kernel_size*2+1))
        mask = cv2.morphologyEx(uniform_kernel_points, cv2.MORPH_CLOSE, kernel)
        rle = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
        output = {'loci': rle}
        return output
    
 
 
