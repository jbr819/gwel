from abc import ABC, abstractmethod
import numpy as np


class Network(ABC):
    @abstractmethod
    def load_weights(self):
        pass

    @abstractmethod
    def inference(self, image : np.ndarray):
        pass

    @abstractmethod
    def set_device(self, device : str):
        pass


class Detector(Network, ABC):
    pass 
    
    #def inference_with_patches(self, image : np.ndarray , patch_size : tuple[int,int]):
        #pass
    

    #def validate_inference():
        #pass

class Segmenter(Network,ABC):
    pass
   

    #def validate_inference():

class Classifier(Network, ABC):
    pass

    # def validate_inference():
