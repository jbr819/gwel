from abc import ABC, abstractmethod
import numpy as np
import os
import sys

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

    def download_model(self, url: str, loc : str = os.path.join('.gwel','models')):  

        if url.startswith('hf'):
            try:
                from huggingface_hub import snapshot_download
            except ImportError:
                print("\033[1;31m"
                    "Error: huggingface_hub is not installed. Please install it with:\n"
                    "pip install huggingface-hub"
                    "\033[0m")
                sys.exit(0)
            repo = url.removeprefix('hf:')
            repo_url = "https://huggingface.co/" + repo
            target_dir = os.path.join(loc, repo) 
            if not os.path.exists(target_dir):
                snapshot_download(
                    repo_id=repo,
                    local_dir=target_dir,
                    local_dir_use_symlinks=False
                )
            return target_dir
        else:
            return None

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
