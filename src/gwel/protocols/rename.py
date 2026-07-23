from gwel.dataset import ImageDataset
from gwel.protocol import Exporter
from tqdm import tqdm
from pycocotools import mask as mask_utils
import pandas as pd
import numpy as np
import json
import shutil
import re
import math
import os
         
class RENAME(Exporter):

    def __init__(self, dataset: ImageDataset):
        self.dataset = dataset

    def export(self,path: str):
        os.makedirs(path, exist_ok=True)
        json_file = input('Enter the path of to dictionary of new image names in .json format (leave blank for .gwel/captions.json):')
        if not json_file:
            json_file = '.gwel/captions.json'
        
        if os.path.exists(json_file):
            with open(json_file, "r") as f:
                data = json.load(f)
        else:
            raise ValueError(f'Could not find file at {json_file}')

        dataset = self.dataset
        images = dataset.images
        for image in images:
            if image in data:
                captions = data[image]
                if image in data:
                    captions = data[image]

                    # Determine new_name
                    if isinstance(captions, str):
                        new_name = captions

                    elif isinstance(captions, list):
                        if len(captions) == 0:
                            print(f"Empty caption list for {image}")
                            continue
                        
                        ### Easter Egg: for renaming scans of leaf images in the style of Ethan Stewart et al. (2016)
                        ### (https://doi.org/10.1094/PHYTO-01-16-0018-R)
                        numbers = []
                        bases = []

                        # Extract suffix pattern _N at the end
                        for item in captions:
                            match = re.search(r'_(?:L)?(\d+)$', item)
                            if match:
                                numbers.append(int(match.group(1)))
                                bases.append(item[:match.start()])
                            else:
                                bases.append(item)

                        # If all entries have the same base and only differ by _N
                        if len(numbers) == len(captions) and len(set(bases)) == 1:
                            # divide by 8 and round down
                            correct_number = math.floor(numbers[0] / 8)+1
                            new_name = f"{bases[0]}_{correct_number}"
                        else:
                            # fallback to first entry
                            new_name = captions[0]

                    else:
                        raise TypeError(f"Unexpected data type for {image}: {type(captions)}")

                
                # Add .png if no file extension is present
                if not os.path.splitext(new_name)[1]:
                    new_name += ".png"

                    shutil.copy(
                        os.path.join(dataset.directory, image),
                        os.path.join(path, new_name)
                    )
                else:
                    print(f"No rename entry for {image}")  
                






