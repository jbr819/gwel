import os
import shutil
import cv2
import glob
from PIL import Image
from tqdm import tqdm
import random
import json
import numpy as np
import pandas as pd
from gwel.network import Detector, Segmenter 
import sys
import os
from colorama import Fore, Style, init
import datetime
from pycocotools import mask as maskUtils

# Initialize colorama
init(autoreset=True)

hidden_file_name = ".gwel"
Image.MAX_IMAGE_PIXELS = None

def scale_contours(contours, scale_x, scale_y):
    scaled_contours = []
    for contour in contours:
        if np.issubdtype(contour.dtype, np.integer):
            contour = contour.astype(np.float32)
            contour = contour + 0.5
        if len(contour.shape) > 2:
            contour = contour[0, :] if contour.shape[0] == 1 else np.squeeze(contour)      
        contour[:, 0] *= scale_x 
        contour[:, 1] *= scale_y
        if scale_x * scale_y > 1:
            contour = np.round(contour).astype(np.int32)
        scaled_contours.append(contour)
    return scaled_contours

class ImageDataset:
    def __init__(self, directory: str, check : bool = False):
       

        self.directory = directory
        self.images = self._get_image_list(self.directory)
        if len(self.images) == 0:
            raise ValueError(f"No images found in directory: {directory}")
        self.image_sizes = self._get_image_sizes(self.directory)
        self.resized_image_sizes = {} 
        self.resized_directory = ''
        self.patches = []
        self.patch_size = None
        self.object_detections = {}
        self.masks = {}
        self.flagged = []
        self.annotations = False
        self.factors = {}
        self.warp = {}
        
        os.makedirs(os.path.join(self.directory, hidden_file_name), exist_ok=True)

        # Check for problematic images and filter them out
        if check:
            self.check_images()

    def _get_image_list(self, directory : str):
        return [f for f in os.listdir(directory) if f.lower().endswith(('.jpg', '.png', '.jpeg', '.bmp', '.tiff','.tif', '.JPG'))]
    
    def _get_image_paths(self,directory : str):
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.gif', '*.bmp', '*.tiff','*.JPG','*.tif']
        image_paths = [] 
        for ext in image_extensions:
            image_paths.extend(glob.glob(os.path.join(directory, ext)))
        return image_paths
    

    def _get_image_sizes(self, directory : str):  
        image_sizes = {}
        for image_name in tqdm(self.images, desc="Retriving image sizes", unit="image"):
            img = Image.open(os.path.join(directory ,image_name))
            if hasattr(img, '_getexif'):
                exif = img._getexif()
                width, height = (img.size[1], img.size[0]) if exif and exif.get(274) in [6, 8] else img.size
            else:
                width, height = (img.size[0], img.size[1])    
            image_sizes[image_name] = [height, width]
        return image_sizes

    def _check_image(self, image_name):
        """Check if a single image can be read."""
        img_path = os.path.join(self.directory, image_name)
       
        return True if cv2.imread(img_path) is not None else False

    def check_images(self):
        """Check if images can be read and report any problematic files."""
        problematic_images = []

        for image in tqdm(self.images, desc = "Checking images", unit = "image"):
            check = self._check_image(image)
            if not check: 
                problematic_images.append(image)

        self.images = [img for img in self.images if img not in problematic_images]

        if problematic_images:
            print("Warning: Unable to read the following images:")
            print("\n".join(f"- {img}" for img in problematic_images))

    def _print_status(self, condition, message):
        """Helper function to print the status with appropriate colors."""
        if condition:
            print(f"{Fore.GREEN}{Style.BRIGHT}{message} ✓{Style.RESET_ALL}")
        else:
            print(f"{Fore.RED}{Style.BRIGHT}{message} X{Style.RESET_ALL}")

    def status(self):
        print("=" * 25)
        print(f"{Style.BRIGHT}{Fore.CYAN}--- STATUS ---{Style.RESET_ALL}")
        self._print_status(self.directory, "Directory")
        self._print_status(self.images, "Images list")
        self._print_status(self.image_sizes, "Image Sizes")
        self._print_status(self.factors, f"Factors ({len(self.factors)})")
        self._print_status(self.resized_image_sizes, "Resized Images")
        self._print_status(self.object_detections, "Object Detections")
        self._print_status(self.masks, "Segmentation")
        print("=" * 25)  

    def factor(self, name: str, values : list):
        self.factors[name]=values

    def sort(self, factors = []):
        self.images = self._get_image_list(self.directory)
        N = len(self.images)
        order = np.arange(N)
        if factors:
            multiplier = 1
            for factor in factors:
                values = self.factors[factor]
                value_set = sorted(list(set(values)), reverse = True) # sort so alpha comes before numeric in string values
                multiplier = N * multiplier
                if isinstance(list(value_set)[0], str):
                    values = [value_set.index(value) for value in values]
                values = np.array(values)
                order = order + multiplier * values
        
        order = list(order)
        self.images = [ x for _,x in sorted(zip(order,self.images))]

    def tree(self, output_dir, factors):
        df = pd.DataFrame(self.factors)
        df["image"] =  self._get_image_list(self.directory)
        for factor_values, df_group in tqdm(df.groupby(factors)):
            path_parts = map(str, factor_values) 
            dir_path = os.path.join(output_dir, *path_parts)
            os.makedirs(dir_path, exist_ok=True)
            images = df_group['image']
            for image in images:
                image_path=os.path.join(self.directory,image)
                new_image_path= os.path.join(dir_path,image)
                shutil.copy(image_path, new_image_path)


    def unflag(self):
        self.flagged = []


    def flip(self, flipped_directory=''):
        response = input("Are you sure you want to flip the currently flagged images? "
                     "Saved annotations may not be flipped if not loaded and saved after flip (y/n): ").strip().lower()

        if response != 'y':
            print("Aborted flipping.")
            return 

        if flipped_directory:
            os.makedirs(os.path.join(self.directory,flipped_directory), exist_ok=True)
        for image in self.flagged:
            image_path = os.path.join(self.directory, image)
            img = cv2.imread(image_path)
            img_flipped =cv2.rotate(img,cv2.ROTATE_180)
            image_flipped_path = os.path.join(self.directory,flipped_directory,image)
            cv2.imwrite(image_flipped_path,img_flipped)
            if self.resized_directory:
                if os.path.exists(os.path.join(self.resized_images_directory,image)): 
                    image_path = os.path.join(self.resized_images_directory,image)
                    img = cv2.imread(image_path)
                    img_flipped =cv2.rotate(img,cv2.ROTATE_180)
                    image_flipped_path = os.path.join(self.directory,self.resized_images_directory,image)
                    if not flipped_directory:
                        cv2.imwrite(image_flipped_path,img_flipped)
        self.unflag()

    
    def filter(self,factors = {}):
        result = []
        images_list = self._get_image_list(self.directory)
        for factor, values in factors.items():
            if factor in self.factors:  
                indices = [i for i, val in enumerate(self.factors[factor]) if val in values]
                result.extend(images_list[i] for i in indices)
            else:
                print(f'Unknown factor: {factor}.')
                return
        return result


    def sample(self, directory : str, N : int = 0, flagged : bool = True, factors : list = {}, resized = False):

        images_list = self._get_image_list(self.directory)
        
        if os.path.exists(directory):
            choice = input("This directory already exists. Do you want to continue and use it anyway? (y/n): ").strip().lower()
            if choice != 'y':
                print("Sampling cancelled.")
                return
        else:
            os.makedirs(directory)

        if factors:
            images_list = self.filter(factors)
            if N == 0:
                N = len(images_list)
        
      
        if flagged:
            N_flagged = len(self.flagged)
            if N < N_flagged:
                print(f'Sample size is less then the number of flagged images. Do you need flagged = False, or set N = {len(self.flagged)}?')
                return
            images = random.sample(images_list,N - N_flagged)
            images.extend(self.flagged)
        else:
            images = random.sample(images_list, N)
        
        
        if N ==0:
            print('N must be a postive integer')
            return
        if resized:
            directory_original = os.path.join(self.resized_images_directory) 
        else:
            directory_original = self.directory

        for image in images:
            shutil.copy(os.path.join(directory_original,image),os.path.join(directory,image))

    def resize(self, resized_directory: str = os.path.join(hidden_file_name,"resized"), max_size = 800):
        os.makedirs(os.path.join(self.directory,resized_directory), exist_ok=True)
        os.makedirs(os.path.join(self.directory,resized_directory,"images"), exist_ok=True)
        self.resized_directory = os.path.join(self.directory,resized_directory)
        self.resized_images_directory = os.path.join(self.directory,resized_directory,"images")

        for image_name in tqdm(self.images, desc="Resizing images", unit="image"):
            resized_img_path = os.path.join(self.directory, resized_directory,"images", image_name)
            if os.path.exists(resized_img_path):
                continue
            img = cv2.imread(os.path.join(self.directory, image_name))
            ratio = min(max_size / img.shape[1], max_size / img.shape[0], 1)
            resized_img = cv2.resize(img, (int(img.shape[1] * ratio), int(img.shape[0] * ratio)), interpolation=cv2.INTER_AREA)
            cv2.imwrite(resized_img_path, resized_img)

        self.resized_image_sizes = self._get_image_sizes(os.path.join(self.directory, resized_directory,"images"))


    def patch_images(self,patch_size, patch_directory: str = os.path.join(hidden_file_name,"patches")):
        os.makedirs(os.path.join(self.directory,patch_directory,"images"), exist_ok=True)
        self.patch_size = patch_size
        self.output_directories["patches"] = patch_directory
        self.output_directories["patch_images"] = os.path.join(patch_directory,"images")
        self.patch_size = patch_size
        for image_name in tqdm(self.images, desc="Patching images", unit="image"):
            img = cv2.imread(os.path.join(self.directory, image_name))

            img_size = self.image_sizes[image_name]
            pad_H = (patch_size - img_size[0] % patch_size) % patch_size
            pad_W = (patch_size - img_size[1] % patch_size) % patch_size
            padded_img = np.pad(img, ((0, pad_H), (0, pad_W), (0, 0)), mode='constant', constant_values=0)
            patches = patchify(padded_img, [patch_size, patch_size, img.shape[2]], step=patch_size)
            patch_count = 0
            for i, patch_row in enumerate(patches):
               for j, patch in enumerate(patch_row):
                    patch_filename = os.path.join(self.directory,patch_directory,"images", f"{image_name}_patch_{i}_{j}.png")
                    print(patch_filename)
                    cv2.imwrite(patch_filename, patch[0])
    
    def sample_patches(self, N : int , dir : str, flagged : bool = True):
        if os.path.exists(dir):
            print('This directory already exists, perhaps try a different one?')
            return
        os.makedirs(dir)
        patch_dir = self.output_directories["patch_images"] 
        patches_list = os.listdir(patch_dir)
        images = random.sample(patches_list, N)
        for image in images:
            shutil.copy(os.path.join(self.directory,patch_dir,image),os.path.join(dir,image))


    def detect(self, detector : Detector = None, 
               use_saved : bool = True, 
               annotations_path : str = None,
               write : bool = True,
               threshold: float = 0.3,
               add = False):
        resized_dir = self.resized_directory
        if use_saved == True:
            pre_saved = self.load_object_detections(annotations_path)
            if pre_saved:
                return
            else:
                print("Pre-saved annotations not found.")
            
            
        if not resized_dir:
            print('Try resizing the images first')
            return
        else:
            resized_dir = self.resized_images_directory
 
       
        if not detector:
            raise ValueError("No Detector object provided.")

        print("Detecting Objects...")
        
        
        detector.threshold = threshold 
        
        if add:
            class_names = self.object_detections.get("class_names", {})
            offset = max(class_names.keys()) + 1 if class_names else 0
            for cid, name in detector.model.names.items():
                class_names[cid + offset] = name
            self.object_detections["class_names"] = class_names
        else:
            self.object_detections["class_names"] = detector.model.names

        for image_name in tqdm(self.images, desc="Processing images", unit="image"):
            if add: 
                self.object_detections[image_name]=self.object_detections.get(image_name,{"image_size": None, "polygons": [],"class_id":[], "conf":[]})
            else:
                self.object_detections[image_name]= {"image_size": None, "polygons": [],"class_id":[],"conf":[]}
            img = cv2.imread(os.path.join(resized_dir, image_name))
            detected_instances = detector.inference(img) 
            height, width, _ = img.shape

            self.object_detections[image_name]["image_size"] = (height, width)
            for class_id, polygon, score in detected_instances:
                self.object_detections[image_name]["polygons"].append(polygon)                
                self.object_detections[image_name]["conf"].append(score)                
                if add:
                    self.object_detections[image_name]["class_id"].append(offset+class_id)               
                else:
                    self.object_detections[image_name]["class_id"].append(class_id)               


        if write: 
            #self.write_object_detections()
            self.write_object_detections(resized=True)




    def load_object_detections(self, annotations_file :str , write : bool = False, add = False):
        
        if not annotations_file:
            subdir = hidden_file_name #self.resized_directory if getattr(self, "resized_directory", "") else hidden_file_name
            annotations_file = os.path.join(self.directory, subdir, "detections.json")
        if not os.path.exists(annotations_file):
            return False

        with open(annotations_file, 'r') as f:
            coco_data = json.load(f)
        
        image_info = {image["id"]: (image["file_name"], image["width"], image["height"]) for image in coco_data["images"]}
        category_info = {cat["id"]:cat["name"] for cat in coco_data["categories"]}
        
        if add:
            class_names = self.object_detections.get("class_names", {})
            offset = max(class_names.keys()) + 1 if class_names else 0
            for cid, name in category_info.items():
                class_names[cid + offset] = name
            self.object_detections["class_names"] = class_names
            for image_name in self.images:
                self.object_detections[image_name]=self.object_detections.get(image_name,{"image_size": None, "polygons": [],"class_id":[], "conf":[]})
        else:
            self.object_detections["class_names"] = category_info
            for image_name in self.images:
                self.object_detections[image_name]= {"image_size": None, "polygons": [],"class_id":[],"conf":[]}

         
        for annotation in tqdm(coco_data["annotations"], desc="Loading annotations", unit="annotation"):
            image_id = annotation["image_id"]
            image_name, width, height = image_info[image_id]
            segmentation = annotation["segmentation"]
            if add:
                cls_id = annotation["category_id"] + offset
            else:    
                cls_id = annotation["category_id"]
            conf = annotation.get("confidence",None)
            contours = []
            if isinstance(segmentation,list): #polygons
                for segment in segmentation:
                    segment = np.array(segment,dtype=np.int32).reshape(-1, 2)
                    contours.append(segment)
            elif isinstance(segmentation,dict) and "counts" in segmentation: #rle (run lenth encoded)
                mask = maskUtils.decode(segmentation)
                contours_cv, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for c in contours_cv:
                    c = c.reshape(-1, 2)
                    contours.append(c)
            else:
                raise ValueError("Unknown segmentation format")


            if image_name in self.images:
                self.object_detections[image_name]["polygons"].append(contours) 
                self.object_detections[image_name]["class_id"].append(cls_id)
                self.object_detections[image_name]["conf"].append(conf)
                self.object_detections[image_name]["image_size"] = (height, width)

        if write:
            #self.write_object_detections()
            self.write_object_detections(resized = True)
        
        return True 
  
    def write_object_detections(self, output_file : str = None, resized : bool = False, overwrite : bool = True, supercategory : str = 'object'):
        if not self.object_detections.get("class_names"):
            print("No objects detections found.")
            return

        coco_data = {
                "licenses": [{"name":"","id":0,"url":""}],
                "info": {"contributor":"","date_created":"","description":"","url":"","version":"","year":""},
                "images": [],
                "annotations": [],
                "categories": [{"id": i, "name": n, "supercategory": supercategory} 
                               for i, n in self.object_detections["class_names"].items() ]
            }
        
        annotation_id = 1
        image_id = 1

        for image_name in tqdm(self.images, desc="Writing COCO annotations", unit="image"):
            
            width, height = self.resized_image_sizes[image_name] if resized else self.image_sizes[image_name]

            coco_data["images"].append({
                "id": image_id,
                "file_name": image_name,
                "height": width,
                "width": height
            })
            
            detections  = self.object_detections.get(image_name, []) 
            for instance, cls_id, conf in zip(detections["polygons"],detections["class_id"], detections["conf"]): 
                segmentation = []
                all_x, all_y = [], []

                for polygon in instance:
                    #polygon = np.squeeze(polygon)
                    #if isinstance(polygon[0], (list, tuple, np.ndarray)):
                    #segment = [int(c) for p in polygon for c in p]
                    polygon = polygon.reshape(-1, 2)  # shape becomes (4,2)
                    xs, ys = zip(*polygon)             # now works
                    segment = polygon.flatten().astype(int).tolist()
                    #else:
                     #   print(annotation_id)
                      #  print(len(polygon))
                       # segment = [int(v) for v in polygon]
                        #xs, ys = segment[::2], segment[1::2]
                    segmentation.extend(segment)
                    all_x.extend(xs)
                    all_y.extend(ys)

                if len(segmentation) < 6:
                    continue

                x_min, y_min = int(min(all_x)), int(min(all_y))
                width, height = int(max(all_x)-x_min), int(max(all_y)-y_min)

                coco_data["annotations"].append({
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": cls_id,
                    "segmentation": [segmentation],
                    "bbox": [x_min, y_min, width, height],
                    "area": width*height,
                    "iscrowd": 0,
                    "confidence": float(conf) if conf is not None else None
                })
                annotation_id += 1
  
            image_id += 1

        
        
        if not output_file: 
            output_file = os.path.join(self.directory,hidden_file_name,"detections.json")
            #resized_dir =  self.resized_directory
            #if resized and resized_dir:
                #output_file = os.path.join(resized_dir ,"detections.json")
        
        if not output_file:
            print('Could not write with no output file defined. For default write locations ensure overwrite = True')
            return

        with open(os.path.join(output_file), 'w') as f:
            json.dump(coco_data, f, indent=4)
	
        
    def crop(self, output_directory: str, object_name : str):
        if os.path.exists(output_directory):
            print(f'The directory {output_directory} already exists.')
        os.makedirs(output_directory, exist_ok=True)
        self.object_images_directory = output_directory  # Keep a record of the output directory
        print("Cropping...")
        
        for image_name in tqdm(self.object_detections.keys(), desc="Cropping Objects", unit="Image"):
            img_path = os.path.join(self.directory, image_name)
            if not os.path.exists(img_path):
                continue

            img = cv2.imread(img_path)
            detections = self.object_detections[image_name]
            W, H = detections['image_size']

            for n, contours in enumerate(detections['polygons']):
                instance_mask = np.zeros((W,H), dtype = np.uint8)
                cv2.drawContours(instance_mask, contours, contourIdx=-1, color=255, thickness=cv2.FILLED)
                instance_mask = cv2.resize(instance_mask,(img.shape[1],img.shape[0]))
                rescaled_contours, _ = cv2.findContours(instance_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                x, y, w, h = cv2.boundingRect(rescaled_contours[0])
                x_min, x_max = x , x+w
                y_min, y_max = y , y+h
                cropped_img = img[int(y_min):int(y_max), int(x_min):int(x_max)]
           
                cv2.imwrite(os.path.join(output_directory, f"{os.path.splitext(image_name)[0]}_{object_name}_{n+1}.jpg"), cropped_img)
            
        return ImageDataset(output_directory)

    
        

    def export_train_data(self, output_dir: str, N: int, format: str = "COCO", include_flagged: bool = True, dataset_name: str = "exported_data"):
        if format not in ["COCO", "YOLO"]:
            raise ValueError("Only COCO and YOLO formats are supported.")

        # Prepare directories for output
        base_output_dir = os.path.join(output_dir, dataset_name)
        output_dir = base_output_dir
        n = 1
        while os.path.exists(output_dir):
            output_dir = f"{base_output_dir}_{n}"
            n += 1
        os.makedirs(output_dir, exist_ok=True)

        # Select images based on the include_flagged parameter
        flagged = self.flagged if include_flagged else []
        remaining = [img for img in self.images if img not in flagged]
        images_to_export = random.sample(flagged, min(N, len(flagged))) + random.sample(remaining, max(0, N - len(flagged)))

        if format == "COCO":
            coco_data = {
                "images": [],
                "annotations": [],
                "categories": [{"id": 1, "name": "object", "supercategory": "plant"}]
            }
            annotation_id = 1
            image_id = 1

            for image_name in tqdm(images_to_export, desc="Preparing COCO annotations", unit="image"):
                # Get image size
                img_path = os.path.join(self.directory, image_name)
                img = Image.open(img_path)
                width, height = (img.size[1], img.size[0]) if img._getexif().get(274) in [6, 8] else img.size

                # Add image information
                coco_data["images"].append({
                    "id": image_id,
                    "file_name": image_name,
                    "height": height,
                    "width": width
                })

                # Add annotations (bounding boxes)
                for box in self.object_detections.get(image_name, []):
                    x1, y1, x2, y2 = map(int, box)  # Get coordinates as integers
                    width_bbox = x2 - x1
                    height_bbox = y2 - y1

                    coco_data["annotations"].append({
                        "id": annotation_id,
                        "image_id": image_id,
                        "category_id": 1,  # 'leaf' category ID
                        "bbox": [x1, y1, width_bbox, height_bbox],  # COCO uses [x, y, width, height]
                        "area": width_bbox * height_bbox,  # Area of the bounding box
                        "iscrowd": 0  # 'iscrowd' is 0 for individual instances
                    })
                    annotation_id += 1

                image_id += 1

            # Save COCO JSON file
            with open(os.path.join(output_dir, "detections.json"), 'w') as f:
                json.dump(coco_data, f, indent=4)

        elif format == "YOLO":
            for image_name in tqdm(images_to_export, desc="Preparing YOLO annotations", unit="image"):
                # Get image size
                img_path = os.path.join(self.directory, image_name)
                img = Image.open(img_path)
                width, height = (img.siz1e[1], img.size[0]) if img._getexif().get(274) in [6, 8] else img.size

                # YOLO annotation file
                yolo_file_path = os.path.join(output_dir, f"{os.path.splitext(image_name)[0]}.txt")
                with open(yolo_file_path, 'w') as yolo_file:
                    for box in self.detections.get(image_name, []):
                        x1, y1, x2, y2 = map(int, box)

                        # Convert bbox to YOLO format (class_id, x_center, y_center, bbox_width, bbox_height), normalized
                        x_center = (x1 + x2) / 2 / width
                        y_center = (y1 + y2) / 2 / height
                        bbox_width = (x2 - x1) / width
                        bbox_height = (y2 - y1) / height

                        # YOLO format: "class_id x_center y_center width height"
                        yolo_file.write(f"0 {x_center} {y_center} {bbox_width} {bbox_height}\n")  # Assuming 'leaf' has class ID 0

        # Copy referenced images with a progress bar
        for image_name in tqdm(images_to_export, desc="Copying images", unit="image"):
            shutil.copy(os.path.join(self.directory, image_name), output_dir)

    def object_dataset(self):
        objects_dir = self.object_images_directory
        object_dataset = ImageDataset(objects_dir)
        return object_dataset

    def segment(self, segmenter: Segmenter = None, use_saved = True, masks_file : str = None):
       

        if use_saved:
            if not masks_file:
                masks_file =  os.path.join(self.directory,hidden_file_name, "masks.json")
            if os.path.exists(masks_file):
                self.read_segmentation(masks_file)
                return

        self.masks["channels"] = segmenter.channels.keys()
        
        
        for image_name in tqdm(self.images, desc= 'Segmenting Images...',unit='image'):
           
            
            if not segmenter:
                print("No segmenter.")
                return
            
            image = cv2.imread(os.path.join(self.directory,image_name)) 
            output = segmenter.inference(image) 
            self.masks[image_name] = output
        
        self.write_segmentation()

    def write_segmentation(self, output_file : str = None, polys : bool = False):
        if not output_file:
            output_file = os.path.join(self.directory,hidden_file_name, 'masks.json')
        
        categories = []
        for n, category in enumerate(self.masks['channels']):
            cat = {"id" : n+1, "name":category, "supercategory": "mask"}
            categories.append(cat)

        today = datetime.date.today().isoformat()

        coco_data = {
            "info": {
                "description": "gwel-dataset",
                "version": "1.0",
                "year": datetime.date.today().year,
                "contributor": "",
                "date_created": today
            },
            "licenses": [],
            "images": [],
            "annotations": [],
            "categories": categories
        }

             
        annotation_id = 1
        image_id = 1

        for image_name in tqdm(self.images, desc="Writing COCO annotations", unit="image"):
            
            height, width = self.image_sizes[image_name]

            coco_data["images"].append({
                "id": image_id,
                "file_name": image_name,
                "height": height,
                "width": width
            })
            

            masks = self.masks.get(image_name, [])
            for n, channel in enumerate(self.masks['channels']):
                rle = masks[channel]
                rle['counts'] = rle['counts'].decode('utf-8')
                if rle['counts'] == 'Pfb\\1':
                    continue
                if polys:   
                     from shapely.geometry import Polygon
                     binary_mask = maskUtils.decode(rle).astype(np.uint8)
                    # kernel = np.ones((3,3), np.uint8)
                    # smoothed_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel, iterations=4)
                     contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                     simplified_polygons = []
                     for contour in contours:
                         points = contour.squeeze()
                         if points.shape[0] < 3:
                             continue
                         polygon = Polygon(points)
                         simplified = polygon.simplify(tolerance=1.5, preserve_topology=True)
                         simplified_coords = np.array(simplified.exterior.coords)
                         simplified_polygons.append(simplified_coords) 
                     polygons = []
                     for contour in simplified_polygons:
                        contour = contour.reshape(-1, 2)
                        if len(contour) >= 3:  # Needs at least 3 points to form a polygon
                             polygon = contour.flatten().tolist()
                             polygons.append(polygon)
                     segmentation = polygons
                else:
                    segmentation = rle
                
                coco_data["annotations"].append({
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": n+1,  
                    "segmentation": segmentation, 
                    "bbox": maskUtils.toBbox(rle).tolist(),
                    "iscrowd": 1 
                })

                annotation_id += 1
        
            image_id += 1

        
        if not output_file:
            print('Could not write with no output file defined. For default write locations ensure overwrite = True')
            return

        with open(os.path.join(output_file), 'w') as f:
            json.dump(coco_data, f, indent=4)
	
    def read_segmentation(self, annotations_file: str = None):
        if annotations_file is None:
            annotations_file = os.path.join(self.directory, '.hidden', 'masks.json')

        with open(annotations_file, 'r') as f:
            coco_data = json.load(f)

        id_to_channel = {cat["id"]: cat["name"] for cat in coco_data["categories"]}

        image_info = {image["id"]: image["file_name"] for image in coco_data["images"]}
        self.masks = {'channels': list(id_to_channel.values())}

        for image_name in self.images:
            self.masks[image_name] = {}

        for ann in tqdm(coco_data["annotations"], desc="Loading annotations", unit="annotation"):
            image_id = ann["image_id"]
            channel = id_to_channel[ann["category_id"]]
            image_name = image_info[image_id]

            rle = ann["segmentation"]
            rle["counts"] = rle["counts"].encode("utf-8")  # required for pycocotools
            if image_name in self.images:
                self.masks[image_name][channel] = rle


