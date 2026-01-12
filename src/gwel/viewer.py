from typing import Literal
import random
import copy
import cv2
import os
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm 
from gwel.dataset import ImageDataset 
import numpy as np
import pycocotools.mask as mask_utils
from matplotlib import colors as mcolors

class Viewer:
    def __init__(self, dataset: ImageDataset,
                 mode: Literal['', 'instance', 'segmentation','coordinates'] = ''  ,
                 max_pixels: int = 800,
                 contour_thickness: int = 1,
                 col_scheme = {}):
        
        self.max_pixels = max_pixels 
        self.scale_ratio = 1
        self.contour_thickness = contour_thickness
        self.mode = mode
        self.dataset = dataset
        self.directory = getattr(self.dataset, "resized_images_directory", self.dataset.directory)
        self.images = self.dataset.images
        self.total_images = len(self.images)
        self.index = 0 
        self.col = False
        self.col_scheme = {
            k: (
                None
                if v is None
                else tuple(int(c * 255) for c in reversed(mcolors.to_rgb(v)))
            )
            for k, v in col_scheme.items()
        }

        
    def load_image(self):
        self.image_name = self.images[self.index]
        self.image_path = os.path.join(self.directory,self.image_name)
        self.image = cv2.imread(self.image_path)
        self.image_size = self.dataset.image_sizes[self.image_name]
        self.flagged = self.image_name in self.dataset.flagged 
        if self.image is None:
            print(f"Warning: Unable to read image {self.image_name}.")
            return False
            
    def open(self):

        while True:
            self.load_image() 
            self.style() 
            self.display_image()
           
            key = cv2.waitKey(0)
        
            if key == ord('q'):  # Quit the viewer
                break
            elif key == ord('p'):  # previous image
                self.index = max(self.index - 1,0)
            elif key == ord('n'):  # next image
                self.index = min (self.index + 1 , self.total_images-1)  
            elif key == ord('f'):  # toggle flag/unflag the image
                if self.image_name in self.dataset.flagged:
                    self.dataset.flagged.remove(self.image_name)
                else:
                    self.dataset.flagged.append(self.image_name)
            elif key == ord('c'):
                self.col = not self.col    # toggle between colour schemes
            elif key == ord('1'):
                self.mode = "instance"
            elif key == ord('2'):
                self.mode = "segmentation"
            elif key == ord('3'):
                self.mode = "seginstance"
            elif key == ord("4"):
                self.mode = "instandseg"
            elif key == ord("5"):
                self.mode = "circandseg"
            elif key == ord("6"):
                self.mode = "block"
            elif key == ord("0"):
                self.mode = ""
            elif key == ord("w"):
                p = input("Enter output path (leave blank for default): ").strip()
                d = os.path.expanduser(self.dataset.directory + "/.gwel/saves") if not p else os.path.dirname(os.path.abspath(p))
                os.makedirs(d, exist_ok=True)
                if not p:
                    i = 1
                    while os.path.exists(os.path.join(d, f"output_{i}.png")): i += 1
                    p = os.path.join(d, f"output_{i}.png")
                else: p = os.path.abspath(p)
                try: self.save(p); print(f"Saved to {p}")
                except Exception as e: print(f"Error saving: {e}")

        cv2.destroyAllWindows()


    def display_image(self):
        
        height, width = self.image.shape[:2]
        scale_ratio = min(self.max_pixels / width, self.max_pixels / height, self.scale_ratio)
        scaled_image = cv2.resize(self.image, (int(width * scale_ratio), int(height * scale_ratio)), interpolation=cv2.INTER_LINEAR)

        bar_height = 40
        bar_color = (142, 76, 195) if self.flagged else (60, 179, 113)
        bar = np.full((bar_height, scaled_image.shape[1], 3), bar_color, dtype=np.uint8)
        overlay_text = f"{self.image_name} | {self.index + 1}/{self.total_images}"
        text_color = (230, 230, 230)
        cv2.putText(bar, overlay_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2,lineType=cv2.LINE_AA)
        combined_image = cv2.vconcat([bar, scaled_image])

        cv2.imshow("Viewer", combined_image)

        return True

    def style(self):

        if self.mode in ("instance","instandseg"): 

            if self.image_name in self.dataset.object_detections:

                self.detections = copy.deepcopy(self.dataset.object_detections[self.image_name])
                self.class_names = copy.deepcopy(self.dataset.object_detections["class_names"])
                
                if self.detections['image_size']:     
                   
                    H, W = self.detections['image_size']
                    h_img, w_img = self.image.shape[:2]

                    sx = w_img / W
                    sy = h_img / H

                    if not self.col:
                        base_colour = tuple(np.random.randint(0, 256, 3).tolist())

                    for contours, cls_id in zip(self.detections['polygons'], self.detections['class_id']):
                        label = self.class_names[cls_id] 
                        colour = tuple(np.random.randint(0, 256, 3).tolist()) if self.col else self.col_scheme.get(label, base_colour) 
                       
                        scaled_contours = []

                        for cnt in contours:
                            cnt = cnt.astype(np.float32)
                            cnt[:, 0] *= sx
                            cnt[:, 1] *= sy                       
                            scaled_contours.append(cnt.astype(np.int32).reshape(-1, 1, 2))

                        cv2.drawContours(
                            self.image,
                            scaled_contours,
                            contourIdx=-1,
                            color=colour,
                            thickness=self.contour_thickness
                        )

        if self.mode in ("segmentation","instandseg","circandseg"):

            if self.image_name  in self.dataset.masks:

                rles_dict = self.dataset.masks[self.image_name] 

                for label, rle in rles_dict.items():
                    mask = mask_utils.decode(rle) 
                    mask = cv2.resize(mask,(self.image.shape[1],self.image.shape[0]))
                    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                    random_colour = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
                    colour = self.col_scheme.get(label, random_colour)
                    if colour:
                        cv2.drawContours(self.image,contours,-1, colour , self.contour_thickness)
           
        

        if self.mode in ("seginstance",):
            
            if self.image_name in self.dataset.masks and self.image_name in self.dataset.object_detections:

                rles_dict = self.dataset.masks[self.image_name] 
                rles = list(rles_dict.values())
                 
                instance_mask = np.ones(self.image.shape[:2], dtype=np.uint8)
                 
                if self.image_name in self.dataset.object_detections:
                    self.detections = copy.deepcopy(self.dataset.object_detections[self.image_name])
                    W, H = self.detections['image_size']
                    instance_mask = np.zeros((W,H), dtype = np.uint8)
                    for contours in self.detections['polygons']:
                        cv2.drawContours(instance_mask, contours, contourIdx=-1, color=255, thickness=cv2.FILLED)
                    instance_mask = cv2.resize(instance_mask,(self.image.shape[1],self.image.shape[0]))

                rles_dict = self.dataset.masks[self.image_name] 

                for label, rle in rles_dict.items():
                    mask = mask_utils.decode(rle)
                    mask = cv2.resize(mask, (self.image.shape[1], self.image.shape[0]))
                    mask = mask * instance_mask
                    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                    random_colour = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
                    colour = self.col_scheme.get(label, random_colour)
                    if colour:
                        cv2.drawContours(self.image, contours, -1, colour, self.contour_thickness)



        if self.mode in ("circandseg",): 

            if self.image_name in self.dataset.object_detections:
                self.detections = copy.deepcopy(self.dataset.object_detections[self.image_name])
                self.class_names = copy.deepcopy(self.dataset.object_detections["class_names"])
     
                if self.detections['image_size']:     
                    H, W = self.detections['image_size']
                    h_img, w_img = self.image.shape[:2]

                    sx = w_img / W
                    sy = h_img / H

                    if not self.col:
                        base_colour = tuple(np.random.randint(0, 256, 3).tolist())

                    for contours, cls_id in zip(self.detections['polygons'],self.detections['class_id']):
                       
                        label = self.class_names[cls_id] 
                        colour = tuple(np.random.randint(0, 256, 3)) if self.col else self.col_scheme.get(label,base_colour)

                        scaled_contours = []

                        for cnt in contours:
                            cnt = cnt.astype(np.float32)
                            cnt[:, 0] *= sx
                            cnt[:, 1] *= sy                       
                            scaled_contours.append(cnt.astype(np.int32).reshape(-1, 1, 2))
                        x, y, w, h = cv2.boundingRect(scaled_contours[0])
                        center = (x + w // 2, y + h // 2)
                        axes = (w // 2, h // 2)
                        cv2.ellipse(self.image, center, axes, 0, 0, 360, colour, self.contour_thickness)

               
       
        if self.mode in ("block",):
            if self.image_name in self.dataset.masks and self.image_name in self.dataset.object_detections:
                self.detections = copy.deepcopy(self.dataset.object_detections[self.image_name])
                self.class_names = copy.deepcopy(self.dataset.object_detections["class_names"])
     
                rles_dict = self.dataset.masks[self.image_name] 
                for label, rle in rles_dict.items():
                    mask = mask_utils.decode(rle) 
                    mask = cv2.resize(mask,(self.image.shape[1],self.image.shape[0]))
                    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                    random_colour = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
                    colour = self.col_scheme.get(label, random_colour)
                    if colour:
                        cv2.drawContours(self.image,contours,-1, colour , cv2.FILLED)
                if self.detections['image_size']:     
                    H, W = self.detections['image_size']
                    h_img, w_img = self.image.shape[:2]

                    sx = w_img / W
                    sy = h_img / H

                    if not self.col:
                        base_colour = tuple(np.random.randint(0, 256, 3).tolist())

                    for contours, cls_id in zip(self.detections['polygons'],self.detections['class_id']):
                       
                        label = self.class_names[cls_id] 
                        colour = tuple(np.random.randint(0, 256, 3)) if self.col else self.col_scheme.get(label,base_colour)

                        scaled_contours = []

                        for cnt in contours:
                            cnt = cnt.astype(np.float32)
                            cnt[:, 0] *= sx
                            cnt[:, 1] *= sy                       
                            scaled_contours.append(cnt.astype(np.int32).reshape(-1, 1, 2))
                        x, y, w, h = cv2.boundingRect(scaled_contours[0])
                        center = (x + w // 2, y + h // 2)
                        axes = (w // 2, h // 2)
                        cv2.ellipse(self.image, center, axes, 0, 0, 360, colour, -1)
                        
    def save(self, output_path):
        self.load_image() 
        self.style() 
        cv2.imwrite(output_path, self.image)



"""
    def _draw_contours(self,image,contour):
        image = np.array(image)
        masks = cv2.imread(mask_path).transpose(2,0,1)
        if masks is None:
            print(f"Warning: Unable to read mask for {image_name}.")
            return False

        masks_split = np.split(masks,masks.shape[0])
        colours = [(0,255,255),(255,255,0),(255,0,255)]

        image_with_contour = image.copy()

        for idx in range(masks.shape[0]):
            mask = masks_split[idx].astype(np.uint8).squeeze()
            contours, _  = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(image_with_contour, contours, -1, colours[idx], 5)
        image_with_contour = Image.fromarray(image_with_contour)
        return image_with_contour


    def resize_width_keep_aspect(self,img, new_width):
        orig_width, orig_height = img.size
        new_height = int((new_width / orig_width) * orig_height)
        return img.resize((new_width, new_height), Image.LANCZOS)


    def _centre_pad(self,image,size):
        width, height = image.size
        image = np.array(image)
        target_height = size[0]
        target_width = size[1]
        top = (target_height - height) // 2
        bottom = target_height - height - top
        left = (target_width - width) // 2
        right = target_width - width - left
        padded_image = cv2.copyMakeBorder(
            image, top, bottom, left, right, 
            borderType=cv2.BORDER_CONSTANT, 
            value=(255, 255, 255)        )
        padded_image = Image.fromarray(padded_image)
        return padded_image

    def extend_image_with_number(self,image, number, font_scale=7, thickness=10):
        # Convert PIL image to OpenCV format if it's a PIL image
        if isinstance(image, Image.Image):
            image = np.array(image)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # Convert RGB (PIL) to BGR (OpenCV)
        
        h, w, c = image.shape
        new_w = int(w * 1.1)  # Extend by 10%
        
        # Create a new white image
        extended_image = np.ones((h, new_w, c), dtype=np.uint8) * 255
        extended_image[:, new_w-w:] = image  # Copy the original image
        
        # Put the number in the center of the new area
        text = str(number)
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
        text_x = (new_w - w - text_size[0]) // 2
        text_y = (h + text_size[1]) // 2
        
        cv2.putText(extended_image, text, (text_x, text_y), font, font_scale, (0, 0, 0), thickness)

        # Convert back to PIL format
        pil_image = Image.fromarray(cv2.cvtColor(extended_image, cv2.COLOR_BGR2RGB))  # Convert BGR to RGB for PIL
        return pil_image
       
    def animate(self,output_dir, directory, image_list, image_sizes, factors,masks,label = True):
        os.makedirs(os.path.join(output_dir),exist_ok=True)
        df = pd.DataFrame(factors)
        df['image_name']=image_list
        factors_list = list(factors.keys())
        if 'time' not in factors_list:
            print('Time is not a factor. Please add "time" as a factor')
            return
        else:
            factors_list.remove('time')
                  
        for group_name, group_df in tqdm(df.groupby(factors_list)):
            images = list(group_df['image_name'])
            time = list(group_df['time'])
            images = [image for _, image in sorted(zip(time,images))]
            time =  [t for t, _ in sorted(zip(time,images))]


            # get the largest size (use padding)
            widths = []
            heights = []
            for image in images:
                size = image_sizes.get(image)
                widths.append(size[0])
                heights.append(size[1])
            
            width = int(np.max(widths))

            height = int(np.max(np.array(heights)/np.array(widths)*width))


            first_image = Image.open(os.path.join(directory,images[0]))
            if masks:
                first_image = self._draw_contours(first_image,masks.get(images[0]))
                first_image = self.resize_width_keep_aspect(first_image,width)
                first_image = self._centre_pad(first_image,(height,width))
                if label:
                    first_image = self.extend_image_with_number(first_image,time[0])
            gif_images = [first_image]

            for i, image in enumerate(images[1:]):
                img = Image.open(os.path.join(directory,image))
                if masks:
                    img = self._draw_contours(img,masks.get(image))
                img  = self.resize_width_keep_aspect(img ,width)
                img = self._centre_pad(img,(height,width)) 
                img = self.extend_image_with_number(img,time[i+1])
                gif_images.append(img)
            
           
            group = "_".join(f"{f}_{v}" for f, v in zip(factors_list, group_name))  
            times = "_".join(f"{t}" for t in time)  
            gif_filename =os.path.join(output_dir,f'{group}'+'_time_'+f'{times}.gif')
        
            gif_images[0].save(
               gif_filename,
                save_all=True,
                append_images=gif_images[1:],  # Append the rest of the images
                duration=500,  # Duration of each frame in milliseconds
                loop=0  # Infinite loop (set to a specific number for a limited number of loops)
            )

        print("GIF created successfully!")
            # group by non-time factors
            #loop over them and add image to animation
            # if masks specfied before add image add contours
            # save animation to output_Dir
                


"""

       
