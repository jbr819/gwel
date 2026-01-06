from gwel.network import Segmenter
import torch
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
from pycocotools import mask as mask_utils
import yaml

class UNet(nn.Module):
    def __init__(self, in_channels, out_channels, max_pool : int = 2):
        super(UNet, self).__init__()
        if max_pool % 2 == 1:
            raise ValueError("max_pool must be even")            
        self.max_pool = max_pool

        # Contracting path (Encoder)
        self.encoder1 = self.conv_block(in_channels, 64)
        self.encoder2 = self.conv_block(64, 128)
        self.encoder3 = self.conv_block(128, 256)
        self.encoder4 = self.conv_block(256, 512)

        # Expanding path (Decoder)
        self.decoder4 = self.upconv_block(512, 256)
        self.decoder3 = self.upconv_block(256, 128)
        self.decoder2 = self.upconv_block(128, 64)
        self.decoder1 = self.out_layer(64, out_channels)
        

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=self.max_pool,stride=int(self.max_pool/2),padding=1)
        )

    def upconv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=self.max_pool, stride=int(self.max_pool/2),padding=1),
            nn.ReLU(inplace=True),
	    nn.Conv2d(out_channels,out_channels,kernel_size=3,padding=1),
            nn.ReLU(inplace=True)
        )
    
    def out_layer(self,in_channels,out_channels):
        return nn.Sequential(
	    nn.ConvTranspose2d(in_channels,out_channels,kernel_size=4,stride = 2,padding=1),
	    nn.Conv2d(out_channels,out_channels,kernel_size=3,padding=1)
        )

    def forward(self, x):
        # Encoder
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)

        # Decoder
        d3 = self.decoder4(e4)
        d2 = self.decoder3(d3 + e3)  
        d1 = self.decoder2(d2 + e2)   
        out = self.decoder1(d1 + e1)   
        return out

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
        

        _, height, width = tensor.shape 

        pad_h = (self.target_multiple - height % self.target_multiple) % self.target_multiple
        pad_w = (self.target_multiple - width % self.target_multiple) % self.target_multiple
        
        pad_h_top = pad_h // 2
        pad_h_bottom = pad_h - pad_h_top
        pad_w_left = pad_w // 2
        pad_w_right = pad_w - pad_w_left
        
        padded_tensor = torch.nn.functional.pad(tensor, (pad_w_left, pad_w_right, pad_h_top, pad_h_bottom),
                                                value=0)
        return padded_tensor



class UNET(Segmenter):
    def __init__(self,
                 weights : str ,
                 patch_size : int,
                 channels : str):
        
        with open(channels) as f:
            self.channels = yaml.safe_load(f)

        self.model = UNet(3,len(self.channels)+1)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
        self.set_device(self.device)
        self.weights = weights
        self.load_weights( self.weights)
        self.patch_size = patch_size
        self.centerpad = CenterPad(patch_size)
           
    def set_device(self, device : str):
        self.device = device
        self.model.to(device)

    def load_weights(self, weights : str = None):
        self.model.load_state_dict(torch.load(weights, map_location=self.device,weights_only=True)['model_state_dict'])
        self.model.eval()


    def inference(self, image : np.ndarray):
 
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = torch.as_tensor(image,dtype=torch.float).permute(2,0,1) / 255
        image_padded = self.centerpad(image)
        image_padded = image_padded.to(self.device)
        
        with torch.no_grad():
            output = self.model(image_padded)
            #output = torch.sigmoid(output)
            output =  torch.argmax(output,dim=0)
        output = output.cpu().numpy()


        _, crop_height, crop_width = image.shape
        height, width = output.shape
        left = (width - crop_width) // 2
        top = (height - crop_height) // 2
        right = (width + crop_width) // 2
        bottom = (height + crop_height) // 2 
        
        cropped_masks = output[top:bottom, left:right]
        
        output = {}
        
        for class_name, channels in self.channels.items():
            mask = np.isin(cropped_masks, list(channels)).astype(np.uint8)

            rle = mask_utils.encode(np.asfortranarray(mask))
            output[class_name] = rle

        return output
