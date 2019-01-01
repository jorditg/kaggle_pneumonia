from __future__ import division
import torch
import math
import random
from PIL import Image, ImageOps, ImageDraw, ImageChops
import numpy as np
import numbers
import types
from random import randint

class RandomVerticalFlip(object):
    """Randomly vertically flips the given PIL.Image with a probability of 0.5
    """
    def __call__(self, img):
        if random.random() < 0.5:
            return img.transpose(Image.FLIP_TOP_BOTTOM)
        return img

class RandomRotation(object):
    """Randomly rotates the given PIL.Image
    """
    def __init__(self, fr=0., to=360.):
        self.delta = to - fr
        self.fr = fr
    def __call__(self,img):
        return img.rotate(self.delta*random.random() + self.fr)

class Random90Rotation(object):
    """Randomly rotates the given PIL.Image
    """
    def __call__(self,img):
        angle = 90.*float(randint(0,3))
        return img.rotate(angle)

class RandomSplitsRotation(object):
    """Randomly rotates in s splits 
    """
    def __init__(self, s):
        self.s = s - 1
        self.angle = 360.0/float(s)
    def __call__(self,img):
        angle = self.angle*float(randint(0,self.s))
        return img.rotate(angle)

class CenteredCircularMask(object):
    def __init__(self, pixels = 0, random = False):
        self.pixels = pixels
        self.random = random
        self.mask = Image.new('1', img.size)
        draw = ImageDraw.Draw(self.mask)
        draw.ellipse((0+pixels,0+pixels,img.size[0]-pixels,img.size[1]-pixels),
                     fill = 'white', outline ='white')
        self.mask = self.mask.convert('RGB')

    def __call__(self,img):
        if self.random:
            # only recalculate mask if random
            pixels = self.pixels            
            rl = random.randint(0, pixels)
            rr = random.randint(0, pixels)
            ru = random.randint(0, pixels)
            rd = random.randint(0, pixels)
            self.mask = Image.new('1', img.size)
            draw = ImageDraw.Draw(self.mask)
            draw.ellipse((0+rl,0+rr,img.size[0]-ru,img.size[1]-rd),
                         fill = 'white', outline ='white')
            self.mask = self.mask.convert('RGB')
        return ImageChops.multiply(img, self.mask)

class CenteredCircularMaskTensor(object):
    def __init__(self, size, val_mask = [], mask_size = 0):
        img_mask = Image.new('1', (size,size))
        draw = ImageDraw.Draw(img_mask)
        # mask size can be fixed to a lower value than image size
        # if mask_size = 0 then same size as image used
        if mask_size != 0:
            border = (size - mask_size)/2
            draw.ellipse((border,border,size-border,size-border),
                         fill = 'white', outline ='white')
        else:            
            draw.ellipse((0,0,size,size),
                        fill = 'white', outline ='white')
                        
        mask = np.asarray(img_mask).astype('float32')
        mask = torch.from_numpy(mask)
        mask = torch.unsqueeze(mask,0).expand(3,size,size)
        self.mask = mask
        self.summask = False
                
        if len(val_mask) > 0:
            self.summask = True
            mask_inverted = torch.from_numpy(np.logical_not(np.asarray(img_mask).astype('bool_')).astype('float32'))
            self.mask_sum = torch.zeros(3,size,size)
            for i in range(len(val_mask)):
                self.mask_sum[i] += val_mask[i]*mask_inverted
        
    def __call__(self,tensor):
        val = torch.mul(tensor, self.mask)
        if self.summask:
            val = val + self.mask_sum
        return val 
        
class AugmentColor(object):
    def __init__(self, gamma, brightness, colors):
        self.gamma = gamma
        self.brightness = brightness
        self.colors = colors

    def __call__(self, img):
        p = np.random.uniform(0, 1, 1)
        if p > 0.5:
            # Randomly shift gamma
            random_gamma = torch.from_numpy(np.random.uniform(\
                                1 - self.gamma, 1+self.gamma, 1))\
                                .type(torch.FloatTensor)
                                #.type(torch.cuda.FloatTensor)
            img  = img  ** random_gamma

        p = np.random.uniform(0, 1, 1)
        if p > 0.5:
            # Randomly shift brightness
            random_brightness =  torch.from_numpy(np.random.uniform(1\
                    / self.brightness, self.brightness, 1))\
                    .type(torch.FloatTensor)
                    #.type(torch.cuda.FloatTensor)
            img  =  img * random_brightness

        p = np.random.uniform(0, 1, 1)
        if p > 0.5:
            # Randomly shift color
            random_colors =  torch.from_numpy(np.random.uniform(1 -\
                    self.colors, 1+self.colors, 3))\
                    .type(torch.FloatTensor)
                    #.type(torch.cuda.FloatTensor)
            white = torch.ones([np.shape(img)[1], np.shape(img)[2]])\
                                .type(torch.FloatTensor)
                                #.type(torch.cuda.FloatTensor)
            color_image = torch.stack([white * random_colors[i]
                                      for i in range(3)], dim=0)
            img  *= color_image

        # Saturate
        img  = torch.clamp(img,  0, 1)
        return img
