import cv2
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms

class ImagePreprocessor:
    def __init__(self):
        self.transform = transforms.Compose([
            transforms.Resize((512, 384)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
    
    def preprocess_person_image(self, image):
        """Preprocess person image for try-on"""
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        return self.transform(image)
    
    def preprocess_clothing(self, image):
        """Preprocess clothing image for try-on"""
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        return self.transform(image) 