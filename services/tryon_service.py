import torch
from models.preprocessing import ImagePreprocessor
from models.tryon_model import TryOnModel
import numpy as np

class TryOnService:
    def __init__(self):
        self.preprocessor = ImagePreprocessor()
        self.model = TryOnModel()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
    
    def process_tryon(self, person_image, clothing_image):
        """Process the virtual try-on"""
        # Preprocess images
        person_tensor = self.preprocessor.preprocess_person_image(person_image)
        clothing_tensor = self.preprocessor.preprocess_clothing(clothing_image)
        
        # Add batch dimension
        person_tensor = person_tensor.unsqueeze(0).to(self.device)
        clothing_tensor = clothing_tensor.unsqueeze(0).to(self.device)
        
        # Generate try-on result
        with torch.no_grad():
            output = self.model(person_tensor, clothing_tensor)
        
        # Convert to numpy array
        output = output.squeeze(0).cpu()
        output = output.permute(1, 2, 0).numpy()
        
        # Post-process
        output = ((output + 1) * 127.5).astype(np.uint8)
        
        return output 