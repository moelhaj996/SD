import gradio as gr
import cv2
import numpy as np
from PIL import Image
import torch
from services.tryon_service import TryOnService

# Initialize the service
tryon_service = TryOnService()

def process_image(input_image, clothing_image):
    """Handle the try-on process"""
    try:
        # Process images through our service
        result = tryon_service.process_tryon(input_image, clothing_image)
        return result
    except Exception as e:
        print(f"Error processing images: {str(e)}")
        return input_image  # Return original image in case of error

# Create the Gradio interface
with gr.Blocks(theme=gr.themes.Base()) as demo:
    gr.Markdown("# Dolabk")
    
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(label="Upload your photo", type="numpy")
            clothing_image = gr.Image(label="Upload clothing item", type="numpy")
        
        with gr.Column():
            output_image = gr.Image(label="Result")
    
    submit_btn = gr.Button("Try On")
    submit_btn.click(
        fn=process_image,
        inputs=[input_image, clothing_image],
        outputs=output_image
    )

if __name__ == "__main__":
    demo.launch() 