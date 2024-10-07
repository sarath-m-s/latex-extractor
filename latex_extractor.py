from transformers import AutoTokenizer, VisionEncoderDecoderModel, AutoProcessor
from PIL import Image
import torch

# Load the model and processor
processor = AutoProcessor.from_pretrained("Norm/nougat-latex-base")
model = VisionEncoderDecoderModel.from_pretrained("Norm/nougat-latex-base")

# Preprocess the image
def preprocess_image(image_path):
    image = Image.open(image_path).convert("RGB")
    # Use the processor to process the image
    inputs = processor(images=image, return_tensors="pt")
    return inputs

# Run inference
def get_latex_from_image(image_path, max_length=50):
    inputs = preprocess_image(image_path)
    # Get the model output with specified max_length
    outputs = model.generate(**inputs, max_length=max_length)
    # Decode the output tokens to get the LaTeX string
    latex_output = processor.tokenizer.decode(outputs[0], skip_special_tokens=True)
    return latex_output

# Example usage
image_path = "./example2.png"  # Update this path to your image file
latex_output = get_latex_from_image(image_path, max_length=500)  # Adjust max_length as needed
print("LaTeX Output:", latex_output)