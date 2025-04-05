import os
import numpy as np
import torch
import torch.nn as nn
from torchvision import models
import rasterio
from tkinter import Tk, filedialog
import matplotlib.pyplot as plt

# Configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
REQUIRED_FILES = ["RED.tif", "GRN.tif", "BLUE.tif"]

def select_file(title="Select file"):
    """Opens a file selection dialog"""
    root = Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(title=title)
    root.destroy()
    return file_path

def select_folder():
    """Opens a folder selection dialog"""
    root = Tk()
    root.withdraw()
    folder = filedialog.askdirectory(title="Select folder with image files")
    root.destroy()
    return folder

def load_model(model_path):
    """
    Loads the model, completely ignoring the aux_classifier
    """
    # Create the model without aux_classifier
    model = models.segmentation.deeplabv3_resnet50(
        pretrained=False,
        num_classes=1,   # For binary segmentation
        aux_loss=False   # Disable aux_classifier
    )

    # Load only compatible parameters
    state_dict = torch.load(model_path, map_location=device)

    # Remove all parameters of aux_classifier
    state_dict = {k: v for k, v in state_dict.items()
                    if not k.startswith('aux_classifier')}

    # Load the remaining parameters
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model.to(device)

class ImageProcessor:
    def __init__(self, folder):
        self.folder = folder
        self.red_img = None
        self.grn_img = None
        self.blue_img = None
        self.rgb_image = None
        self.load_images()

    def load_images(self):
        """Loads images from the specified folder"""
        bands = []
        for filename in REQUIRED_FILES:
            filepath = os.path.join(self.folder, filename)
            if not os.path.exists(filepath):
                raise FileNotFoundError(f"File {filename} not found in {self.folder}")
            with rasterio.open(filepath) as src:
                bands.append(src.read(1))
        self.red_img, self.grn_img, self.blue_img = bands
        self.rgb_image = np.stack(bands, axis=-1)

    def get_full_image_tensor(self):
        """Returns the full image as a tensor"""
        img_tensor = np.stack([self.red_img, self.grn_img, self.blue_img], axis=0)
        return torch.FloatTensor(img_tensor).unsqueeze(0)

def process_image(model, image_processor, threshold=0.5):
    """Processes the image using the model"""
    full_image_tensor = image_processor.get_full_image_tensor().to(device)
    original_height, original_width = image_processor.red_img.shape

    with torch.no_grad():
        output = model(full_image_tensor)['out']
        pred = torch.sigmoid(output).cpu().squeeze()

    binary_mask = (pred > threshold).numpy().astype(np.uint8)
    return binary_mask[:original_height, :original_width]

def visualize_results(image_processor, prediction_mask):
    """Visualizes the results"""
    plt.figure(figsize=(20, 10))

    plt.subplot(1, 2, 1)
    plt.imshow(image_processor.rgb_image)
    plt.title("Original Image")

    plt.subplot(1, 2, 2)
    plt.imshow(image_processor.rgb_image)
    plt.imshow(prediction_mask, alpha=0.3, cmap='jet')
    plt.title("Predicted Areas (overlay)")

    plt.tight_layout()
    plt.show()


def main():
    # 1. Load the model
    model_path = select_file(title="Select trained model file (.pth)")
    if not model_path:
        print("No model selected. Exiting.")
        return

    model = load_model(model_path)

    # 2. Load images for prediction
    folder = select_folder()
    if not folder:
        print("No folder selected. Exiting.")
        return

    try:
        processor = ImageProcessor(folder)
        print("Images loaded successfully.")

        # 3. Process the image
        print("Processing image...")
        prediction = process_image(model, processor)

        # 4. Visualize the results
        visualize_results(processor, prediction)

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()