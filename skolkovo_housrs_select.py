import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from tqdm import tqdm
import matplotlib.pyplot as plt
import rasterio
from tkinter import Tk, filedialog
from matplotlib.widgets import Button

# Configuration
TILE_SIZE = 512
BATCH_SIZE = 4 
EPOCHS = 1000
LR = 1e-4
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
REQUIRED_FILES = ["RED.tif", "GRN.tif", "BLUE.tif", "all.tif"]

def select_folder():
    """Opens a folder selection dialog"""
    root = Tk()
    root.withdraw()
    folder = filedialog.askdirectory(title="Select folder with RED.tif, GRN.tif, BLUE.tif, all.tif")
    root.destroy()
    return folder

class DataSelector:
    def __init__(self, folder):
        self.folder = folder
        self.train_coords = []
        self.val_coords = []
        self.red_img = None
        self.grn_img = None
        self.blue_img = None
        self.mask_img = None
        self.rgb_image = None
        self.load_images()

    def load_images(self):
        """Loads all necessary images"""
        bands = []
        for filename in REQUIRED_FILES[:-1]:  # Load RED, GRN, BLUE
            filepath = os.path.join(self.folder, filename)
            if not os.path.exists(filepath):
                raise FileNotFoundError(f"File {filename} not found in folder {self.folder}")
            with rasterio.open(filepath) as src:
                bands.append(src.read(1))
        self.red_img, self.grn_img, self.blue_img = bands
        self.rgb_image = np.stack(bands, axis=-1)

        mask_filename = REQUIRED_FILES[-1]
        mask_filepath = os.path.join(self.folder, mask_filename)
        if not os.path.exists(mask_filepath):
            raise FileNotFoundError(f"File {mask_filename} not found in folder {self.folder}")
        with rasterio.open(mask_filepath) as mask:
            self.mask_img = mask.read(1)

    def select_areas(self, selection_type="train"):
        """Interactive selection of multiple areas"""
        fig, ax = plt.subplots(figsize=(12, 12))
        ax.imshow(self.rgb_image)
        ax.set_title(f"Click to select areas (512x512) for {selection_type}. Close window when done.")
        current_coords = self.train_coords if selection_type == "train" else self.val_coords

        def onclick(event):
            if event.xdata and event.ydata:
                x, y = int(event.xdata), int(event.ydata)
                x1 = max(0, x - TILE_SIZE//2)
                y1 = max(0, y - TILE_SIZE//2)
                x2 = x1 + TILE_SIZE
                y2 = y1 + TILE_SIZE

                # Boundary check
                if x2 > self.rgb_image.shape[1]:
                    x2 = self.rgb_image.shape[1]
                    x1 = x2 - TILE_SIZE
                if y2 > self.rgb_image.shape[0]:
                    y2 = self.rgb_image.shape[0]
                    y1 = y2 - TILE_SIZE

                # Add coordinates and draw a rectangle
                current_coords.append((x1, y1, x2, y2))
                rect = plt.Rectangle((x1, y1), TILE_SIZE, TILE_SIZE,
                                     linewidth=2, edgecolor='r', facecolor='none')
                ax.add_patch(rect)
                plt.draw()

        fig.canvas.mpl_connect('button_press_event', onclick)
        plt.show()

        return current_coords

    def get_tiles(self, selection_type="train"):
        """Returns the selected tiles for the specified type (train or val)"""
        img_tiles = []
        mask_tiles = []
        coords = self.train_coords if selection_type == "train" else self.val_coords

        for x1, y1, x2, y2 in coords:
            img_tile = np.stack([
                self.red_img[y1:y2, x1:x2],
                self.grn_img[y1:y2, x1:x2],
                self.blue_img[y1:y2, x1:x2]
            ], axis=0)

            mask_tile = self.mask_img[y1:y2, x1:x2]

            img_tiles.append(img_tile)
            mask_tiles.append(mask_tile)

        return np.array(img_tiles), np.array(mask_tiles)

# Dataset
class MultiTileDataset(Dataset):
    def __init__(self, img_tiles, mask_tiles):
        self.img_tiles = torch.FloatTensor(img_tiles)
        self.mask_tiles = torch.FloatTensor(mask_tiles)

    def __len__(self):
        return len(self.img_tiles)

    def __getitem__(self, idx):
        return self.img_tiles[idx], self.mask_tiles[idx]

# Model (no changes)
def create_model():
    model = models.segmentation.deeplabv3_resnet50(pretrained=True)
    model.classifier[4] = nn.Conv2d(256, 1, kernel_size=1)
    return model.to(device)

def calculate_iou(preds, targets):
    intersection = (preds * targets).sum()
    union = (preds + targets).sum() - intersection
    iou = (intersection + 1e-7) / (union + 1e-7)  # Add 1e-7 to avoid division by zero
    return iou

# Training with validation
def train(model, train_loader, val_loader, epochs=10):
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.BCEWithLogitsLoss()

    best_val_iou = 0.0
    for epoch in tqdm(range(epochs), desc="Training"):
        # Training
        model.train()
        train_loss = 0.0
        train_iou = 0.0
        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            outputs = model(images)['out']
            loss = criterion(outputs, masks.unsqueeze(1))
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                preds = torch.sigmoid(outputs)
                iou = calculate_iou(preds.round(), masks.unsqueeze(1))
            train_loss += loss.item()
            train_iou += iou.item()
        avg_train_loss = train_loss / len(train_loader)
        avg_train_iou = train_iou / len(train_loader)

        # Validation
        model.eval()
        val_loss = 0.0
        val_iou = 0.0
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)['out']
                loss = criterion(outputs, masks.unsqueeze(1))
                preds = torch.sigmoid(outputs)
                iou = calculate_iou(preds.round(), masks.unsqueeze(1))
                val_loss += loss.item()
                val_iou += iou.item()
        avg_val_loss = val_loss / len(val_loader)
        avg_val_iou = val_iou / len(val_loader)

        tqdm.write(
            f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.4f} | Train IoU: {avg_train_iou:.4f} | "
            f"Val Loss: {avg_val_loss:.4f} | Val IoU: {avg_val_iou:.4f}"
        )

        # Save the best model based on validation IoU
        if avg_val_iou > best_val_iou:
            best_val_iou = avg_val_iou
            torch.save(model.state_dict(), "best_model.pth")
            tqdm.write(f"New best model saved with Val IoU: {best_val_iou:.4f}")

# Visualization of results (can be improved for validation data)
def visualize_results(model, dataset, num_samples=3):
    model.eval()
    indices = np.random.choice(len(dataset), min(num_samples, len(dataset)), replace=False)

    plt.figure(figsize=(15, 5*num_samples))
    for i, idx in enumerate(indices, 1):
        img, mask = dataset[idx]

        with torch.no_grad():
            pred = model(img.unsqueeze(0).to(device))['out']
            pred = torch.sigmoid(pred).cpu().squeeze()

        plt.subplot(num_samples, 3, 3*i-2)
        plt.imshow(img.permute(1, 2, 0))
        plt.title(f"Sample {i} - Input")

        plt.subplot(num_samples, 3, 3*i-1)
        plt.imshow(mask, cmap='gray')
        plt.title("Ground Truth")

        plt.subplot(num_samples, 3, 3*i)
        plt.imshow(pred > 0.5, cmap='gray')
        plt.title("Prediction")

    plt.tight_layout()
    plt.show()

# Main process
def main():
    # 1. Folder selection
    print("Select folder containing RED.tif, GRN.tif, BLUE.tif, all.tif")
    folder = select_folder()
    if not folder:
        print("No folder selected. Exiting.")
        return

    # 2. Selection of areas for training
    selector = DataSelector(folder)
    print("Select multiple areas for training by clicking. Close window when done.")
    try:
        selector.select_areas(selection_type="train")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    if not selector.train_coords:
        print("No training areas selected. Exiting.")
        return

    # 3. Selection of areas for validation
    print("Select multiple areas for validation by clicking. Close window when done.")
    selector.select_areas(selection_type="val")

    if not selector.val_coords:
        print("No validation areas selected. Continuing without validation.")
        img_tiles_train, mask_tiles_train = selector.get_tiles(selection_type="train")
        train_dataset = MultiTileDataset(img_tiles_train, mask_tiles_train)
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
        val_loader = None
        print(f"Selected {len(train_dataset)} tiles for training.")
    else:
        # 4. Data preparation
        img_tiles_train, mask_tiles_train = selector.get_tiles(selection_type="train")
        img_tiles_val, mask_tiles_val = selector.get_tiles(selection_type="val")

        train_dataset = MultiTileDataset(img_tiles_train, mask_tiles_train)
        val_dataset = MultiTileDataset(img_tiles_val, mask_tiles_val)

        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)

        print(f"Selected {len(train_dataset)} tiles for training.")
        print(f"Selected {len(val_dataset)} tiles for validation.")

    # 5. Model training
    model = create_model()
    train(model, train_loader, val_loader, epochs=EPOCHS)

    # 6. Visualization of results
    if train_dataset:
        visualize_results(model, train_dataset)

    print("Training complete!")

if __name__ == "__main__":
    main()