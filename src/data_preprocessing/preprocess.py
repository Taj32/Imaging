import torch
import numpy as np

def preprocess_sample(image, width, height):
    """Normalize width and height."""
    
    # Normalize width & height (assuming max size is 1024)
    width = width / 1024.0
    height = height / 1024.0

    return image, width, height

def preprocess_dataset(dataloader):
    """Preprocess dataset using batch-wise processing."""
    processed_data = []
    
    for batch in dataloader:  # ✅ Efficient batch loading
        images = batch["images"]  # Already tensors
        widths = batch["metadata/orig_img_w"]  # Keep as tensor
        heights = batch["metadata/orig_img_h"]
        labels = batch["findings"]  # Keep as tensor

        for i in range(len(images)):
            image = images[i]
            width = widths[i].item()  # Convert to scalar
            height = heights[i].item()
            label = labels[i].tolist()  # Convert to list (handles multi-label cases)

            image, width, height = preprocess_sample(image, width, height)
            processed_data.append((image, width, height, label))

    return processed_data
