import deeplake
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch.utils.data._utils.collate import default_collate
import numpy as np
import torch

# Define a transformation to preprocess images
transform = transforms.Compose([
    transforms.Lambda(lambda img: img.convert("RGB")),  # Convert grayscale images to RGB
    transforms.Resize((224, 224)),  # Resize images to 224x224
    transforms.ToTensor(),  # Convert PIL images to tensors
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize to [-1, 1]
])

def custom_collate_fn(batch):
    print("\n--- DEBUG: Original Batch Structure ---")
    for i, item in enumerate(batch):
        print(f"Sample {i}: Keys -> {item.keys()}")
        print(f"Sample {i}: Image Type -> {type(item['images'])}")
        print(f"Sample {i}: Metadata -> { {k: item[k] for k in item.keys() if k != 'images'} }\n")

    batch_dict = {"images": [], "findings": [], "boxes/bbox": [], "boxes/finding": []}
    
    metadata_fields = [
        "metadata/patient_id", "metadata/patient_age", "metadata/patient_gender",
        "metadata/follow_up_num", "metadata/view_position",
        "metadata/orig_img_w", "metadata/orig_img_h",
        "metadata/orig_img_pix_spacing_x", "metadata/orig_img_pix_spacing_y"
    ]
    
    for field in metadata_fields:
        batch_dict[field] = []

    max_findings = 0  # Track max findings length
    max_boxes = 0  # Track max bounding boxes

    for item in batch:
        batch_dict["images"].append(transform(item["images"]))  # Transform images
        
        # Convert `findings` field to tensor (ensure same shape)
        findings_tensor = torch.tensor(item["findings"], dtype=torch.long)
        batch_dict["findings"].append(findings_tensor)
        max_findings = max(max_findings, findings_tensor.shape[0])  # Track max findings length

        # Handle bounding boxes
        bbox_tensor = torch.tensor(item["boxes/bbox"]) if item["boxes/bbox"].size else torch.zeros((0, 4))
        batch_dict["boxes/bbox"].append(bbox_tensor)
        batch_dict["boxes/finding"].append(torch.tensor(item["boxes/finding"], dtype=torch.long))
        max_boxes = max(max_boxes, bbox_tensor.shape[0])

        for field in metadata_fields:
            value = torch.tensor(item[field])
            if value.numel() == 1:  # Ensure scalar values remain scalars
                value = value.squeeze(0)
            batch_dict[field].append(value)

    # Pad findings so all samples have the same length
    for i in range(len(batch)):
        findings_tensor = batch_dict["findings"][i]
        if findings_tensor.shape[0] < max_findings:
            pad_tensor = torch.full((max_findings - findings_tensor.shape[0],), -1)  # Use -1 for padding
            batch_dict["findings"][i] = torch.cat([findings_tensor, pad_tensor], dim=0)

        # Pad bounding boxes
        bbox_tensor = batch_dict["boxes/bbox"][i]
        if bbox_tensor.shape[0] < max_boxes:
            pad_tensor = torch.zeros((max_boxes - bbox_tensor.shape[0], 4))
            batch_dict["boxes/bbox"][i] = torch.cat([bbox_tensor, pad_tensor], dim=0)

        # Pad bounding box findings
        bbox_finding_tensor = batch_dict["boxes/finding"][i]
        if bbox_finding_tensor.shape[0] < max_boxes:
            pad_tensor = torch.full((max_boxes - bbox_finding_tensor.shape[0],), -1, dtype=torch.long)  # Use -1 for padding
            batch_dict["boxes/finding"][i] = torch.cat([bbox_finding_tensor, pad_tensor], dim=0)

    # Convert lists to tensors using `default_collate`
    for key in batch_dict:
        batch_dict[key] = default_collate(batch_dict[key])

    print("\n--- DEBUG: Final Batch Structure After Collation ---")
    print(f"Final Batch Keys: {batch_dict.keys()}")

    return batch_dict

def load_dataset(batch_size=32):
    """
    Loads the NIH Chest X-ray dataset from DeepLake and returns a PyTorch DataLoader.
    """
    ds = deeplake.load("hub://activeloop/nih-chest-xray-test")  # Load dataset
    
    # Use DeepLake's built-in PyTorch support
    dataloader = ds.pytorch(
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=0,  # Set to 0 to avoid multiprocessing issues on Windows
        decode_method={"images": "pil"},  # Faster image decoding
        collate_fn=custom_collate_fn  # Use custom collate function
    )
    
    return dataloader