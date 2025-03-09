from src.data_preprocessing.load_data import load_dataset
from src.data_preprocessing.preprocess import preprocess_dataset
import matplotlib.pyplot as plt
import numpy as np
import sys
import os
import deeplake

# Define the path to save the DeepLake dataset
dataset_path = "my_xray_dataset"

if __name__ == "__main__":  # ✅ Prevents multiprocessing error on Windows
    # Load dataset with batch size 32 and shuffle
    dataloader = load_dataset(batch_size=32)

    # Check if the DeepLake dataset exists
    if os.path.exists(dataset_path):
        # Delete the existing dataset to create a new one
        deeplake.delete(dataset_path)
        print("Existing DeepLake dataset deleted.")

    # Limit the number of batches to process
    max_batches = 5  # Adjust this number to process more or fewer batches
    processed_data = []

    # Iterate over the items in the DataLoader
    for i, batch in enumerate(dataloader):
        if i >= max_batches:
            break

        print(f"Batch {i}:")
        print(f"Image shape: {batch['images'].shape}")
        print(f"Findings shape: {batch['findings'].shape}")
        print(f"Bounding boxes shape: {batch['boxes/bbox'].shape}")
        print(f"Bounding box findings shape: {batch['boxes/finding'].shape}")
        for field in batch.keys():
            if field.startswith("metadata"):
                print(f"{field} shape: {batch[field].shape}")

        # Check for discrepancies in image shapes
        if batch['images'].shape != (32, 3, 224, 224):
            print(f"Discrepancy found in batch {i} with image shape: {batch['images'].shape}")
            break

        processed_data.extend(preprocess_dataset([batch]))

    print("Finished iterating over the DataLoader.")

    # Print sample output
    if processed_data:
        print("Processed Data Sample:", processed_data[0])
    else:
        print("No processed data found.")

    # Create a new DeepLake dataset
    ds = deeplake.empty(dataset_path)
    
    # Define the schema for the dataset
    ds.create_tensor("images", htype="image", dtype="float32", sample_compression="jpeg")
    ds.create_tensor("findings", htype="generic", dtype="int64")
    ds.create_tensor("boxes/bbox", htype="bbox", dtype="float32")
    ds.create_tensor("boxes/finding", htype="generic", dtype="int64")
    
    # Add metadata fields
    metadata_fields = [
        "metadata/patient_id", "metadata/patient_age", "metadata/patient_gender",
        "metadata/follow_up_num", "metadata/view_position",
        "metadata/orig_img_w", "metadata/orig_img_h",
        "metadata/orig_img_pix_spacing_x", "metadata/orig_img_pix_spacing_y"
    ]
    for field in metadata_fields:
        ds.create_tensor(field, htype="generic", dtype="int64")
    
    # Save the processed data to the DeepLake dataset
    for item in processed_data:
        item_dict = {
            "images": item["images"],
            "findings": item["findings"],
            "boxes/bbox": item["boxes/bbox"],
            "boxes/finding": item["boxes/finding"],
            **{field: item[field] for field in metadata_fields}
        }
        ds.append(item_dict)
    
    print("Processed data saved to DeepLake dataset")

    # Print sample output
    if len(ds) > 0:
        print("Processed Data Sample:", ds[0])
    else:
        print("No data found in the DeepLake dataset.")

    # Now you can proceed with training the model using the processed data
    # Example code for training the model
    # ...