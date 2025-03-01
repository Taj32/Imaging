from src.data_preprocessing.load_data import load_dataset
from src.data_preprocessing.preprocess import preprocess_dataset
import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":  # ✅ Prevents multiprocessing error on Windows
    # Load dataset
    dataloader = load_dataset(batch_size=5)

    # Get the first batch (first sample)
    print("here first")
    first_batch = next(iter(dataloader))  # Extract one batch
    print(first_batch.keys())  # Confirm keys exist
    print(type(first_batch["images"]))  # Check image data type after transformation

    
    print("here....")

    # Extract image and patient age
    image = first_batch["images"][0].numpy().transpose(1, 2, 0)  # Convert tensor to NumPy array and transpose to (H, W, C)
    patient_age = first_batch["metadata/patient_age"][0].item()  # Convert tensor to scalar

    # Display the image
    plt.imshow(image, cmap="gray")  # X-rays are grayscale
    plt.title(f"Patient Age: {patient_age}")
    plt.axis("off")
    plt.show()

    print("temp end 1")
    quit()
 
    # Preprocess dataset
    processed_data = preprocess_dataset(dataloader)

    print("temp end")
    quit()
    # Print sample output
    print("Processed Data Sample:", processed_data[0])