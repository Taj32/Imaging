import deeplake
ds = deeplake.load('hub://activeloop/nih-chest-xray-test')

# Filter the data to include only images with 'Pneumonia' in the 'Finding Labels' column
ds.visualize()

print("End")