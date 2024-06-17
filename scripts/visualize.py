import numpy as np
from PIL import Image
import os

# Load the .npz file
data = np.load('/proj/berzelius-2021-89/users/x_tohop/guided-diffusion/samples/test/samples_10x28x28x1.npz')

# Extract the array of images
images = data['arr_0']  # Adjust the key if it's different

# Ensure output directory exists
output_dir = 'output_images'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Save each image as a .png file
for i in range(images.shape[0]):
    # Remove the single color channel dimension
    img = images[i, :, :, 0]
    
    img = Image.fromarray(img, 'L')
    img.save(os.path.join(output_dir, f'sample_{i}.png'))

print(f'Saved {images.shape[0]} images to the {output_dir} directory.')


