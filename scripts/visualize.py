import numpy as np
from PIL import Image
import os
from argparse import ArgumentParser

def main(path):
    data = np.load(path)
    output_dir = os.path.dirname(path)
    images = data['arr_0']

    for i in range(images.shape[0]):

        img = images[i, :, :, 0]
        img = Image.fromarray(img, 'L')
        img.save(os.path.join(output_dir, f'sample_{i}.png'))

    print(f'Saved {images.shape[0]} images to the {output_dir} directory.')

if __name__=="__main__":

    parser = ArgumentParser()
    parser.add_argument("--path", type=str, help="path to the .npz file, where the .png files will be stored")
    args = parser.parse_args()

    main(**vars(args))


