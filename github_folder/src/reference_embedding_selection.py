import os
from glob import glob
import tifffile
import h5py
import json
import random

from .setup import dataset_path, directory_path
from .utils import display_image

def main():
    files = sorted(os.listdir(dataset_path), key=os.path.basename)
    random.shuffle(files)

    coords_file = os.path.join(directory_path, "coords.json")

    coordinates = {}
    count = 0

    for file in files:
        _, file_extension = os.path.splitext(file)
        if file_extension == '.tiff':
            image = tifffile.imread(os.path.join(dataset_path, file)) # type: ignore
        elif file_extension == '.hdf5':
            with h5py.File(os.path.join(dataset_path, file), 'r') as f: # type: ignore
                image = f['volumes/raw'][:] # type: ignore
        else:
            raise ValueError(f'Unknown file extension: {file_extension}')
        
        display_image(image)

        coords = input('Enter coordinates in (z, y, x) format (press enter to skip): ')
        if len(coords) != 0:
            coords = coords.split(',')
            coords = [c for c in coords]
            coordinates[file] = coords
            count += 1
            print(f'You have selected {count} files')
            if count == 10:
                print('You have selected 10 files, exiting')
                break

    with open(coords_file, 'w') as f:
        json.dump(coords, f, indent=2)

if __name__ == '__main__':
    main()