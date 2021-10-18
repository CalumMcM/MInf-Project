import numpy as np
import os

"""
For a given directory of images belonging to one class,
generates a label file for all the images in the directory
"""
def getImages(DIR):

    img_names = []

    for filename in os.listdir(DIR):
        if filename.endswith('.tif'):
            img_names.append(filename)

    return img_names

DIR = r'/Users/calummcmeekin/Documents/GitHub/MInf-Project/SatelliteSegmentation/tile2vec/data/Caatinga/Quad 3'

imgs = getImages(DIR)

# Put class here, 0 = Amazon, 1 = Cerrado, 2 = Caatinga
arr = [2]*len(imgs)

np.save(os.path.join(DIR, 'labels.npy'), arr)
