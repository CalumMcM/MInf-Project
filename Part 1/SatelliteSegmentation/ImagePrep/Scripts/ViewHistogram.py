import matplotlib.pyplot as plt
from matplotlib import pyplot
import rasterio as rs
from rasterio.plot import show
import numpy as np

import os

def viewHist(images, DIR):
    """
    Will open the given image at the given
    directory using Rasterio and plot it
    as a histogram
    """
    f,a = plt.subplots(2,2)
    a = a.ravel()

    for idx,ax in enumerate(a):

        # Open the iamge
        img = rs.open(os.path.join(DIR, images[idx]))

        # Extract RGB bands
        img = img.read([1,2,3])

        img = remove_nan(img)

        img_max = np.max(img)
        img_min = np.min(img)

        """
        print ("\n\n")
        print (images[idx])
        print (img_max)
        print (img_min)
        print ("STD: {:.2f}".format(np.std(img)))
        """

        ax.hist(img[2], bins=10)
        ax.hist(img[1], bins=10)
        ax.hist(img[0], bins=10)
        ax.set_xticks([-0.1, -0.05, 0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3])
        ax.set_xticklabels(ax.get_xticks(), rotation = 45)
        ax.set_title(images[idx])

    plt.tight_layout()
    plt.savefig('ClusterHistogram.png')
    plt.show()

def remove_nan(new_tile):
    """
    Takes a tile and replaces any nan values with the mean
    of the image the nan appears in
    """

    mean = np.nanmean(new_tile)

    new_tile = np.nan_to_num(new_tile, nan=mean, posinf=mean, neginf=mean)

    return new_tile

def main():

    DIR = r'/Users/calummcmeekin/Documents/GitHub/MInf-Project/SatelliteSegmentation/tile2vec/data/Example Quad/Quad 2/'

    corrupt_images = ['1305.tif', '590.tif', '3496.tif', '3098.tif']
    good_images =  ['473.tif', '2016.tif', '5882.tif', '5160.tif']


    viewHist(corrupt_images, DIR)

    viewHist(good_images, DIR)



if __name__ == "__main__":
    main()
