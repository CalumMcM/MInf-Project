from skimage.transform import resize
import rasterio as rs
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from rasterio.plot import show
import matplotlib

def load_rs_img(img_file, val_type='unit8', bands_only=False, num_bands=3):
    """
    Will open an image using rasterio
    """

    # Open the iamge
    img = rs.open(img_file)

    # Print number of bands in the image:
    img = img.read([1,2,3])

    return img

DIR = r'/Users/calummcmeekin/Documents/GitHub/MInf-Project/SatelliteSegmentation/tile2vec/data/Caatinga/Quad 3/5.tif'

img = load_rs_img(DIR)
show(img)
x, y, z = img.shape
img = img.reshape(y,z,x)

dmg = resize(img, (224, 224,x))

show(dmg.reshape(x,224,224))

matplotlib.image.imsave('original_image.png', img)
matplotlib.image.imsave('resized_image.png', dmg)
