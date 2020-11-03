from PIL import Image
import numpy as np
import rasterio as rs
import math
from rasterio.plot import show
import xarray as xr
import georaster
import matplotlib.pyplot as plt

#####################################################
########## SCRIPT TO DISPLAY A .GEOTIF IMG ##########
##########   OR SAVE IMAGE MATRIX AS .TXT  ##########
#####################################################

# Takes in the band of an image and saves the matrix
# for that band to a .txt file
def make_matrix(i):
    output_dir = "Extracted/LC080900862019072401T1-SC20200930162234/"

    img = Image.open(output_dir + 'LC08_L1TP_090086_20190724_20190801_01_T1_sr_band' + str(i) + '.tif')

    mat = np.matrix(img)
    with open('band'+str(i)+'.txt','wb') as f:
        for line in mat:
            np.savetxt(f, line, fmt='%.2f')

# Using Rasterio will plot an image  
def view_image(image_dir):

    # Show the image:
    fp = r'/Users/calummcmeekin/Downloads/LE07_221077_19990901-0000000000-0000000000.tif'
    img = rs.open(fp)

    # Print number of bands in the image:

    show(img.read([1,2,3]))

if __name__ == "__main__":

    MapBiome_data = '/Users/calummcmeekin/Downloads/mosaic-2019-0000000000-0000000000.tif'
    Landsat_image = '/Users/calummcmeekin/Downloads/LE07_220076_20000608.tif'

    view_image(Landsat_image)
