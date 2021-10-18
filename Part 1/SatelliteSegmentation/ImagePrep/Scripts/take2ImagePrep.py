from os import listdir
from glob import glob
import matplotlib.pyplot as plt
import rasterio as rio
from rasterio.plot import plotting_extent
import geopandas as gpd
import earthpy as et
import earthpy.spatial as es
import earthpy.plot as ep
import cv2
import numpy as np
from PIL import Image

#####################################################
##########  FILE USED TO EXPLORE MULTIPLE  ##########
########## IMAGE PRE-PROCESSING TECHNIQUES ##########
#####################################################

def normalize(arr):
    """
    Linear normalization
    http://en.wikipedia.org/wiki/Normalization_%28image_processing%29
    """
    arr = arr.astype('float')
    # Do not touch the alpha channel
    for i in range(3):
        minval = arr[...,i].min()
        maxval = arr[...,i].max()
        if minval != maxval:
            arr[...,i] -= minval
            arr[...,i] *= (255.0/(maxval-minval))
    return arr

def demo_normalize(dir):
    img = Image.open(dir+"/ImageMerge.tif").convert('RGBA')
    arr = np.array(img)
    new_img = Image.fromarray(normalize(arr).astype('uint8'),'RGBA')
    new_img.save('/tmp/normalized.png')

def image_prep3(upper_dir):
    dir = "Extracted/" + upper_dir + '/'

    blue = Image.open(dir+"LC08_L1TP_090086_20190724_20190801_01_T1_sr_band2.tif").convert('L')
    green = Image.open(dir+"LC08_L1TP_090086_20190724_20190801_01_T1_sr_band3.tif").convert('L')
    red = Image.open(dir+"LC08_L1TP_090086_20190724_20190801_01_T1_sr_band4.tif").convert('L')

    out = Image.merge("RGB", (red, green, blue))

    out.save(dir+"/ImageMerge.tif")


def image_prep(upper_dir, wanted_files):

    output_dir = "Extracted/" + upper_dir + '/output3.tiff'

    landsat_bands_data_path = "Extracted/" + upper_dir + "/" + wanted_files + "*[1-7]*.tif"

    stack_band_paths = glob(landsat_bands_data_path)
    stack_band_paths.sort()

    # Create image stack and apply nodata value for Landsat
    arr_st, meta = es.stack(stack_band_paths, nodata=-9999, out_path = output_dir)


        # Create figure with one plot
    fig, ax = plt.subplots(figsize=(12, 12))

    # Plot red, green, and blue bands, respectively
    ep.plot_rgb(arr_st, rgb=(3, 2, 1), ax=ax, title="Landsat 8 RGB Image")
    #plt.savefig("Extracted/" + upper_dir + '/output2.tiff')
    #plt.show()


def get_dir_frame(upper_dir):

    dir_list = listdir('Extracted/' + upper_dir)

    desired_bands = ['band2', 'band3', 'band4']

    desired_types = ['toa', 'sr']

    # Filter all bands except for 2 and 3
    wanted_files = [str for str in dir_list if any(sub in str for sub in desired_bands)]

    # Filter all other images except for toa and sr types
    wanted_files = [str for str in wanted_files if any(sub in str for sub in desired_types)]

    wanted_files = wanted_files[0].split('.')[0]

    #Cut off integer at the end

    wanted_files = wanted_files[:-1]

    return wanted_files

if __name__ == "__main__":
    #image_prep();

    upper_dir = "LC080900862019072401T1-SC20200930162234"

    wanted_files = get_dir_frame(upper_dir)

    image_prep3(upper_dir)

    demo_normalize("Extracted/" + upper_dir + '/')
    #image_prep(upper_dir, wanted_files)
