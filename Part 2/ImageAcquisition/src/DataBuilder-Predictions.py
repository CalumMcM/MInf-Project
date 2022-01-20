import numpy as np
import os
import sys
import time
from sample_tiles import *

SWIR = False

def condition(quad_DIR, biome_DIR):
    return True
    quad = quad_DIR.split('/')[-1]
    biome = biome_DIR.split('/')[-1]
    if ((quad == "Quad_1" or quad == "Quad_2" or quad == "Quad_3") and biome == "TemporalAmazonia"):
        return False
    return True

def get_spatial_data():

    X_data = np.zeros((1,51,51,3))

    head_DIR = '/Volumes/GoogleDrive/My Drive/AmazonToCerrado1-2016/'
    
    cur_img = 0

    images = [f.path for f in os.scandir(head_DIR) if f.is_file() and '.tif' in f.path]

    previous_progress = 0
    start = time.time()

    for image in images:
        # Open the image
        raster = rs.open(image)

        if  SWIR:
            img = raster.read(6)

        else:
            red = raster.read(4)
            green = raster.read(3)
            blue = raster.read(2)
        
            # Stack bands
            img = []
            img = np.dstack((red, green, blue))

        # Ignore images that are mishapen
        if SWIR:
            x, y = img.shape
        else:
            x, y, _  = img.shape

        if (x > 48 and x < 54) and (y > 48 and y < 54):
            if SWIR:
                reset_img = reset_shape_2d(img)
            else:
                reset_img = reset_shape(img)

            clean_img = remove_nan(reset_img)

            if clean_img.shape == (51,51,3) or clean_img.shape == (51,51):
                X_data = np.append(X_data, np.array([clean_img]), axis = 0)
                #X_data[cur_img,:] = clean_img

                cur_img += 1

        if (cur_img%50 == 0):

                try:
                    progress = (cur_img/len(images))*100
                    end = time.time()
                    time_remaining = ((end - start)/(progress-previous_progress)) * (100-progress)

                    print ("Progress: {:.2f}% TIME REMAINING: {:.2f} seconds ".format(progress, time_remaining))
                    previous_progress= progress
                    start = time.time()
                except:
                    print (cur_img)
    
    X_data = np.delete(X_data, (0), axis=0)

    np.save(os.path.join(head_DIR, 'pred_data.npy'), X_data)
                
def main():
    get_spatial_data()

if __name__ == "__main__":
    main() 