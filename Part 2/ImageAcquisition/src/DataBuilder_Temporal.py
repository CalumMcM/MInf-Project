import numpy as np
import os
import sys
import time
from sample_tiles import *

_SWIR = False

def biome_numerator(biome_DIR):
    biome = biome_DIR.split('/')[-1]

    if biome == "TemporalAmazonia":
        return [1,0,0]
    if biome == "TemporalCerrado":
        return [0,1,0]
    if biome == "TemporalCaatinga":
        return [0,0,1]

    # if biome == "TemporalAmazonia":
    #     return np.zeros((4,51,51,1))
    # if biome == "TemporalCerrado": 
    #     return np.ones((4,51,51,1))
    # if biome == "TemporalCaatinga":
    #     return np.full((4,51,51,1), 2)

def condition(quad_DIR, biome_DIR):
    return True
    quad = quad_DIR.split('/')[-1]
    biome = biome_DIR.split('/')[-1]
    if (quad == "Quad_1" and biome == "TemporalAmazonia"):
        return False
    return True

# def get_labels():
#     y_data = np.zeros((1,3))
#     head_DIR = '/Volumes/GoogleDrive/My Drive/TemporalData'
#     save_DIR = '/Volumes/GoogleDrive/My Drive/TemporalData-Processed'

def get_data():

    X_data = np.zeros((1,4,51,51,3))
    y_data = np.zeros((1,3))

    head_DIR = '/Volumes/GoogleDrive/My Drive/TemporalData-Train'
    save_DIR = '/Volumes/GoogleDrive/My Drive/TemporalData-Processed-SNDVI'
    subfolders = [f.path for f in os.scandir(head_DIR) if f.is_dir()]
    idx = 0

    failed = 0
    years_dict = {"0": 0, "1": 0, "2": 0, "3": 0}

    for biome_DIR in subfolders:
        print ("CUR BIOME: {}".format(biome_DIR.split('/')[-1]))
        cur_biome_label = biome_numerator(biome_DIR)
        biome_idx = 0
        biome_quads = [f.path for f in os.scandir(biome_DIR) if f.is_dir()]

        for quad_DIR in biome_quads:
            print ("CUR QUAD: {}".format(quad_DIR.split('/')[-1]))
            quad_images = [f.path for f in os.scandir(quad_DIR) if f.is_dir()]

            img_processed = 1
            previous_progress = 0
            start = time.time()
                    
            for image_DIR in quad_images:

                if condition(quad_DIR, biome_DIR):

                    years = [f.path for f in os.scandir(image_DIR) if f.is_file() and '.tif' in f.path]

                    # Create array to store each set of images
                    years_array = np.zeros((4, 51, 51, 3))
                    cur_year = 0
                    for year_DIR in years:
                        # Open the iamge
                        raster = rs.open(year_DIR)

                        if _SWIR:
                            red = raster.read(4)
                            green = raster.read(3)
                            blue = raster.read(2)
                            
                            #red_norm = normalize_red(red)
                            #green_norm = normalize_green(green)
                            #blue_norm = normalize_blue(blue)
                            
                            # Stack bands
                            img = []
                            img = np.dstack((red, green, blue))

                            # Ignore images that are mishapen
                            x, y, _  = img.shape

                        else:# (SNDVI)
                            red = raster.read(4)
                            nir = raster.read(5)
                            swir1 = raster.read(6)
                            swir2 = raster.read(7)

                            swir = swir1+swir2

                            img = (swir+nir-red)/(swir+nir+red)

                            # Ignore images that are mishapen
                            x, y  = img.shape

                        if (x >= 48 and x <= 54) and (y >= 48 and y <= 54):
                            if _SWIR:
                                reset_img = reset_shape_2d(img)
                            else:
                                reset_img = reset_shape(img)

                            clean_img = remove_nan(reset_img)

                            if clean_img.shape == (51,51,3):
                                years_array[cur_year,:] = clean_img

                        else:
                            print (x)
                            print (y)
                            years_dict[str(cur_year)] += 1
                            failed += 1
                        
                        cur_year += 1

                
                    # Save each series of images as a numpy array
                    #np.save(os.path.join(save_DIR, '{}.npy'.format(idx)), years_array)
                    
                    # Create training and label matrices
                    #X_data = np.append(X_data, np.array([years_array]), axis = 0)
                    #y_data = np.append(y_data, np.array([cur_biome_label]), axis = 0)
                    
                    if (img_processed%5 == 0):

                        progress = (img_processed/len(quad_images))*100
                        end = time.time()
                        time_remaining = ((end - start)/(progress-previous_progress)) * (100-progress)

                        print ("Progress: {:.2f}% TIME REMAINING: {:.2f} seconds ".format(progress, time_remaining))
                        previous_progress= progress
                        start = time.time()
                        break

                # Method to stop early after idx number of images
                    
                # if idx == 30:
                #     # Remove first set of zeros 
                #     X_data = np.delete(X_data, (0), axis=0)
                #     y_data = np.delete(y_data, (0), axis=0)
                #     return X_data, y_data
                img_processed += 1
                biome_idx+= 1
                idx += 1

        print (biome_idx)
        print (failed)
        print (years_dict)
        y_data = np.append(y_data, np.array([cur_biome_label]*biome_idx), axis = 0)
    print (failed)
    np.save(os.path.join(save_DIR, 'labels.npy'), y_data)
    return idx
                
def main():
    get_data()

if __name__ == "__main__":
    main() 