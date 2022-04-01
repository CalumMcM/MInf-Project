import numpy as np
import os
import sys
import time
import rasterio as rs

"""
Desc: This program goes through a preset directory, combines each year of images for one location into one numpy array
that is stored in a directory with the same name + '-Processed'
"""

_SNDVI = True
TRAIN = False


def biome_numerator(biome_DIR):
    biome = biome_DIR.split('/')[-1]

    if "Amazonia" in biome:
        return [1,0,0]
    if "Cerrado" in biome:
        return [0,1,0]
    if "Caatinga" in biome:
        return [0,0,1]

def condition(quad_DIR, biome_DIR):
    return True
    quad = quad_DIR.split('/')[-1]
    biome = biome_DIR.split('/')[-1]
    if ((quad == "Quad_1" or quad == "Quad_2" or quad == "Quad_3") and biome == "TemporalAmazonia"):
        return False
    return True

def get_spatial_data():

    X_data = np.zeros((1,51,51,3))
    y_data = np.zeros((1,3))

    head_DIR = '/Volumes/GoogleDrive/My Drive/'
    save_DIR = '/Volumes/GoogleDrive/My Drive/ResNet18-2019-Training'

    biomes = ['Amazonia', 'Cerrado', 'Caatinga']
    
    cur_img = 0
        
    for biome in biomes:

        if TRAIN:
            if biome == "Amazonia":
                desired_quads = [2,3,4]

            if biome == "Cerrado":
                desired_quads = [1,2,3]

            if biome == "Caatinga":
                desired_quads = [1,2,4]

        else:
            if biome == "Amazonia":
                desired_quads = [1]
                
            if biome == "Cerrado":
                desired_quads = [4]

            if biome == "Caatinga":
                desired_quads = [3]

        cur_biome_label = biome_numerator(biome)
        biome_idx = 0

        for quad in desired_quads:

            quad_idx = 0

            biome_DIR = head_DIR + biome + ' 2019 Quad ' + str(quad) + '/'

            images = [f.path for f in os.scandir(biome_DIR) if f.is_file() and '.tif' in f.path]

            print (biome + "\tQuad: " + str(quad) + "\tImages: " + str(len(images)))

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
                        quad_idx += 1
                        biome_idx += 1
        
                if (quad_idx%50 == 0):

                        try:
                            progress = (quad_idx/len(images))*100
                            end = time.time()
                            time_remaining = ((end - start)/(progress-previous_progress)) * (100-progress)

                            print ("Progress: {:.2f}% TIME REMAINING: {:.2f} seconds ".format(progress, time_remaining))
                            previous_progress= progress
                            start = time.time()
                        except:
                            print (quad_idx)

        y_data = np.append(y_data, np.array([cur_biome_label]*biome_idx), axis = 0)
    
        X_data = np.delete(X_data, (0), axis=0)
        y_data = np.delete(y_data, (0), axis=0)

    np.save(os.path.join(save_DIR, 'test_data.npy'), X_data)
    np.save(os.path.join(save_DIR, 'test_labels.npy'), y_data)


def get_temporal_data():
    #X_data = np.zeros((1,4,51,51,3))
    y_data = np.zeros((1,3))
    print ("TEMPORAL")

    head_DIR = '/Volumes/GoogleDrive-103278653964135897318/My Drive/TemporalData-Train'
    save_DIR = '/Volumes/GoogleDrive-103278653964135897318/My Drive/TemporalData-Processed-Train-SNDVI'
    subfolders = [f.path for f in os.scandir(head_DIR) if f.is_dir()]
    idx = 0
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

                    if _SNDVI:
                        # Create array to store each set of images
                        years_array = np.zeros((4, 51, 51))
                    else:
                        # Create array to store each set of images
                        years_array = np.zeros((4, 51, 51, 3))
                        
                    cur_year = 0
                    for year_DIR in years[0:4]:
                        # Open the iamge
                        raster = rs.open(year_DIR)

                        if not _SNDVI:

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

                        else: # (SNDVI)

                            red = raster.read(4)
                            nir = raster.read(5)
                            swir1 = raster.read(6)
                            swir2 = raster.read(7)

                            swir = swir1+swir2
                            
                            try: 
                                nom = (swir+nir-red)
                                nom[nom == 0] = 1
                                dom = (swir+nir+red)
                                dom[dom == 0] = 1
                                img = remove_nan(nom)/remove_nan(dom)

                            except:
                                print ("failure on division")

                            # Ignore images that are mishapen
                            x, y  = img.shape

                        if (x > 48 and x < 54) and (y > 48 and y < 54):
                            if _SNDVI:
                                reset_img = reset_shape_2d(img)
                            else:
                                reset_img = reset_shape(img)

                            clean_img = remove_nan(reset_img)

                            if clean_img.shape == (51,51,3) or clean_img.shape == (51,51):
                                years_array[cur_year,:] = clean_img
                        
                        cur_year += 1
                    
                    # Save each series of images as a numpy array
                    np.save(os.path.join(save_DIR, '{}.npy'.format(idx)), years_array)

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

                # Method to stop early after idx number of images
                    
                # if idx == 30:
                #     # Remove first set of zeros 
                #     X_data = np.delete(X_data, (0), axis=0)
                #     y_data = np.delete(y_data, (0), axis=0)
                #     return X_data, y_data
                img_processed += 1
                biome_idx+= 1
                idx += 1

        y_data = np.append(y_data, np.array([cur_biome_label]*biome_idx), axis = 0)

    # Remove first empty row of 0's
    y_data = np.delete(y_data, (0), axis=0)
    np.save(os.path.join(save_DIR, 'labels.npy'), y_data)
    return idx

def reset_shape_2d(tile):
    """
    Takes a tile and removes/pads it
    so that the returned tile has shape
    (51, 51)
    """
    x, y = tile.shape
        
    # Reduce shape
    if (x == 51 and y == 51):

        return tile
    
    elif (x>51):
        tile = np.delete(tile, -1, 0)
        return reset_shape_2d(tile)
    
    elif (y>51):
        tile = np.delete(tile, -1, 1)
        return reset_shape_2d(tile)

    # Pad shape
    elif (x<51):

        basic = np.array([tile[0]])

        tile = np.vstack((tile, basic))
        
        return reset_shape_2d(tile)
    
    elif (y < 51):
                
        mean = np.mean(tile)
        new_col = np.full((51,1), mean)

        tile = np.append(tile, new_col, axis=1)
        
        #tile = np.append(tile, basic, axis=1)
                
        return reset_shape_2d(tile)
        
def reset_shape(tile):
    """
    Takes a tile and removes/pads it
    so that the returned tile has shape
    (51, 51, 3)
    """
    x, y, z = tile.shape
        
    # Reduce shape
    if (x == 51 and y == 51):

        return tile
    
    elif (x>51):
        tile = np.delete(tile, -1, 0)
        return reset_shape(tile)
    
    elif (y>51):
        tile = np.delete(tile, -1, 1)
        return reset_shape(tile)

    # Pad shape
    elif (x<51):

        basic = np.array([tile[0]])

        tile = np.vstack((tile, basic))
        
        return reset_shape(tile)
    
    elif (y < 51):
                
        mean = np.mean(tile)
        new_col = np.full((51,1,3), mean)

        tile = np.append(tile, new_col, axis=1)
        
        #tile = np.append(tile, basic, axis=1)
                
        return reset_shape(tile)

def remove_nan(new_tile):
    """
    Takes a tile and replaces any nan values with the mean
    of the image the nan appears in
    """
    
    mean = np.nanmean(new_tile)
        
    new_tile = np.nan_to_num(new_tile, nan=mean, posinf=mean, neginf=mean)
    
    return new_tile

def main():
    get_temporal_data()

if __name__ == "__main__":
    main() 