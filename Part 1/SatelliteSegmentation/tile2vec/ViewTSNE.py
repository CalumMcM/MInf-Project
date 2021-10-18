import os
import re
import math
import time
import torch
import pickle
import numpy as np
import pandas as pd
import rasterio as rs
from PIL import Image
from affine import Affine
from matplotlib import cm
from pyproj import Proj, transform
from img2vec_pytorch import Img2Vec


# Takes a directory of spreadsheets for a given set of classes and
# computes the mean NDVI score for every 950 images, adds that score
# to a dictionary for each quad which is then appended to a dictionary
# storing a mapping of the biome and the quad to the mean ndvi scores
def get_NDVI(quads, DIR):

    classes = ['Amazon', 'Caatinga', 'Cerrado']
    classes = {'Amazon': 0, 'Caatinga': 1, 'Cerrado': 2}

    # Stores a mapping of the current image number to its corresponding NDVI score
    biome_to_NDVI = {}

    # Loop through the directory for each class and combine all CSV files
    # for each class into one dataframe
    for cur_class in classes.keys():

        cur_folder_dir = DIR+cur_class+' NDVI Quads'

        for quad in quads[cur_class]:

            # Stores a mapping of seed to NDVI score for current quad
            bin_NDVI_Mean = {}

            # Stores the current bin for NDVI, at a resolution of 950
            cur_NDVI_section = 0

            quad_folder_dir = cur_folder_dir + '/Quad ' + str(quad)

            # Sort the name of each spreadsheet by number
            files = os.listdir(quad_folder_dir)

            if '.DS_Store' in files: files.remove('.DS_Store')

            files.sort(key=lambda f: int(re.sub('\D', '', f)))

            # Append each spreadsheet NDVI value to array of all NDVI values
            for file in files:

                cur_df = pd.read_csv(quad_folder_dir+'/'+file, sep=',',header=0, index_col =0, engine='python')

                array = list(np.array(cur_df['NDVI']))

                bin_NDVI_Mean[cur_NDVI_section] = np.nanmean(array)

                # NDVI images were extracted in batches of 950
                cur_NDVI_section += 950

            biome_to_NDVI[str(cur_class) + "/Quad " + str(quad)] = bin_NDVI_Mean

    return biome_to_NDVI


def getImages(quads, DIR, step):
    """
    Given a directory of RGB images and a list of
    quadrants, will append the image name for every nth (step)
    image of each given quadrant to a dictionary storing
    the image names for each quadrant
    """

    biome_to_images = {}
    classes = []

    biome_class = [0, 0, 0, 2, 2, 2, 1, 1, 1]
    cur_quad = 0

    for cur_class in quads.keys():

        cur_class_DIR = os.path.join(DIR, cur_class)

        for quad in quads[cur_class]:

            img_names = []

            cur_quad_DIR = os.path.join(cur_class_DIR, "Quad " + str(quad))

            for filename in os.listdir(cur_quad_DIR):
                if filename.endswith('.tif'):
                    img_names.append(filename)

            biome_to_images[str(cur_class) + "/Quad " + str(quad)] = img_names[::step]

            classes += [biome_class[cur_quad]]*len(img_names[::step])

            cur_quad += 1



    return biome_to_images, classes

def img2NDVI(NDVI_Scores_DICT, image_names_DICT, DIR, img2vec, z_dim):
    """
    Given a dictionary of NDVI scores for each quadrant
    and a dictionary of image names for each quadrant,
    returns an array of image names and an array of ndvi scores
    where the nth element of the NDVI scores array is the NDVI
    score for the nth element of the image names array.
    Also returns the embedded vector for each image
    """
    NDVI_Scores_arr = []
    img_names_arr = []

    # Get total number of tiles:
    n_tiles = 0
    for quad in image_names_DICT:
        n_tiles += len(image_names_DICT[quad])

    embeddings = np.zeros((n_tiles, z_dim))

    idx = 0

    # loop through each quadrant
    for quad in image_names_DICT:

        # Get mean NDVI scores for this quadrant
        NDVI_Scores_quad = NDVI_Scores_DICT[quad]

        cur_DIR = os.path.join(DIR, quad)

        # Construct time for progress updates:
        progress = 1
        previous_percentage = 0
        start = time.time()

        print ("\nPROCESSING {}...\n".format(quad))

        # For each image get the bin it belongs to and
        # append the image name and NDVI value to respective
        # arrays
        for image_name in image_names_DICT[quad]:

            seed = image_name.split('.tif')[0]

            bin = math.floor(int(seed)/950)

            NDVI_Scores_arr.append(NDVI_Scores_quad[bin*950])

            img_names_arr.append(image_name)

            img_embedding = compute_embedding(image_name, cur_DIR, img2vec, z_dim)

            embeddings[idx, :] = img_embedding

            idx += 1

            if progress%100 == 0:

                percentage_progress = progress/len(image_names_DICT[quad]) * 100
                end = time.time()
                time_remaining = ((end - start)/(percentage_progress-previous_percentage)) * (100-percentage_progress)

                print ("PROGRESS: {:.0f}% TIME REMAINING: {:.2f} seconds ".format(percentage_progress, time_remaining))
                previous_percentage = percentage_progress
                start = time.time()

            progress += 1

    return img_names_arr, NDVI_Scores_arr, embeddings


def openImage(img_file, DIR):
    """
    Will open the given image at the given
    directory using Rasterio
    """

    # Open the iamge
    img = rs.open(os.path.join(DIR, img_file))

    # Extract RGB bands
    img = img.read([1,2,3])

    return img

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

    if np.isnan(mean):
        print ("IS NAN FOUND")
        print (new_tile)

    new_tile = np.nan_to_num(new_tile, nan=mean, posinf=mean, neginf=mean)

    return new_tile

def generateTile(img):
    """
    Returns a given tile after converting it to
    the correct format for the complex model
    """

    tile_img =  np.swapaxes(img, 0, 2)

    reset_shape_tile =  reset_shape(tile_img)

    processed_tile =  remove_nan(reset_shape(reset_shape_tile))

    return processed_tile

def compute_embedding(image_name, DIR, img2vec, z_dim):
    """
    Takes in the name of an image, converts it to a numpy tile
    and if the image is well formed (not NaN) then will embed it in
    the best model (currently pre-trained ResNet18) and return class predicted by
    Random Forest trained on embeddings for all biomes
    """

    # Convert image to tile
    img = openImage(image_name, DIR)

    # Skip images that are corrupt (i.e. they are only NaN)
    if not (np.isnan(np.nanmean(img))):

        # Convert image to tile
        tile = generateTile(img)

        # Convert tile to Pillow image
        im = Image.fromarray((tile * 255).astype(np.uint8))

        # Get ResNet18 embedded vector for image
        vec = img2vec.get_vec(im, tensor=True)

        vectorized_tile = [x for x in vec[0]]

        return vectorized_tile

    #return []

def main():
    """
    Given a directory (DIR) to '.tif' images,
    will loop through each image, vectorise it
    and compute the class it belongs to
    """

    DIR_RGB = r'/Users/calummcmeekin/Documents/GitHub/MInf-Project/SatelliteSegmentation/tile2vec/data/'

    DIR_NDVI = r'/Users/calummcmeekin/Documents/GitHub/MInf-Project/SatelliteSegmentation/baseline/Data/'

    biome_quads = {'Amazon': [1,3,4], 'Caatinga': [1,2,3], 'Cerrado': [1,2,4]}

    NDVI_Scores = get_NDVI(biome_quads, DIR_NDVI)

    step = 2

    image_names, classes = getImages(biome_quads, DIR_RGB, step)

    # Set up intial variables for ResNet18 Model
    cuda = torch.cuda.is_available()
    in_channels = 3
    z_dim = 512
    img2vec = Img2Vec(cuda=cuda)

    img_names_arr, NDVI_Scores_arr, embeddings = img2NDVI(NDVI_Scores, image_names, DIR_RGB, img2vec, z_dim)

    np.save(os.path.join('embeddings/', 'ndvi_scores.npy'), NDVI_Scores_arr)
    np.save(os.path.join('embeddings/', 'resnet18_sample.npy'), embeddings)
    np.save(os.path.join('embeddings/', 'classes.npy'), classes)

    #print (img_names_arr[:10], NDVI_Scores_arr[:10], embeddings[:10])


if __name__ == "__main__":
    main()
