import os
import time
import torch
import pickle
import numpy as np
import rasterio as rs
from PIL import Image
from time import time
from affine import Affine
from matplotlib import cm
from pyproj import Proj, transform
from img2vec_pytorch import Img2Vec

def getImages(DIR):

    img_names = []

    for filename in os.listdir(DIR):
        if filename.endswith('.tif'):
            img_names.append(filename)

    return img_names

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

def compute_class(image_name, DIR, img2vec, z_dim, rf):
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

        tile_class = rf.predict([vectorized_tile])

        return tile_class[0]

    else:
        return -1

def get_coord(img, DIR):
    """
    Given a tif image and a directory this function will return the lat and long
    coordinates for the bottom left of the image as an Earth Engine Feature object
    """

    path_vis = DIR + img

    # Read raster
    with rs.open(path_vis) as r:
        T0 = r.transform  # upper-left pixel corner affine transform
        p1 = Proj(r.crs)
        A = r.read()  # pixel values

    # All rows and columns
    cols, rows = np.meshgrid(np.arange(A.shape[2]), np.arange(A.shape[1]))

    # Get affine transform for pixel centres
    T1 = T0 * Affine.translation(0.5, 0.5)
    # Function to convert pixel row/column index (from 0) to easting/northing at centre
    rc2en = lambda r, c: (c, r) * T1

    # All eastings and northings (there is probably a faster way to do this)
    eastings, northings = np.vectorize(rc2en, otypes=[np.float, np.float])(rows, cols)

    # Project all longitudes, latitudes
    p2 = Proj(proj='latlong',datum='WGS84')
    longs, lats = transform(p1, p2, eastings, northings)

    return ("ee.Feature(ee.Geometry.Point({}, {})),".format(longs[0][0], lats[0][0]))


def main():
    """
    Given a direcotry to '.tif' images (DIR),
    will loop through each image, vectorise it
    and compute the class it belongs to
    """

    DIR = r'/Users/calummcmeekin/Documents/GitHub/MInf-Project/SatelliteSegmentation/tile2vec/data/Cross Section Quads/Quad 5/'

    image_names = getImages(DIR)

    # Set up intial variables for ResNet18 Model
    cuda = torch.cuda.is_available()
    in_channels = 3
    z_dim = 512
    img2vec = Img2Vec(cuda=cuda)

    # Load the Random Forest Classifier
    rf = pickle.load(open('models/full_rf.sav', 'rb'))

    # Construct Feature Collections for each biome for
    # Earth Engine
    ama_FC = "var ama_fc = ee.FeatureCollection(["
    cer_FC = "var cer_fc = ee.FeatureCollection(["
    cat_FC = "var cat_fc = ee.FeatureCollection(["

    # Construct time for progress updates:
    #start = time.process_time()

    for image_name in image_names:

        image_class = compute_class(image_name, DIR, img2vec, z_dim, rf)

        image_Feature = get_coord(image_name, DIR)

        if image_class == 0:
            ama_FC += "\n" + image_Feature

        elif image_class == 1:
            cer_FC += "\n" + image_Feature

        elif image_class == 2:
            cat_FC += "\n" + image_Feature


    ama_FC +=  "\n]);"
    cer_FC +=  "\n]);"
    cat_FC +=  "\n]);"

    print ("\n\nAmazonia Feature Collection:\n\n")
    print (ama_FC)

    print ("\n\nCerrado Feature Collection:\n\n")
    print (cer_FC)

    print ("\n\nCaatinga Feature Collection:\n\n")
    print (cat_FC)

if __name__ == "__main__":
    main()
