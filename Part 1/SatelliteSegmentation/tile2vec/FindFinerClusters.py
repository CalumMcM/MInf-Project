import os
import time
import torch
import pickle
import numpy as np
import pandas as pd
import rasterio as rs
from PIL import Image
from affine import Affine
from matplotlib import cm
from matplotlib import pyplot
from rasterio.plot import show
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from pyproj import Proj, transform
from img2vec_pytorch import Img2Vec
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances
from matplotlib_scalebar.scalebar import ScaleBar

"""
Allows exploratory analysis on a given directory of images to inspect
what images are closest to the clusters created by a SOTA algorithm
"""

def getImages(DIR, quads):

    """
    Given a directory of RGB images and a list of
    quadrants, will returns a dictionary of each
    quadrant mapped to a list of all image
    names in that quadrants
    """

    quad_images = {}

    for quad in quads:

        img_names = []

        cur_quad_DIR = os.path.join(DIR, "Quad " + str(quad))

        for filename in os.listdir(cur_quad_DIR):
            if filename.endswith('.tif'):
                img_names.append(filename)

        quad_images[quad] = img_names

    return quad_images

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

def get_image_embedding(image_name, DIR, img2vec, z_dim):
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

    else:
        return -1

def get_coord(img):
    """
    Given a tif diretory for a tif image this function will return the lat and long
    coordinates for the bottom left of the image as an Earth Engine Feature object
    """

    # Read raster
    with rs.open(img) as r:
        T0 = r.transform  # upper-left pixel corner affine transform
        p1 = Proj(r.crs)
        A = r.read()  # pixel values

    # Following code obtained from Stack Overflow user Mike T
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

def normalize(array):
    array = remove_nan(array)
    array_min, array_max = array.min(), array.max()

    return (array - array_min) / (array_max - array_min)

def view_images(closest):
    """
    Given a list of image names plots them in a
    grid where each row is the current class
    """
    #fig, axarr = plt.subplots(5,5)
    fig, axarr = pyplot.subplots(len(closest),len(closest[0]), figsize=(10,10), sharex=True, sharey=True)
    cur_cluster = 0

    # Generate scale for ScaleBar to add to each image

    for cluster in closest:
        #fig, axarr = pyplot.subplots(1,len(closest[0]), figsize=(10,10), sharex=True, sharey=True)
        for cur_img in range(0,len(cluster)):
            img = rs.open(cluster[cur_img])
            img_2 = img.read([1,2,3])

            img_2[0] = normalize(img_2[0])
            img_2[1] = normalize(img_2[1])
            img_2[2] = normalize(img_2[2])

            # Add scalebar

            new_res = img.width/0.075
            scale = new_res/30

            scalebar = ScaleBar(scale, location='lower right')
            axarr[cur_cluster][cur_img].add_artist(scalebar)

            axarr[cur_img][cur_img].set_xticks([])
            axarr[cur_img][cur_img].set_yticks([])

            axarr[cur_cluster][0].set_ylabel(cur_cluster)

            show(img_2, ax=axarr[cur_cluster][cur_img])

        cur_cluster += 1

    pyplot.subplots_adjust(
                    wspace=0,
                    hspace=0.07)
    pyplot.savefig('figures/CorruptImages.png', bbox_inches='tight')
    pyplot.show()

def printCluster(cluster, labels, X_dict):
    """
    Given a list of clusters (clusters) and
    a specific cluster, print out all images
    in that cluster
    """

    labels = np.array(labels)
    image_pos = np.where(labels == int(cluster))[0]
    image_names = [X_dict[cur_pos] for cur_pos in image_pos]
    print (image_names)


def main():
    """
    Given a directory (DIR) to '.tif' images,
    will loop through each image, vectorise it
    and compute the class it belongs to
    """

    DIR = r'/Users/calummcmeekin/Documents/GitHub/MInf-Project/SatelliteSegmentation/tile2vec/data/Amazon'

    quads = [1,3,4]

    quad_images = getImages(DIR, quads)

    # Set up intial variables for ResNet18 Model
    cuda = torch.cuda.is_available()
    in_channels = 3
    z_dim = 512
    img2vec = Img2Vec(cuda=cuda)

    # Construct Feature Collections for each biome for
    # Earth Engine
    points_UL = "var ul_fc = ee.FeatureCollection(["
    points_ML = "var ml_fc = ee.FeatureCollection(["
    points_MR = "var mr_fc = ee.FeatureCollection(["
    points_LL = "var ll_fc = ee.FeatureCollection(["
    points_LM = "var lm_fc = ee.FeatureCollection(["
    points_LR = "var lr_fc = ee.FeatureCollection(["


    # Construct time for progress updates:
    progress = 1
    previous_percentage = 0
    start = time.time()

    X_dict = {}
    X = []
    cur_tile = 0

    num_imgs = sum([len(quad_images[quad]) for quad in quad_images])

    print ("NUMBER OF IMAGES: {}".format(num_imgs))

    # Get image embeddings
    for quad in quad_images:

        CUR_DIR = os.path.join(DIR, "Quad " + str(quad))

        for image_name in quad_images[quad]:

            img_embedding = get_image_embedding(image_name, CUR_DIR, img2vec, z_dim)

            CUR_IMG_DIR = os.path.join(CUR_DIR, image_name)

            if (img_embedding != -1):
                X_dict[cur_tile] = CUR_IMG_DIR
                X.append(np.array(img_embedding))
                cur_tile += 1
            else:
                pass

            if progress%100 == 0:

                percentage_progress = progress/num_imgs * 100
                end = time.time()
                time_remaining = ((end - start)/(percentage_progress-previous_percentage)) * (100-percentage_progress)

                print ("PROGRESS: {:.2f} TIME REMAINING: {:.2f} seconds ".format(percentage_progress, time_remaining))
                previous_percentage = percentage_progress
                start = time.time()




            progress += 1

    # Get PCA for the data
    # Create a PCA instance: pca
    pca = PCA(n_components=20)
    principalComponents = pca.fit_transform(X)

    # Plot the explained variances
    features = range(pca.n_components_)
    plt.bar(features, pca.explained_variance_ratio_, color='black')
    plt.xlabel('PCA features')
    plt.ylabel('variance %')
    plt.xticks(features)
    plt.show()

    PCA_components = pd.DataFrame(principalComponents)

    Sum_of_squared_distances = []
    K = range(1,15)

    for k in K:
        print ("K: {}".format(k))
        km = KMeans(n_clusters=k)
        km = km.fit(PCA_components)
        Sum_of_squared_distances.append(km.inertia_)

    plt.plot(K, Sum_of_squared_distances, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Sum of Squared Distances')
    plt.show()

    kmeans = KMeans(n_clusters=5)
    kmeans.fit(PCA_components)
    y_kmeans = kmeans.predict(PCA_components)

    plt.scatter(PCA_components[0], PCA_components[1], c=y_kmeans, s=50, cmap='viridis')

    labels = ['0','1','2','3','4','5']
    colours = ['black','red', 'blue', 'green', 'orange', 'pink']
    centers = kmeans.cluster_centers_
    for i in range(0,len(centers)):
        plt.scatter(centers[i,0], centers[i,1], c=colours[i], marker = 'x', label=labels[i], s=200, alpha=0.5);

    plt.xlabel('PCA 1')
    plt.ylabel('PCA 2')
    plt.legend()
    plt.savefig('figures/kMeansClusters.png')
    plt.show()

    # Calculate the points closest to the center of K-means centroids
    # Calculate distance between centers and every points
    distances = pairwise_distances(centers, PCA_components, metric='euclidean')

    # Return the position of the 5 points closest to the centers
    ind = [np.argpartition(i, 5)[:5] for i in distances]

    cur_cluster = 0
    closest = []

    for cluster in ind:
        closest.append([X_dict[indexes] for indexes in cluster])
    #closest = find_k_closest(centers[0], PCA_components[0])

    print ("Cluster 0: {}\n".format(closest[0]))
    print ("Cluster 1: {}\n".format(closest[1]))
    print ("Cluster 2: {}\n".format(closest[2]))
    print ("Cluster 3: {}\n".format(closest[3]))

    for x in closest[0]:

        image_Feature = get_coord(x)

        points_UL += "\n" + image_Feature

    for x in closest[1]:

        image_Feature = get_coord(x)

        points_LM += "\n" + image_Feature

    for x in closest[2]:

        image_Feature = get_coord(x)

        points_LL += "\n" + image_Feature


    for x in closest[3]:

        image_Feature = get_coord(x)

        points_LR += "\n" + image_Feature

    for x in closest[4]:

        image_Feature = get_coord(x)

        points_MR += "\n" + image_Feature

    """
    for x in closest[5]:

        image_Feature = get_coord(x)

        points_ML += "\n" + image_Feature
    """


    points_UL += "\n]);"
    #points_ML += "\n]);"
    points_MR += "\n]);"
    points_LL += "\n]);"
    points_LM += "\n]);"
    points_LR += "\n]);"

    print ("\n\n\n")
    print (points_UL)

    #print ("\n\n\n")
    #print (points_ML)

    print ("\n\n\n")
    print (points_MR)

    print ("\n\n\n")
    print (points_LL)

    print ("\n\n\n")
    print (points_LM)

    print ("\n\n\n")
    print (points_LR)

    # Construct plot of all images closest to the center:
    view_images(closest)

    # Print all images belonging to desired cluster
    print ("\n\n\nCLUSTER: 0")
    printCluster(0, kmeans.labels_, X_dict)

    print ("\n\n\nCLUSTER: 1")
    printCluster(1, kmeans.labels_, X_dict)

    print ("\n\n\nCLUSTER: 2")
    printCluster(2, kmeans.labels_, X_dict)

    print ("\n\n\nCLUSTER: 3")
    printCluster(3, kmeans.labels_, X_dict)

    print ("\n\n\nCLUSTER: 4")
    printCluster(4, kmeans.labels_, X_dict)

    #print ("\n\n\nCLUSTER: 5")
    #printCluster(5, kmeans.labels_, X_dict)

if __name__ == "__main__":
    main()
