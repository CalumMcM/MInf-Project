from PIL import Image
import numpy as np
import rasterio as rs
import math
from rasterio.plot import show
import xarray as xr
import georaster
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.image as mpimg
from osgeo import gdal
from os import listdir
from os.path import isfile, join
import pandas as pd
import sklearn
import sklearn.mixture
from scipy.spatial.distance import cdist
import scipy.stats as stats
from scipy.stats import norm
import subprocess
import os
from affine import Affine
from pyproj import Proj, transform
import time

plt.style.use('ggplot')

# Knn
from matplotlib.colors import ListedColormap
from sklearn import neighbors, datasets

# Guassian Naive Bayes
from sklearn.naive_bayes import GaussianNB

#####################################################
########## SCRIPT TO DISPLAY A .GEOTIF IMG ##########
##########   OR SAVE IMAGE MATRIX AS .TXT  ##########
#####################################################
"""
save_dir = r'/Users/calummcmeekin/Documents/GitHub/MInf-Project/SatelliteSegmentation/ImagePrep/Amazonia/png'
scale = '-scale min_val max_val'

options_list = [
    '-ot Byte',
    '-of PNG',
    scale
]
options_string = " ".join(options_list)

gdal.Translate(save_dir,
               image_dir,
               options=options_string)
"""

os_dir = '/Users/calummcmeekin/Documents/GitHub/MInf-Project/SatelliteSegmentation/ImagePrep/Labelled/'

def plot_group(biome):
    path = os_dir + biome + '/tif'
    path = r'/Users/calummcmeekin/Documents/GitHub/MInf-Project/SatelliteSegmentation/tile2vec/data/Amazonia'

    all_files = [f for f in listdir(path) if isfile(join(path, f)) ]
    png_files = [f.split('.')[0] for f in all_files if not 'png.aux.xml' in f and not '.DS_Store' in f]
    png_files.sort(key=int)
    print (png_files)
    w=10
    h=10
    fig=plt.figure(figsize=(8, 8))
    plt.title('NDVI Scores For Cerrado', pad=20)
    plt.axis('off')
    plt.tight_layout()
    columns = 5
    rows = 6
    for i in range(1, columns*rows +1):
        img = mpimg.imread(path + '/' + png_files[i-1] + '.tif')
        fig.add_subplot(rows, columns, i)
        plt.axis('off')
        plt.title(str(png_files[i-1])) # Adds tile name
        plt.imshow(img)
        fig.tight_layout()
    plt.savefig(os_dir + biome + '/' + biome + '_classes.png')

    plt.show()

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
    fp = r'/Users/calummcmeekin/Downloads/0-7.tif'
    path_2 = r'/Users/calummcmeekin/Documents/GitHub/MInf-Project/SatelliteSegmentation/tile2vec/data/Amazonia/39094.tif'
    img = rs.open(fp)

    # Print number of bands in the image:
    #show(img.read([1,2,3]))
    return img.read([1,2,3])

def get_cord(img):

    path_vis = r'/Users/calummcmeekin/Documents/GitHub/MInf-Project/SatelliteSegmentation/tile2vec/data/Cross Section Quads/Quad 5/'

    path_vis += str(img) + ".tif"

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


def produce_EE_FC_points(arr):
    """
    Given an array of tif file names, the corresponding
    lat and long coordinates for each image will be returned as a
    string that can be copied and pasted into Earth Engine to produce
    a FeatureCollection of Geometry Points
    """
    output = "var pointsFC = ee.FeatureCollection(["
    progress = 0
    start = time.process_time()
    for img in arr:
        img_Feature = get_cord(img)
        output += "\n" + img_Feature

        if progress%100 == 0:
            end = time.process_time()
            percentage_progress = progress/len(arr) * 100
            time_remaining = (end - start) * (100-percentage_progress)
            print ("PROGRESS: {:.2f} TIME REMAINING: {:.2f} seconds".format(percentage_progress,time_remaining ))
            start = time.process_time()

        progress += 1
    output += "\n]);"

    print (output)

def view_image_2(image_dir):

    path_2 = r'/Users/calummcmeekin/Documents/GitHub/MInf-Project/SatelliteSegmentation/tile2vec/data/Amazonia/'
    path_ama = r'/Users/calummcmeekin/Documents/GitHub/MInf-Project/SatelliteSegmentation/tile2vec/data/Amazonia Quads/Quad 4/0.tif'
    path_cer = r'/Users/calummcmeekin/Documents/GitHub/MInf-Project/SatelliteSegmentation/tile2vec/data/Cerrado/'
    path_cat = r'/Users/calummcmeekin/Documents/GitHub/MInf-Project/SatelliteSegmentation/tile2vec/data/Caatinga/'
    path_download = r'/Users/calummcmeekin/Downloads/0.tif'
    path_ama = r'/Users/calummcmeekin/Documents/GitHub/MInf-Project/SatelliteSegmentation/tile2vec/data/Visual Quads/Quad 1/8.tif'
    # SHow single image:
    if os.path.isfile(path_ama):

        #img1 = rs.open(path_download)
        img2 = rs.open(path_ama)

        #img1 = img1.read([1,2,3])
        img2 = img2.read([1,2,3])

        #img_np1 =  np.swapaxes(img1, 0, 2)#np.array(anchor_img)
        img_np2 =  np.swapaxes(img2, 0, 2)#np.array(anchor_img)

        #sum1 = np.sum(img_np1)
        sum2 = np.sum(img_np2)

        #show(img1)
        #print (img2)
        show(img2)

    #ds = gdal.Open(path_2)
    #myarray = np.array(ds.GetRasterBand(1).ReadAsArray())
    #i = 39094
    """
    for i in range (0, 10):

        if os.path.isfile(path_2+str(i)+".tif"):

            img = rs.open(path_2+str(i)+".tif")

            img = img.read([1,2,3])

            img_np =  np.swapaxes(img, 0, 2)#np.array(anchor_img)

            sum = np.sum(img_np)

            if not np.isnan(sum):
                print (i)

            show(img)

    """
    """
    for i in range (2038, 2039):

        if os.path.isfile(path_ama+str(i)+".tif"):

            img = rs.open(path_ama+str(i)+".tif")

            print (path_ama+str(i)+".tif")
            img = img.read([1,2,3])

            img_np =  np.swapaxes(img, 0, 2)#np.array(anchor_img)

            sum = np.sum(img_np)

            print (img)

            print (i)

            show(img)
    """
    """
    for i in range (10, 20):

        if os.path.isfile(path_cer+str(i)+".tif"):
            print ("IMAGE: " + str(i))
            img = rs.open(path_cer+str(i)+".tif")

            img_bands = img.read([1,2,3])
            band1 = img.read([1])
            band2 = img.read([1])
            band3 = img.read([1])
            band1 =  np.swapaxes(band1, 0, 2)#np.array(anchor_img)
            band2 =  np.swapaxes(band2, 0, 2)#np.array(anchor_img)
            band3 =  np.swapaxes(band3, 0, 2)#np.array(anchor_img)

            print (np.sum(band1[1]))
            print (np.sum(band2))
            print (np.sum(band3[1]))

            img_np =  np.swapaxes(img_bands, 0, 2)#np.array(anchor_img)

            sum = np.sum(img_np)

            if not np.isnan(sum):
                print (i)

            show(img_bands)

    for i in range (0, 10):
        print ("CAATINGA")
        print (str(i)+".tif")
        if os.path.isfile(path_cat+str(i)+".tif"):

            img = rs.open(path_cat+str(i)+".tif")

            img = img.read([1,2,3])

            img_np =  np.swapaxes(img, 0, 2)#np.array(anchor_img)

            sum = np.sum(img_np)

            if not np.isnan(sum):
                print (i)

            show(img)




            """
    # Print number of bands in the image:
    #show(img.read([1,2,3]))
    #show(img.read([1]))
    #show(img.read([2]))
    #show(img.read([3]))
    #show(img_2.read([1,2,3]))
    """
    options_list = [
    '-ot Byte',
    '-of PNG',
    '-b 3',
    '-scale'
    ]

    options_string = " ".join(options_list)

    gdal.Translate(
        str(i)+'.png',
        path_2+str(i)+".tif",
        options=options_string
    )
    """
    #gdal_translate -ot Byte -scale -of GTiff /Users/calummcmeekin/Documents/GitHub/MInf-Project/SatelliteSegmentation/tile2vec/data/Amazonia/0.tif output.tif
    #gdal_translate -ot Byte -scale -of PNG /Users/calummcmeekin/Documents/GitHub/MInf-Project/SatelliteSegmentation/tile2vec/data/Amazonia/0.tif output.png

def view_batch_image(image_dir):
    files = [f for f in listdir(image_dir) if isfile(join(image_dir, f)) and '.DS_Store' not in f]

    iter = 1

    for file in files:
        fp = r'/Users/calummcmeekin/Documents/GitHub/MInf-Project/SatelliteSegmentation/ImagePrep/Labelled/Amazonia/tif/' + file
        img = rs.open(fp)

        # Print number of bands in the image:
        img = img.read([1,2,3])
        return img

def convertPNG(biome):

    image_dir = os_dir + biome + '/tif'
    files = [f for f in listdir(image_dir) if isfile(join(image_dir, f)) and '.DS_Store' not in f]

    iter = 1

    for file in files:
        iter += 1

        print (file)

        save_dir = os_dir + biome + '/png/' + file.split('.')[0] + '.png'
        file_dir = os_dir + biome + '/tif/' + file

        scale = '-scale min_val max_val'

        print (file_dir)
        options_list = [
            '-ot Byte',
            '-of PNG',
            scale
        ]
        options_string = " ".join(options_list)

        gdal.Translate(save_dir, file_dir, options=options_string)

def plotScatterNDVI(df):
    print (len(amazoniaNDVI))

    xCAT = np.arange(1,31)
    xCER = np.arange(1,29)
    xAMA = np.arange(1,30)

    plt.scatter(xCAT, caatingaNDVI, color= 'green', marker='.', label='Caatinga')
    m, b = np.polyfit(xCAT, caatingaNDVI, 1)
    plt.plot(xCAT, m*xCAT + b, color= 'green', label='Caatinga LBF')

    plt.scatter(xCER, cerradoNDVI, color= 'blue', marker='*', label='Cerrado')
    m, b = np.polyfit(xCER, cerradoNDVI, 1)
    plt.plot(xCER, m*xCER + b, color= 'blue', label='Cerrado LBF')

    plt.scatter(xAMA, amazoniaNDVI, color= 'red', marker='v', label='Amazonia')
    m, b = np.polyfit(xAMA, amazoniaNDVI, 1)
    plt.plot(xAMA, m*xAMA + b, color= 'red', label='Amazonia LBF')

    plt.legend()
    plt.grid(color='grey', linestyle='-', linewidth=0.1)
    plt.ylim(-1,1)
    plt.xlim(0,35)
    plt.title('NDVI\'s For Each Biome')
    plt.xlabel('Tile Number')
    plt.ylabel('NDVI Score')

    plt.savefig(os_dir + 'ndviPlot.png')
    plt.show()

def plot_histogram(df):

    cmap = ListedColormap(['#FF0000', '#00FF00','#00AAFF'])

    ax = df.plot.hist(bins=100, alpha=0.9, cmap=cmap)

    plt.title("Median NDVI Frequency per Biome")
    plt.xlim(-1,1)
    plt.xlabel('Median NDVI')
    plt.savefig('medianNDVIHistogramWinter.png')
    plt.show()

def gmmElbow(df):

    X = np.concatenate((np.array(df.Amazonia), np.array(df.Caatinga), np.array(df.Cerrado))).reshape(-1, 1)

    X_ama = np.array(df.Amazonia).reshape(-1, 1)
    X_cat = np.array(df.Caatinga).reshape(-1, 1)
    X_cer = np.array(df.Cerrado).reshape(-1, 1)

    datasets = [X_ama, X_cat, X_cer]
    gmm_names = ['Amazonia', 'Cerrado', 'Caatinga']

    clusters = np.arange(1,20)

    for X in datasets:

        likelihoods = []

        for k in clusters:

            gmm = sklearn.mixture.GaussianMixture(n_components=k).fit(X)

            likelihoods.append(gmm.score(X))

            #plt.savefig('gmmNDVI')

        plt.plot(clusters, likelihoods)
        plt.xlabel('Number of Clusters')
        plt.xticks(np.arange(0, 20, step=1))
        plt.ylabel('Log-Likelihood')
        plt.title('Log-Likelihood of Gaussian Mixture Model\nFor Different Number of Clusters')
        plt.show()

def gmm(df, season):

    X_ama = np.array(df.Amazonia).reshape(-1, 1)
    X_cat = np.array(df.Caatinga).reshape(-1, 1)
    X_cer = np.array(df.Cerrado).reshape(-1, 1)
    # Construct training data
    X = np.concatenate((np.array(df.Amazonia), np.array(df.Caatinga), np.array(df.Cerrado)))

    colours = ['#FF0000', '#00FF00','#00AAFF']
    linestyles = ['dashed', 'dotted','dashdot']
    labels = ['Amazonia', 'Cerrado', 'Caatinga']

    #    Plot histogram
    cmap = ListedColormap(colours)

    X =  X.reshape(-1, 1)

    # Train Gaussian Mixture Model
    gmm = sklearn.mixture.GaussianMixture(n_components=3).fit(X)
    gmm_ama = sklearn.mixture.GaussianMixture(n_components=2).fit(X_ama)
    gmm_cer = sklearn.mixture.GaussianMixture(n_components=2).fit(X_cer)
    gmm_cat = sklearn.mixture.GaussianMixture(n_components=2).fit(X_cat)

    gmms = [gmm_ama, gmm_cer, gmm_cat]

    lines = []

    for i in range(len(gmms)):
        # Plot histogram
        cmap = ListedColormap(colours[i])
        ax = df[labels[i]].plot.hist(bins=100, alpha=0.8, cmap=cmap, density=True)

        gmm = gmms[i]

        means = gmm.means_
        covariances = gmm.covariances_

        for j in range(len(gmm.means_)):
            mean = means[j][0]
            sigma = np.sqrt(covariances[j][0][0])

            x = np.linspace(start = -0.7, stop = 0.6, num = 1000)
            """
            if mean < 0:
                colour = colours[2]
            elif mean < 0.2:
                colour = colours[1]
            else:
                colour = colours[0]
            """

            #line = plt.plot(x, y, label = '$\mu$ = {}, $\sigma^2$ = {}'.format(m, v), linestyle = linestyles[i], linewidth=1, color= 'black')
            #plt.legend()
            #x = np.linspace(mean - 1*sigma, mean + 1*sigma, 100)
            #line, = plt.plot(x, norm.pdf(x, mean, sigma), linestyle = linestyles[i], label=labels[i]+' GMM', linewidth=1, color= 'black')

            line, = plt.plot(x, norm(mean, sigma).pdf(x), linestyle = '-', label=labels[i]+' GMM', linewidth=1, color= colours[i])
            lines.append(line)
            print (labels[i])



        plt.xlim(-0.67, 0.6)

    plt.title('Normalised Histogram And \nGaussian Mixture Model For Each Biome In '+season)
    plt.xlabel('NDVI Score')
    # Legend
    ama_leg = mpatches.Patch(color='#FF0000', label='Amazonia')
    cat_leg = mpatches.Patch(color='#00FF00', label='Caatinga')
    cer_leg = mpatches.Patch(color='#00AAFF', label='Cerrado')

    plt.legend(handles=[cer_leg, cat_leg, ama_leg, lines[0] , lines[2], lines[4]], loc='upper left' )
    plt.savefig('NDVIHistogramGMM'+season+'ALL_2.png')
    plt.show()


def kmeans(df):

    X = X.dropna(axis=0)
    X = np.array(X)
    SecDim = np.zeros(len(X))
    X = np.array(list(zip(X,SecDim)))


    kmeans = sklearn.cluster.KMeans(n_clusters=3)
    kmeans.fit(X)
    y_kmeans = kmeans.predict(X)

    plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=50, cmap='viridis')

    centers = kmeans.cluster_centers_
    plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5);

def GuassianNaiveBayes(df):

    X = np.concatenate((np.array(df.Amazonia), np.array(df.Caatinga), np.array(df.Cerrado)))

    y = [0]*len(df.Amazonia) + [1]*len(df.Caatinga) + [2]*len(df.Cerrado)

    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.20)

    features_train = list(zip(X_train, [0]*len(X_train)))

    features_test = list(zip(X_test, [0]*len(X_test)))

    #Create a Gaussian Classifier
    model = GaussianNB()

    # Train the model using the training sets
    model.fit(features_train, y_train)

    # Predict the model using the test set
    ypred = model.predict(features_test)

    # Compute accuracy of the model on the test set
    print("Accuracy: " + str(model.score(features_test, y_test)))

    cmap = ListedColormap(['#FF0000', '#00FF00','#00AAFF'])
    plt.scatter(X_test, y_test, c=ypred, s=20, cmap=cmap, alpha=0.1)
    plt.title('Gaussian Naive Bayes')
    plt.xlabel('NDVI')
    plt.yticks(np.arange(0, 3, step=1), ['Amazonia', 'Caatinga', 'Cerrado'])
    plt.ylabel('Class')

    # Legend
    ama_leg = mpatches.Patch(color='#FF0000', label='Amazonia')
    cat_leg = mpatches.Patch(color='#00FF00', label='Caatinga')
    cer_leg = mpatches.Patch(color='#00AAFF', label='Cerrado')

    plt.legend(handles=[cer_leg, cat_leg, ama_leg], loc='center left' )
    plt.savefig('GaussianNaiveBayesWinter')
    plt.show()

if __name__ == "__main__":

    MapBiome_data = '/Users/calummcmeekin/Downloads/mosaic-2019-0000000000-0000000000.tif'
    Landsat_image = '/Users/calummcmeekin/Downloads/1.tif'

    #convertPNG('AmazoniaNN')

    #view_batch_image(r'/Users/calummcmeekin/Documents/GitHub/MInf-Project/SatelliteSegmentation/ImagePrep/Labelled/AmazoniaNN/tif')

    #plot_group("Amazonia")
    ama_imgs = [11, 42, 147, 159, 247, 294, 443, 463, 472, 597, 602]

    cer_imgs = [1, 23, 38, 41, 46, 50, 52, 62, 77, 89, 101, 139, 151, 170, 180, 184, 219, 239, 243, 257, 259, 261, 262, 270, 284, 288, 313, 317, 327, 334, 339, 353, 367, 388, 398, 400, 403, 415, 416, 429, 447, 456, 459, 466, 487, 490, 508, 512, 513, 534, 537, 553, 561, 566, 568, 570, 581, 598, 614, 616, 618, 620, 632]

    cat_imgs = [0, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 39, 40, 43, 44, 45, 47, 48, 49, 53, 54, 55, 56, 57, 58, 59, 60, 61, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 102, 103, 104, 105, 106, 107, 108, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 140, 141, 142, 143, 144, 145, 146, 148, 149, 150, 152, 153, 154, 155, 156, 157, 158, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 171, 172, 173, 174, 175, 176, 177, 178, 179, 181, 182, 183, 186, 187, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 240, 241, 242, 244, 245, 246, 248, 249, 251, 252, 253, 254, 255, 256, 258, 260, 263, 264, 265, 266, 267, 268, 269, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 285, 286, 287, 289, 290, 291, 292, 293, 295, 296, 297, 298, 299, 300, 301, 302, 303, 304, 305, 306, 307, 309, 310, 311, 312, 314, 315, 316, 318, 319, 320, 321, 322, 323, 324, 325, 326, 328, 329, 330, 331, 332, 333, 335, 336, 337, 338, 340, 341, 342, 343, 344, 345, 346, 347, 348, 349, 350, 351, 352, 354, 355, 356, 357, 358, 359, 360, 361, 362, 363, 364, 365, 366, 368, 369, 370, 371, 372, 373, 374, 375, 376, 377, 378, 379, 380, 381, 382, 383, 384, 385, 386, 387, 389, 390, 391, 392, 393, 394, 395, 396, 397, 399, 402, 404, 405, 406, 407, 408, 409, 410, 411, 412, 413, 414, 418, 419, 420, 421, 422, 423, 424, 425, 426, 427, 428, 430, 431, 432, 433, 434, 435, 436, 437, 438, 439, 440, 441, 442, 444, 445, 446, 448, 449, 450, 451, 452, 453, 454, 455, 457, 458, 460, 461, 462, 464, 465, 467, 468, 469, 470, 471, 473, 474, 475, 476, 477, 478, 479, 480, 481, 482, 483, 485, 486, 488, 489, 491, 492, 493, 494, 495, 496, 497, 498, 499, 500, 501, 502, 503, 504, 505, 506, 507, 509, 510, 511, 514, 515, 516, 517, 518, 519, 521, 522, 523, 524, 525, 526, 527, 528, 529, 530, 531, 532, 533, 535, 536, 538, 539, 540, 541, 542, 543, 544, 545, 546, 547, 548, 549, 550, 551, 552, 554, 555, 556, 557, 558, 559, 560, 562, 563, 564, 565, 567, 569, 571, 572, 573, 574, 575, 576, 577, 578, 579, 580, 582, 583, 584, 585, 586, 587, 588, 589, 590, 591, 592, 593, 594, 595, 596, 599, 600, 601, 603, 604, 605, 606, 607, 608, 609, 610, 611, 612, 613, 615, 617, 619, 621, 622, 623, 624, 625, 626, 627, 628, 629, 630, 631, 633, 634, 635, 636, 637, 638, 639, 640, 641, 642, 643, 644, 645, 646, 647, 648]

    produce_EE_FC_points(cer_imgs)

    #view_image_2(Landsat_image)
    #img = view_image(Landsat_image)

    #print (img.shape)

    #pin = np.swapaxes(img, 0, 2)

    #print (pin.shape)
    #show(img)

    season = "Summer"

    """ ORIGINAL NDVI SPREADSHEETS
    dir_amazon = os_dir + 'AmazoniaNDVI/AmazoniaMedianNDVI1000'+season+'.csv'
    dir_caat = os_dir + 'CerradoNDVI/CerradoMedianNDVI1000'+season+'.csv'
    dir_cerr = os_dir + 'CaatingaNDVI/CaatingaMedianNDVI1000'+season+'.csv'

    df_ama = pd.read_csv(dir_amazon, sep=',',header=0, index_col =0)
    df_cat= pd.read_csv(dir_caat, sep=',',header=0, index_col =0)
    df_cer = pd.read_csv(dir_cerr, sep=',',header=0, index_col =0)
    """

    """ NEW NDVI SPREADSHEETS """
    dir_amazon1 = os_dir + 'AmazoniaNDVI/AmazoniaNDVI950'+season+'.csv'
    dir_amazon2 = os_dir + 'AmazoniaNDVI/AmazoniaNDVI1000'+season+'.csv'

    dir_cerr1 = os_dir + 'CerradoNDVI/CerradoNDVI950'+season+'.csv'
    dir_cerr2 = os_dir + 'CerradoNDVI/CerradoNDVI1000'+season+'.csv'

    dir_caat1 = os_dir + 'CaatingaNDVI/CaatingaNDVI950'+season+'.csv'
    dir_caat2 = os_dir + 'CaatingaNDVI/CaatingaNDVI1000'+season+'.csv'

    df_ama = pd.read_csv(dir_amazon1, sep=',',header=0, index_col =0)
    df_ama2 = pd.read_csv(dir_amazon2, sep=',',header=0, index_col =0)
    df_cat1 = pd.read_csv(dir_caat1, sep=',',header=0, index_col =0)
    df_cat2 = pd.read_csv(dir_caat2, sep=',',header=0, index_col =0)
    df_cer1 = pd.read_csv(dir_cerr1, sep=',',header=0, index_col =0)
    df_cer2 = pd.read_csv(dir_cerr2, sep=',',header=0, index_col =0)


    df_ama = df_ama.append(df_ama2)
    df_cat1 = df_cat1.append(df_cat2)
    df_cer1 = df_cer1.append(df_cer2)



    df_ama['Caatinga'] = df_cat1['NDVI']

    df_ama['Cerrado'] =  df_cer1['NDVI']


    del df_ama['.geo']

    df_ama.columns = ['Amazonia', 'Caatinga', 'Cerrado']
    df = df_ama.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(axis=0)

    #GuassianNaiveBayes(df)

    #gmmElbow(df)

    #gmm(df, season)

#    plot_histogram(df)

    #plotScatterNDVI(df)
