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
    fp = r'/Users/calummcmeekin/Downloads/1.tif'
    img = rs.open(fp)

    # Print number of bands in the image:
    show(img.read([1,2,3]))

def view_batch_image(image_dir):
    files = [f for f in listdir(image_dir) if isfile(join(image_dir, f)) and '.DS_Store' not in f]

    iter = 1

    for file in files:
        fp = r'/Users/calummcmeekin/Documents/GitHub/MInf-Project/SatelliteSegmentation/ImagePrep/Labelled/Amazonia/tif/' + file
        img = rs.open(fp)

        # Print number of bands in the image:
        show(img.read([1,2,3]))

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
    plt.savefig('medianNDVIHistogram.png')
    plt.show()

def gmm(df):

    X = np.concatenate((np.array(df.Amazonia), np.array(df.Caatinga), np.array(df.Cerrado)))

    SecDim = np.zeros(len(X))
    X = np.array(list(zip(X,SecDim)))

    y = [0]*len(df.Amazonia) + [1]*len(df.Caatinga) + [2]*len(df.Cerrado)

    gmm = sklearn.mixture.GaussianMixture(n_components=3).fit(X)
    labels = gmm.predict(X)

    unique, counts = np.unique(labels, return_counts=True)
    print (dict(zip(unique, counts)))

    plt.scatter(X[:, 0], X[:, 1], c=labels, s=40, cmap='viridis');
    plt.savefig('gmmNDVI')


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
    plt.savefig('GaussianNaiveBayes')
    plt.show()

if __name__ == "__main__":

    MapBiome_data = '/Users/calummcmeekin/Downloads/mosaic-2019-0000000000-0000000000.tif'
    Landsat_image = '/Users/calummcmeekin/Downloads/LE07_220076_20000608.tif'

    #convertPNG('AmazoniaNN')

    #view_batch_image(r'/Users/calummcmeekin/Documents/GitHub/MInf-Project/SatelliteSegmentation/ImagePrep/Labelled/AmazoniaNN/tif')

    #plot_group("CerradoNN")
    #plotNDVI()
    #view_image(Landsat_image)

    dir_amazon = os_dir + 'AmazoniaNDVI/AmazoniaMedianNDVI1000.csv'
    dir_caat = os_dir + 'CerradoNDVI/CerradoMedianNDVI1000.csv'
    dir_cerr = os_dir + 'CaatingaNDVI/CaatingaMedianNDVI1000.csv'

    df_ama = pd.read_csv(dir_amazon, sep=',',header=0, index_col =0)
    df_cat= pd.read_csv(dir_caat, sep=',',header=0, index_col =0)
    df_cer = pd.read_csv(dir_cerr, sep=',',header=0, index_col =0)

    df_ama['Caatinga'] = df_cat['NDVI']
    df_ama['Cerrado'] = df_cer['NDVI']

    del df_ama['.geo']
    df_ama.columns = ['Amazonia', 'Caatinga', 'Cerrado']
    df = df_ama.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(axis=0)

    GuassianNaiveBayes(df)

    plot_histogram(df)

    #plotScatterNDVI(df)
