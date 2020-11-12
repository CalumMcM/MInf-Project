from PIL import Image
import numpy as np
import rasterio as rs
import math
from rasterio.plot import show
import xarray as xr
import georaster
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from osgeo import gdal
from os import listdir
from os.path import isfile, join

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

def plotNDVI():
    caatingaNDVI = [0.282,0.270,0.334,0.159,0.019,0.048,-0.018,0.114,-0.103,0.232,0.364,0.165,0.082,0.003,-0.021,-0.035,0.172,-0.119,0.040,0.040,0.040,-0.060,-0.128,0.025,0.048,-0.003,-0.260,0.354,0.061,-0.183]
    cerradoNDVI = [0.181,0.270,0.086,0.117,0.111,0.062,0.028,0.191,0.141,0.227,0.092,0.291,0.124,0.267,0.149,0.334,0.123,0.251,0.250,0.261,0.297,0.192,0.264,0.225,0.163,0.157,0.157,0.044]
    amazoniaNDVI = [-0.393,0.356,0.526,0.554,0.077,0.362,0.484,0.544,0.544,0.367,0.374,0.360,0.358,0.511,0.477,0.386,0.353,0.098,0.319,0.513,0.466,0.287,0.388,0.308,0.377,0.362,0.386,0.351,0.124]
    print (len(amazoniaNDVI))
    xCAT = np.arange(1,31)
    xCER = np.arange(1,29)
    xAMA = np.arange(1,30)

    plt.scatter(xCAT, caatingaNDVI, color= 'red', marker='.', label='Caatinga')
    m, b = np.polyfit(xCAT, caatingaNDVI, 1)
    plt.plot(xCAT, m*xCAT + b, color= 'red', label='Caatinga LBF')

    plt.scatter(xCER, cerradoNDVI, color= 'blue', marker='*', label='Cerrado')
    m, b = np.polyfit(xCER, cerradoNDVI, 1)
    plt.plot(xCER, m*xCER + b, color= 'blue', label='Cerrado LBF')

    plt.scatter(xAMA, amazoniaNDVI, color= 'green', marker='v', label='Amazonia')
    m, b = np.polyfit(xAMA, amazoniaNDVI, 1)
    plt.plot(xAMA, m*xAMA + b, color= 'green', label='Amazonia LBF')

    plt.legend()
    plt.grid(color='grey', linestyle='-', linewidth=0.1)
    plt.ylim(-1,1)
    plt.xlim(0,35)
    plt.title('NDVI\'s For Each Biome')
    plt.xlabel('Tile Number')
    plt.ylabel('NDVI Score')

    plt.savefig(os_dir + 'ndviPlot.png')
    plt.show()



if __name__ == "__main__":

    MapBiome_data = '/Users/calummcmeekin/Downloads/mosaic-2019-0000000000-0000000000.tif'
    Landsat_image = '/Users/calummcmeekin/Downloads/LE07_220076_20000608.tif'

    #convertPNG('AmazoniaNN')

    #view_batch_image(r'/Users/calummcmeekin/Documents/GitHub/MInf-Project/SatelliteSegmentation/ImagePrep/Labelled/AmazoniaNN/tif')

    #plot_group("CerradoNN")
    plotNDVI()
    #view_image(Landsat_image)
