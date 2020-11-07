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
    path = os_dir + biome + '/png'

    all_files = [f for f in listdir(path) if isfile(join(path, f)) ]
    png_files = [f for f in all_files if not 'png.aux.xml' in f and not '.DS_Store' in f]
    print (len(png_files))
    w=10
    h=10
    fig=plt.figure(figsize=(8, 8))
    columns = 5
    rows = 6
    for i in range(1, columns*rows +1):
        img = mpimg.imread(path + '/' + png_files[i-1])
        fig.add_subplot(rows, columns, i)
        plt.axis('off')
        plt.imshow(img)
    plt.savefig(os_dir + biome + '/all_classes.png')
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
        fp = r'/Users/calummcmeekin/Documents/GitHub/MInf-Project/SatelliteSegmentation/ImagePrep/Labelled/Cerrado/tif/' + file
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



if __name__ == "__main__":

    MapBiome_data = '/Users/calummcmeekin/Downloads/mosaic-2019-0000000000-0000000000.tif'
    Landsat_image = '/Users/calummcmeekin/Downloads/LE07_220076_20000608.tif'

    convertPNG('Amazonia')

    #view_batch_image(r'/Users/calummcmeekin/Documents/GitHub/MInf-Project/SatelliteSegmentation/ImagePrep/Labelled/Caatinga/tif')

    plot_group("Amazonia")
    #view_image(Landsat_image)
