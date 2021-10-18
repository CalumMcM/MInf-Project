
import os
from PIL import Image
from osgeo import gdal
from subprocess import DEVNULL, STDOUT, run

"""
Given a directory of existing images (OLD_DIR) will convert each image
to png format in a new directory (DIR) and delete any other files that are
generated
"""

OLD_DIR = '/Users/calummcmeekin/Documents/GitHub/MInf-Project/SatelliteSegmentation/tile2vec/data/Amazon/'
DIR = '/Users/calummcmeekin/Downloads/swav-master/DIR=TestData/0'
# Classes for DIR: Ama = 0, Cer = 1, Cat = 2
cur_filename = 0

cat_train_quads = [1,2,4]
cat_test_quads = [3]

ama_train_quads = [1,3,4]
ama_test_quads = [2]

cer_train_quads = [1,2,3]
cer_test_quads = [4]

for quad in ama_test_quads:

    cur_DIR = os.path.join(OLD_DIR, "Quad" + str(quad))

    print (cur_DIR)

    for filename in os.listdir(cur_DIR):

        if filename.endswith('.tif'):

            input_file = os.path.join(cur_DIR, filename)
            output_file = os.path.join(DIR, str(cur_filename)+ '.png')
            os.system('gdal_translate -of PNG -ot Byte -scale 0 2550 0 255 '+input_file + ' ' + output_file + " >/dev/null 2>&1")

            cur_filename += 1


for filename in os.listdir(DIR):

    if filename.endswith('.png.aux.xml'):
        os.remove(os.path.join(DIR, filename))
