import os
import time
import numpy as np
import rasterio as rs
from PIL import Image
from affine import Affine
from matplotlib import cm
from sample_tiles import *
from pyproj import Proj, transform
import tensorflow as tf
from keras.models import Model
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten, GlobalAveragePooling2D,  BatchNormalization, Layer, Add, Dropout

"""
Given a directory of images this file will classify each image
using the best ResNet model and return the location of each of 
images.
"""

class ResnetBlock(Model):
    """
    A standard resnet block.
    """

    def __init__(self, channels: int, down_sample=False):
        """
        channels: same as number of convolution kernels
        """
        super().__init__()

        self.__channels = channels
        self.__down_sample = down_sample
        self.__strides = [2, 1] if down_sample else [1, 1]

        KERNEL_SIZE = (3, 3)
        # use He initialization, instead of Xavier (a.k.a 'glorot_uniform' in Keras), as suggested in [2]
        INIT_SCHEME = "he_normal"

        self.conv_1 = Conv2D(self.__channels, strides=self.__strides[0],
                             kernel_size=KERNEL_SIZE, padding="same", kernel_initializer=INIT_SCHEME)
        self.bn_1 = BatchNormalization()
        self.conv_2 = Conv2D(self.__channels, strides=self.__strides[1],
                             kernel_size=KERNEL_SIZE, padding="same", kernel_initializer=INIT_SCHEME)
        self.bn_2 = BatchNormalization()
        self.merge = Add()

        if self.__down_sample:
            # perform down sampling using stride of 2, according to [1].
            self.res_conv = Conv2D(
                self.__channels, strides=2, kernel_size=(1, 1), kernel_initializer=INIT_SCHEME, padding="same")
            self.res_bn = BatchNormalization()

    def call(self, inputs):
        res = inputs

        x = self.conv_1(inputs)
        x = self.bn_1(x)
        x = tf.nn.relu(x)
        x = self.conv_2(x)
        x = self.bn_2(x)

        if self.__down_sample:
            res = self.res_conv(res)
            res = self.res_bn(res)

        # if not perform down sample, then add a shortcut directly
        x = self.merge([x, res])
        out = tf.nn.relu(x)
        return out


class ResNet18(Model):

    def __init__(self, num_classes, **kwargs):
        """
            num_classes: number of classes in specific classification task.
        """
        super().__init__(**kwargs)
        self.conv_1 = Conv2D(32, (7, 7), strides=2,
                             padding="same", kernel_initializer="he_normal")
        self.init_bn = BatchNormalization()
        self.pool_2 = MaxPool2D(pool_size=(2, 2), strides=2, padding="same")
        self.res_1_1 = ResnetBlock(32)
        self.res_1_2 = ResnetBlock(32)
        self.dropout1_3 = Dropout(0.3)
        self.res_2_1 = ResnetBlock(64, down_sample=True)
        self.res_2_2 = ResnetBlock(64)
        self.dropout2_3 = Dropout(0.3)
        #self.res_3_1 = ResnetBlock(32, down_sample=True)
        #self.res_3_2 = ResnetBlock(32)
        # self.res_4_1 = ResnetBlock(512, down_sample=True)
        # self.res_4_2 = ResnetBlock(512)
        self.avg_pool = GlobalAveragePooling2D()
        self.flat = Flatten()
        self.fc = Dense(num_classes, activation="softmax")

    def call(self, inputs):
        out = self.conv_1(inputs)
        out = self.init_bn(out)
        out = tf.nn.relu(out)
        out = self.pool_2(out)
        i = 0
        #for res_block in [self.res_1_1, self.res_1_2, self.res_2_1, self.res_2_2, self.res_3_1, self.res_3_2, self.res_4_1, self.res_4_2]:
        for res_block in [self.res_1_1, self.res_1_2, self.res_2_1, self.res_2_2]:
            out = res_block(out)
            i += 1
            if i %2 == 0:
                out = self.dropout1_3(out)
        out = self.avg_pool(out)
        out = self.flat(out)
        out = self.fc(out)
        return out

def get_model(model_DIR):
    """
    Takes the path to a pretrained ResNet model
    that has the same structure as ResNet18 class 
    above. 
    Returns the model with the weights of the 
    pre-trained model loaded into it.
    """
    model = ResNet18(3)

    model.build(input_shape = (None, 51, 51, 3))

    model.compile(optimizer = 'adam', loss = 'catergorical_crossentropy', metrics=["accuracy"])

    model.load_weights(model_DIR)
    
    return model


def get_coord(DIR):
    """
    Given a tif image and a directory this function will return the lat and long
    coordinates for the bottom left of the image
    """
    # Read raster
    with rs.open(DIR) as r:
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

    return longs, lats

def make_feature(longs, lats):
    point = False
    """
    Converts the given longs and lats for an image into a
    google earth engine feature
    """
    if point:
            return ("ee.Feature(ee.Geometry.Point({}, {})),".format(longs[0][0], lats[0][0]))
    else:
        top_long = longs[0][0]+0.015
        top_lat = lats[0][0]+0.015
        return ("ee.Feature(ee.Geometry.Rectangle({}, {}, {}, {})),".format(longs[0][0], lats[0][0], top_long, top_lat))

def get_class(y_pred):
    """
    Returns the class with the highest
    confidence. If no class has a 
    confidence greater than the 
    threshold then the inconclusive class
    is returned.
    """
    threshold = 0.8
    if np.amax(y_pred) > threshold:
        return np.argmax(y_pred)
    else:
        return -2

def build_dataset(head_DIR):

    X_data = np.zeros((1,51,51,3))
    
    cur_img = 0

    images = [f.path for f in os.scandir(head_DIR) if f.is_file() and '.tif' in f.path]
    clean_images = []

    previous_progress = 0
    start = time.time()

    print ("BUILDING DATASET...")
    for image in images:
        # Open the image
        raster = rs.open(image)

        red = raster.read(4)
        green = raster.read(3)
        blue = raster.read(2)
    
        # Stack bands
        img = []
        img = np.dstack((red, green, blue))

        # Ignore images that are mishapen
        x, y, _  = img.shape

        if (x > 48 and x < 54) and (y > 48 and y < 54):

            reset_img = reset_shape(img)

            clean_img = remove_nan(reset_img)

            if clean_img.shape == (51,51,3) or clean_img.shape == (51,51):
                X_data = np.append(X_data, np.array([clean_img]), axis = 0)
                clean_images.append(image)

                cur_img += 1

                longs, lats = get_coord(image)

                if longs[0][0] == -53.29796532743698 and lats[0][0] == -6.539741488787744:
                    print (image)

        if (cur_img%50 == 0):

                try:
                    progress = (cur_img/len(images))*100
                    end = time.time()
                    time_remaining = ((end - start)/(progress-previous_progress)) * (100-progress)

                    print ("Progress: {:.2f}% TIME REMAINING: {:.2f} seconds ".format(progress, time_remaining))
                    previous_progress= progress
                    start = time.time()
                except:
                    print (cur_img)
    
    X_data = np.delete(X_data, (0), axis=0)

    np.save(os.path.join(head_DIR, 'pred_data.npy'), X_data)
    np.save(os.path.join(head_DIR, 'cleaned_images'), clean_images)

    print ("BUILDING DATASET... COMPLETE")
    return X_data, clean_images
                
def main():
    """
    Given a directory (DIR) till load the
    built dataset and predict the biome 
    of each of the images
    """

    # Test quads for each biome
    # Cat quad 3, Cer Quad 4, Ama Quad 2
    
    DIR = r'/Volumes/GoogleDrive/My Drive/AreaOfDeforestation-2021'

    X_data, images = build_dataset(DIR)
    X_data = np.load(os.path.join(DIR, 'pred_data.npy'))
    images = np.load(os.path.join(DIR, 'cleaned_images.npy'))

    # Load the Random Forest Classifier
    model = get_model("/Volumes/GoogleDrive/My Drive/ResNet/94_Model.h5")

    # Get Predictions
    y_pred = model.predict(X_data)

    # Construct Feature Collections for each biome for
    # Earth Engine
    ama_FC = "var ama_fc_2021 = ee.FeatureCollection(["
    cer_FC = "var cer_fc_2021 = ee.FeatureCollection(["
    cat_FC = "var cat_fc_2021 = ee.FeatureCollection(["
    inconclusive_FC = "var inconclusive_fc_2021 = ee.FeatureCollection(["

    # Construct time for progress updates:
    previous_percentage = 0
    start = time.time()

    ama_count, cer_count, cat_count, inc_count = 0, 0, 0, 0

    if not len(images) == len(y_pred):
        print ("ERROR IN PROCESSING")
        print (len(images))
        print (len(y_pred))
        print (images[0:10])
        print (y_pred[0:10])
        quit()


    for img_idx, image_name in enumerate(images):

        image_class = get_class(y_pred[img_idx])

        image_longs, image_lats = get_coord(image_name)

        image_Feature = make_feature(image_longs, image_lats)

        if image_class == 0:
            ama_FC += "\n" + image_Feature
            ama_count += 1

        elif image_class == 1:
            cer_FC += "\n" + image_Feature
            cer_count += 1

        elif image_class == 2:
            cat_FC += "\n" + image_Feature
            cat_count += 1

        elif image_class == -2:
            inconclusive_FC += "\n" + image_Feature
            inc_count += 1

        if (img_idx+1)%100 == 0:

            percentage_progress = img_idx/len(images) * 100
            end = time.time()
            time_remaining = ((end - start)/(percentage_progress-previous_percentage)) * (100-percentage_progress)

            print ("PROGRESS: {:.2f} TIME REMAINING: {:.2f} seconds ".format(percentage_progress, time_remaining))
            previous_percentage = percentage_progress
            start = time.time()

    ama_FC +=  "\n]);"
    cer_FC +=  "\n]);"
    cat_FC +=  "\n]);"
    inconclusive_FC +=  "\n]);"

    f = open("EarthEngine_Classifications2021.txt", "w")
    f.write(ama_FC)
    f.close()

    f = open("EarthEngine_Classifications2021.txt", "a")

    f.write("\n\n\n")
    f.write (cer_FC)

    f.write ("\n\n\n")
    f.write (cat_FC)

    f.write("\n\n\n")
    f.write(inconclusive_FC)
    f.close()

    print (len(images))
    ama_per = (ama_count/len(images)) * 100
    cer_per = (cer_count/len(images)) * 100
    cat_per = (cat_count/len(images)) * 100
    inc_per = (inc_count/len(images)) * 100

    print ("\nAmazon: {:.2f}%\nCerrado: {:.2f}%\nCaatinga: {:.2f}%\nInconclusive: {:.2f}\%".format(ama_per, cer_per, cat_per, inc_per))
if __name__ == "__main__":
    main()
