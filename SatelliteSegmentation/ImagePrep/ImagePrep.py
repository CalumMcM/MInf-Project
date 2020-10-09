import glob
import numpy as np
import cv2
from osgeo import gdal
from PIL import Image
from os import listdir
from os import remove
import shutil
from skimage import exposure
from PIL import ImageEnhance
from matplotlib import pyplot as plt

class ImagePrep:

    def __init__(self, file_name):
        self.file_name = file_name
        self.dir_path = 'Extracted/' + file_name + '/'

    # Just open bands 2, 3 and 4
    def get_bands(self):
        dir_list = listdir('Extracted/' + self.file_name)

        desired_bands = ['band2', 'band3', 'band4']

        desired_types = ['toa', 'sr']

        # Filter all bands except for 2 and 3
        wanted_files = [str for str in dir_list if any(sub in str for sub in desired_bands)]

        # Filter all other images except for toa and sr types
        wanted_files = [str for str in wanted_files if any(sub in str for sub in desired_types)]

        bands = []

        sorted_files = self.sort_files(wanted_files)

        for file in sorted_files:

            # Open file
            band_file = glob.glob(self.dir_path + file)
            bands.append(band_file)

        return bands

    # Return a list of file band names in order: 2 -> 3 -> 4
    def sort_files(self, files):
        sorted = []
        i = 2
        while (len(sorted) < 3):
            for j in range(len(files)):
                if (('band' + str(i)) in str(files[j])):
                    sorted.append(files[j])
                    i+=1
        return sorted

    # Normalise the entire image
    def norm(self, band):
        band_min, band_max = band.min(), band.max()
        #mean = np.mean(band)
        #stand_dev = np.std(band)
        return ((band - band_min)/(band_max - band_min))
        #return (band-mean)/stand_dev

    # Method to process the red band of the image (includes image stretch)
    def normalizeRed(self, intensity):

        iI      = intensity

        minI    = 86

        maxI    = 230

        minO    = 0

        maxO    = 255

        iO      = (iI-minI)*(((maxO-minO)/(maxI-minI))+minO)
        if (iO < 20):
            return 0
        return iO

    # Method to process the green band of the image (includes image stretch)
    def normalizeGreen(self, intensity):

        iI      = intensity

        minI    = 90

        maxI    = 225

        minO    = 0

        maxO    = 255

        iO      = (iI-minI)*(((maxO-minO)/(maxI-minI))+minO)
        if (iO < 20):
            return 0
        return iO

    # Method to process the blue band of the image (includes image stretch)
    def normalizeBlue(self, intensity):

        iI      = intensity

        minI    = 100

        maxI    = 210

        minO    = 0

        maxO    = 255

        iO      = (iI-minI)*(((maxO-minO)/(maxI-minI))+minO)

        if (iO < 40):
            return 0

        return iO


    # Delete the downloaded images, freeing up memory
    def deleteSource(self):
        #shutil.rmtree('Extracted/' + self.file_name)
        #remove('raw_images/' + self.file_name + '.tar.gz')
        pass

    def linear_stretching(self, input, lower_stretch_from, upper_stretch_from):
        """
        Linear stretching of input pixels
        :param input: integer, the input value of pixel that needs to be stretched
        :param lower_stretch_from: lower value of stretch from range - input
        :param upper_stretch_from: upper value of stretch from range - input
        :return: integer, integer, the final stretched value
        """

        lower_stretch_to = 0  # lower value of the range to stretch to - output
        upper_stretch_to = 255  # upper value of the range to stretch to - output

        output = (input - lower_stretch_from) * ((upper_stretch_to - lower_stretch_to) / (upper_stretch_from - lower_stretch_from)) + lower_stretch_to

        return output
    # Main method
    def prepareImage(self):

        bands = self.get_bands()

        for i in range(len(bands[0])):


            b2_link = gdal.Open(bands[0][i])
            b3_link = gdal.Open(bands[1][i])
            b4_link = gdal.Open(bands[2][i])


            print ("calculating norm")
            # call the norm function on each band as array converted to float
            b2 = self.norm(b2_link.ReadAsArray().astype(np.float))
            b3 = self.norm(b3_link.ReadAsArray().astype(np.float))
            b4 = self.norm(b4_link.ReadAsArray().astype(np.float))


            # Create RGB
            rgb = np.dstack((b4,b3,b2))
            #rgb = np.dstack((b2_link.ReadAsArray().astype(np.float), b3_link.ReadAsArray().astype(np.float), b4_link.ReadAsArray().astype(np.float)))
            #rgb = np.concatenate((b2, b3, b4))
            del b2, b3, b4

            # Visualize RGB
            #import matplotlib.pyplot as plt
            #plt.imshow(rgb)
            #plt.show()

            # Export RGB as TIFF file
            # Important: Here is where you can set the custom stretch
            # I use min as 2nd percentile and max as 98th percentile
            rgb = (rgb * 255).astype(np.uint8)

            #-----Converting image to LAB Color model-----------------------------------
            lab = cv2.cvtColor(rgb, cv2.COLOR_BGR2LAB)

            #-----Splitting the LAB image to different channels-------------------------
            l, a, b = cv2.split(lab)

            #-----Applying CLAHE to L-channel-------------------------------------------
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
            cl = clahe.apply(l)

            #-----Merge the CLAHE enhanced L-channel with the a and b channel-----------
            limg = cv2.merge((cl,a,b))

            #-----Converting image from LAB Color model to RGB model--------------------
            final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
            cv2.imwrite(self.dir_path + 'final_before.png', final)

            """
            im = np.asarray(rgb)


             # assign variable to max and min value of image pixels
            max_value = np.max(im)
            min_value = np.min(im)
            print (max_value)
            print (min_value)

            print (im.shape())
            im = (im - min_value) * ((255-0)/(max_value-min_value))
            print (im.shape())
            #cv2.imwrite(self.dir_path + 'mergedNotContrasted.tif', im)
            """
            im = Image.fromarray(rgb)

            im.save(self.dir_path + 'mergedNotContrasted.tif')

            #-----Reading the image-----------------------------------------------------
            img = cv2.imread(self.dir_path + 'mergedNotContrasted.tif', 1)

            #-----Converting image to LAB Color model-----------------------------------
            lab= cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

            #-----Splitting the LAB image to different channels-------------------------
            l, a, b = cv2.split(lab)

            #-----Applying CLAHE to L-channel-------------------------------------------
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
            cl = clahe.apply(l)

            #-----Merge the CLAHE enhanced L-channel with the a and b channel-----------
            limg = cv2.merge((cl,a,b))

            #-----Converting image from LAB Color model to RGB model--------------------
            final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
            cv2.imwrite(self.dir_path + 'final_after.png', final)

            # Create an image object
            imageObject     = Image.open(self.dir_path + "mergedNotContrasted.tif")

            # Get histogram

            img = cv2.imread(self.dir_path + "final_after.png")
            color = ('b','g','r')
            for i,col in enumerate(color):
                histr = cv2.calcHist([img],[i],None,[256],[0,256])
                plt.plot(histr,color = col)
                plt.xlim([0,256])
            plt.show()

            """
            print (imageObject)
            hist, bins = np.histogram(imageObject, bins=50)

            width = 0.7 * (bins[1] - bins[0])
            center = (bins[:-1] + bins[1:]) / 2
            plt.bar(center, hist, align='center', width=width)
            plt.title("Gaussian Histogram")
            plt.xlabel("Value")
            plt.ylabel("Frequency")
            plt.show()
            """
            #enhancer_object = ImageEnhance.Contrast(imageObject)
            #normalizedImage = enhancer_object.enhance(3)

            # Split the red, green and blue bands from the Image
            multiBands      = imageObject.split()

            # Apply point operations that does contrast stretching on each color band
            normalizedRedBand      = multiBands[0].point(self.normalizeRed)

            normalizedGreenBand    = multiBands[1].point(self.normalizeGreen)

            normalizedBlueBand     = multiBands[2].point(self.normalizeBlue)


            # Create a new image from the contrast stretched red, green and blue brands
            normalizedImage = Image.merge("RGB", (normalizedRedBand, normalizedGreenBand, normalizedBlueBand))

            # Rotate the image
            normalizedImage = normalizedImage.rotate(12.4)

            # Crop the image
            normalizedImage = normalizedImage.crop((691,630,6890,7100))


            # Save the image
            normalizedImage.save('data/images/' + self.file_name + '.tif')

            #img = cv2.imread('data/images/' + self.file_name + '.tif' ,0)
            #plt.hist(b2,256,[0,256]); plt.show()

        self.deleteSource()


if __name__ == "__main__":
    #imagePrep = ImagePrep("LC080900862013101101T1-SC20200930162215")
    #imagePrep.prepareImage();

    #imagePrep = ImagePrep("LC080900862019072401T1-SC20200930162234")
    #imagePrep.prepareImage();

    imagePrep = ImagePrep("LC080900862020050701T1-SC20200930161922")
    imagePrep.prepareImage();
