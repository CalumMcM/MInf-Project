import glob
import numpy as np
import cv2
from osgeo import gdal
from PIL import Image
from os import listdir
from os import remove
import shutil

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
        return ((band - band_min)/(band_max - band_min))

    # Method to process the red band of the image (includes image stretch)
    def normalizeRed(self, intensity):

        iI      = intensity

        minI    = 86

        maxI    = 230

        minO    = 0

        maxO    = 255

        iO      = (iI-minI)*(((maxO-minO)/(maxI-minI))+minO)

        return iO

    # Method to process the green band of the image (includes image stretch)
    def normalizeGreen(self, intensity):

        iI      = intensity

        minI    = 90

        maxI    = 225

        minO    = 0

        maxO    = 255

        iO      = (iI-minI)*(((maxO-minO)/(maxI-minI))+minO)

        return iO

    # Method to process the blue band of the image (includes image stretch)
    def normalizeBlue(self, intensity):

        iI      = intensity

        minI    = 100

        maxI    = 210

        minO    = 0

        maxO    = 255

        iO      = (iI-minI)*(((maxO-minO)/(maxI-minI))+minO)

        return iO


    # Delete the downloaded images, freeing up memory
    def deleteSource(self):
        shutil.rmtree('Extracted/' + self.file_name)
        remove('raw_images/' + self.file_name + '.tar.gz')
        pass

    # Main method
    def prepareImage(self):

        bands = self.get_bands()

        for i in range(len(bands[0])):

            b2_link = gdal.Open(bands[0][i])
            b3_link = gdal.Open(bands[1][i])
            b4_link = gdal.Open(bands[2][i])

            # call the norm function on each band as array converted to float
            b2 = self.norm(b2_link.ReadAsArray().astype(np.float))
            b3 = self.norm(b3_link.ReadAsArray().astype(np.float))
            b4 = self.norm(b4_link.ReadAsArray().astype(np.float))

            # Create RGB
            rgb = np.dstack((b4,b3,b2))
            del b2, b3, b4

            # Visualize RGB
            #import matplotlib.pyplot as plt
            #plt.imshow(rgb)
            #plt.show()

            # Export RGB as TIFF file
            # Important: Here is where you can set the custom stretch
            # I use min as 2nd percentile and max as 98th percentile
            rgb = (rgb * 255).astype(np.uint8)
            im = Image.fromarray(rgb)
            im.save(self.dir_path + 'mergedNotContrasted.tif')

            # Create an image object

            imageObject     = Image.open(self.dir_path + "mergedNotContrasted.tif")

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
            normalizedImage = normalizedImage.crop((691,607,6908,7140))

            # Save the image
            normalizedImage.save('data/images/' + self.file_name + '.tif')

        self.deleteSource()
