import numpy as np
from skimage import exposure
import cv2
from os import listdir

class ChannelMerge:

    def __init__(self, file_name):
        self.file_name = file_name


    def get_file_names(self):
        dir_list = listdir('Extracted/' + self.file_name)

        desired_bands = ['band2', 'band3', 'band4']

        desired_types = ['toa', 'sr']

        wanted_files = [str for str in dir_list if any(sub in str for sub in desired_bands)]

        wanted_files = [str for str in wanted_files if any(sub in str for sub in desired_types)]

        return wanted_files

    def multi_channel_img(self, extracted_files):

        """Read each channel into a numpy array.
        Of course your data set may not be images yet.
        So just load them into three different numpy arrays as neccessary"""

        dir_path = 'Extracted/' + self.file_name + '/'
        a = cv2.imread(dir_path + extracted_files[0], 0)
        b = cv2.imread(dir_path + extracted_files[1], 0)
        c = cv2.imread(dir_path + extracted_files[2], 0)

        """Create a blank image that has three channels
        and the same number of pixels as your original input"""
        needed_multi_channel_img = np.zeros((a.shape[0], a.shape[1], 3))

        """Increase the dynamic range of each band"""
        a = a*20
        b = b*20
        c = c*20

        """Add the channels to the needed image one by one"""
        needed_multi_channel_img [:,:,0] = a
        needed_multi_channel_img [:,:,1] = b
        needed_multi_channel_img [:,:,2] = c

        """Save the needed multi channel image"""
        cv2.imwrite(dir_path + 'MultiChannel_' + self.file_name+'.png',needed_multi_channel_img)

        print ("Image " + self.file_name + " PROCESSED ✅")

    def process_image(self):

        desired_files = self.get_file_names()

        if (len(desired_files) > 3 and len(desired_files > 0)):
            print ("INCORRECT NUMBER OF FILES EXTRACED\nFile Name: " + self.file_name + "❌")
            print ("Number of files extracted: " + str(len(desired_files)))
            print ("Deisred Files list: " + str(desired_files))

        self.multi_channel_img(desired_files)
