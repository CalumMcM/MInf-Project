import os
from shutil import copyfile 
import numpy as np

"""
Goes through a given list of directories that store the yearly iamges 
for each array, combining images of the same area into one shared directory

"""
def make_biome_folder(biomeName: str, DIR: str):
    subfolders = [f.path for f in os.scandir(DIR) if f.is_dir()]
    
    path = os.path.join(DIR, biomeName)

    if not path in subfolders:
        os.mkdir(path)

    return path

def make_quad_folder(quadNum: int, DIR: str):
    subfolders = [f.path for f in os.scandir(DIR) if f.is_dir()]

    quad_folder_name = 'Quad_'+str(quadNum)
    
    path = os.path.join(DIR, quad_folder_name)

    if not path in subfolders:
        os.mkdir(path)
    
    return path

def condition(biome, year):
    return False
    # if (year == '2016' or year == '2017'):
    #     return True
    # if (year == '2017'):
    #     if (biome == "TemporalAmazonia" or biome == 'TemporalCerrado'):
    #         return True
    # return False

def main():

    _TRAIN = False

    head_DIR = '/Volumes/GoogleDrive/My Drive/' # G Drive folder location
    
    if _TRAIN:
        dest_DIR = '/Volumes/GoogleDrive/My Drive/TemporalData-Train'
        years = ['2015', '2016', '2017', '2018']
    else:
        dest_DIR = '/Volumes/GoogleDrive/My Drive/TemporalData-Eval'
        years = ['2016', '2017', '2018', '2019']

    biomes = ['TemporalAmazonia', 'TemporalCerrado', 'TemporalCaatinga']
    
    make_biome_folder('TemporalAmazonia', dest_DIR)
    make_biome_folder('TemporalCerrado', dest_DIR)
    make_biome_folder('TemporalCaatinga', dest_DIR)

    for year in years:

        print ("YEAR: {}".format(year))

        for biome in biomes:
            print ("BIOME: {}".format(biome))
            if not (condition(biome, year)):

                src_DIR = os.path.join(head_DIR, biome+year)

                dir_contents = np.sort(os.listdir(src_DIR))

                dest_biome_DIR = os.path.join(dest_DIR, biome)

                quad1_dir = make_quad_folder(1, dest_biome_DIR)
                quad2_dir = make_quad_folder(2, dest_biome_DIR)
                quad3_dir = make_quad_folder(3, dest_biome_DIR)
                quad4_dir = make_quad_folder(4, dest_biome_DIR)

                quad_dirs = [quad1_dir, quad2_dir, quad3_dir, quad4_dir]


                if _TRAIN:
                    if biome == "TemporalAmazonia":
                        desired_quads = [2,3,4]

                    if biome == "TemporalCerrado":
                        desired_quads = [1,2,3]

                    if biome == "TemporalCaatinga":
                        desired_quads = [1,2,4]

                else:
                    if biome == "TemporalAmazonia":
                        desired_quads = [1]
                        
                    if biome == "TemporalCerrado":
                        desired_quads = [4]

                    if biome == "TemporalCaatinga":
                        desired_quads = [3]

                for filename in dir_contents:
                    if 'tif' in filename:
                        meta        = filename.split('_')
                        quad        = meta[0]
                        img_id      = meta[1]
                        seed        = img_id.split('-')[0]
                        img_num     = img_id.split('-')[1]
                        img_year    = meta[2]

                        # Only copy images that are actually needed
                        if (int(quad) in desired_quads):
                            quad_dir = quad_dirs[int(quad)-1]
                        
                            # Refresh directory
                            updated_subfolders = [f.path for f in os.scandir(quad_dir) if f.is_dir()]
                            img_folder_path = os.path.join(quad_dir, img_num)

                            if not img_folder_path in updated_subfolders:
                                os.mkdir(img_folder_path)
                            
                            original_file_DIR = os.path.join(src_DIR, filename)
                            new_file_DIR = os.path.join(img_folder_path, img_year)

                            copyfile(original_file_DIR, new_file_DIR)

if __name__ == "__main__":
    main()