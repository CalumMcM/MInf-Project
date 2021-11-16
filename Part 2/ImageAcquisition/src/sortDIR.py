import os
from shutil import copyfile 

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
    if (year == '2015' or year == '2016'):
        return True
    if (year == '2017'):
        if (biome == "TemporalAmazonia" or biome == 'TemporalCerrado'):
            return True
    return False

def main():

    head_DIR = '/Volumes/GoogleDrive/My Drive/' # G Drive folder location
   
    dest_DIR = '/Volumes/GoogleDrive/My Drive/TemporalData'
   
    biomes = ['TemporalAmazonia', 'TemporalCerrado', 'TemporalCaatinga']
   
    years = ['2015', '2016', '2017', '2018']
                
    Ama_DIR = make_biome_folder('TemporalAmazonia', dest_DIR)
    Cer_DIR = make_biome_folder('TemporalCerrado', dest_DIR)
    Cat_DIR = make_biome_folder('TemporalCaatinga', dest_DIR)

    for year in years:

        print ("YEAR: {}".format(year))

        for biome in biomes:
            print ("BIOME: {}".format(biome))
            if not (condition(biome, year)):

                src_DIR = os.path.join(head_DIR, biome+year)

                dir_contents = os.listdir(src_DIR)

                dest_biome_DIR = os.path.join(dest_DIR, biome)

                quad1_dir = make_quad_folder(1, dest_biome_DIR)
                quad2_dir = make_quad_folder(2, dest_biome_DIR)
                quad3_dir = make_quad_folder(3, dest_biome_DIR)
                quad4_dir = make_quad_folder(4, dest_biome_DIR)

                quad_dirs = [quad1_dir, quad2_dir, quad3_dir, quad4_dir]
                
                for filename in dir_contents:
                    if 'tif' in filename:
                        meta        = filename.split('_')
                        quad        = meta[0]
                        img_id      = meta[1]
                        seed        = img_id.split('-')[0]
                        img_num     = img_id.split('-')[1]
                        img_year        = meta[2]

                        quad_dir = quad_dirs[int(quad)-1]
                        
                        # Refresh directory
                        updated_subfolders = [f.path for f in os.scandir(quad_dir) if f.is_dir()]
                        img_folder_path = os.path.join(quad_dir, img_id)

                        if not img_folder_path in updated_subfolders:
                            os.mkdir(img_folder_path)
                        
                        original_file_DIR = os.path.join(src_DIR, filename)
                        new_file_DIR = os.path.join(img_folder_path, img_year)

                        copyfile(original_file_DIR, new_file_DIR)
                        #print ("Image: {} \t Quad: {} \t Seed: {} \t Year: {}".format(img_num, quad, seed, year))

if __name__ == "__main__":
    main()