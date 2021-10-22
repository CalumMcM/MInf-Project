from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive

def main():
    gauth = GoogleAuth()
    drive = GoogleDrive(gauth)

    DIR = r'/Volumes/GoogleDrive/My\ Drive/PlanetScope\ Imagery/Quads/'
    upload_file = '632-1034.tiff'
    """

    """
    get_files(gauth, drive)
    #gfile = drive.CreateFile({'parents': [{'id': '1vma9es9Kd36m8pIrTSEkZ5J3C-nY66ao'}]})
    #gfile.SetContentFile(upload_file)
    #gfile.Upload() # Upload the file.

def get_files(gauth, drive):
    # Check if file exists
    biome_dir_query = "'1MA0AUgR_ucfXc5OGvDadF2AzVDUXG_9U' in parents and trashed=false"
    file_list = drive.ListFile({'q': "'1uq0IW28Sb7r3rtzW5_V_3cZJ7W7K16za' in parents and trashed=false"}).GetList()
    file_names = [file['title'] for file in file_list]

    print (file_names)
    #for file1 in file_list:
    #    print (file1['title'])
        #if file1['title'] == gpath:
            #id = file1['id']

if __name__ == "__main__":
    main()
