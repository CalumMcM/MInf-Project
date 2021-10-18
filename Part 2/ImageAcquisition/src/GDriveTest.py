from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive

def main():
    gauth = GoogleAuth()
    drive = GoogleDrive(gauth)

    DIR = r'/Volumes/GoogleDrive/My\ Drive/PlanetScope\ Imagery/Quads/'
    upload_file = 'test.txt'

    gfile = drive.CreateFile({'parents': [{'id': '1vma9es9Kd36m8pIrTSEkZ5J3C-nY66ao'}]})
    # Read file and set it as the content of this instance.
    gfile.SetContentFile(upload_file)
    gfile.Upload() # Upload the file.

if __name__ == "__main__":
    main()
