import requests
import json
import os
import urllib.request
import random
import matplotlib.pyplot as plt
import rasterio
import gzip
import numpy as np
from os.path import *
from os import listdir
import pickle
from rasterio.plot import show
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive

def get_mosaics(API_URL, session):
    # Arrays to hold a list of all start and end dates
    start_dates = []
    end_dates = []

    res = session.get(API_URL)
    mosaics = json.loads(res.text)
    print ("Number of Mosaics: " + str(len(mosaics["mosaics"])))
    for mosaic in mosaics["mosaics"]:
        start_dates.append(mosaic["first_acquired"])
        end_dates.append(mosaic["last_acquired"])

    # Sort both arrays from first photos taken to final photos taken
    start_dates.sort()
    end_dates.sort()

    return start_dates, end_dates, mosaics

def get_mosaic_id(start_date, mosaics):
    """
    Search and get the mosaic associated with the desired start date
    """
    mosaic_id = ""
    for mosaic in mosaics["mosaics"]:
        if mosaic["first_acquired"] == start_date:
            return mosaic["id"]
    return "ERROR"

def get_quads_for_date(items, bbox_name):
    """
    For the first time each bbox is approached 20% of quads will be randomly selected for export
    """

    with open('selected_quads.pickle', 'rb') as file:
        selected = pickle.load(file)

        if bbox_name not in selected:
            # Randomly extract 20% of quads in the geometry
            selected[bbox_name] = np.random.choice(items, int(len(items)*0.2), replace=False)
            selected_quads = selected[bbox_name]
            file.close()

            # Save new quads dictionary
            with open('selected_quads.pickle', 'wb') as file:
                pickle.dump(selected, file, protocol=pickle.HIGHEST_PROTOCOL)
                file.close()

        else:
            selected_quads = selected[bbox_name]
            file.close()

    return selected_quads

# todo: currently only takes first 50 images, need to add paginination
def download_Quads(items, bbox_name, start_dates, cur_date, gauth, drive):
    """
    Randomly selects 20% of the available quads and downloads them
    """
    # Location to save the quads
    DIR = '/Volumes/GoogleDrive/My Drive/PlanetScope Imagery/Quads/' # G Drive folder location


    selected_quads = get_quads_for_date(items, bbox_name)

    count = 0
    num_items = len(selected_quads)

    # Determine which biome we are in
    # and what folder the quads should belong to
    biome_id = ''
    biome_name = ''
    biome_dir_query = ''
    print (bbox_name)
    if 'Ama' in bbox_name:
        biome_id = '1c3bP0XyztMpCMivuscMMoOKpfn_8m_Gp'
        biome_name = 'Amazon'
        biome_dir_query = "'1c3bP0XyztMpCMivuscMMoOKpfn_8m_Gp' in parents and trashed=false"
    elif 'Cer' in bbox_name:
        biome_id = '1yIytjv8yX_hSJv4vWjXFt5jStrObeQVo'
        biome_name = 'Cerrado'
        biome_dir_query = "'1yIytjv8yX_hSJv4vWjXFt5jStrObeQVo' in parents and trashed=false"
    elif 'Cat' in bbox_name:
        biome_id = '1MA0AUgR_ucfXc5OGvDadF2AzVDUXG_9U'
        biome_name = 'Caatinga'
        biome_dir_query = "'1MA0AUgR_ucfXc5OGvDadF2AzVDUXG_9U' in parents and trashed=false"

    temp_DIR = os.path.join(DIR, biome_name)
    temp_DIR = os.path.join(temp_DIR, bbox_name)


    # Get list of all quad folders already
    # generated for this bbox
    quads_list = drive.ListFile({'q': biome_dir_query}).GetList()
    quad_dirs = [file['title'] for file in quads_list]
    print (quad_dirs)
    for chosen_quad in items:
        # Only download selected quads
        print (chosen_quad)
        if (chosen_quad in selected_quads):
            print ("Progress: {:.2f}% ({}/{:.0f})".format(count/(num_items*0.2)*100, count+1, num_items*0.2))

            # Format name
            download_link = chosen_quad['_links']['download']
            name = chosen_quad['id']

            # Add quad to DIR
            quad_DIR = os.path.join(temp_DIR, name)
            print (quad_DIR)

            # If this quad does not have a folder then create it
            if not chosen_quad in quad_dirs:
                print ("MAKING NEW QUAD DIR: {}".format(name))
                folder_name = input(quad_DIR)
                folder = drive.CreateFile({'title' : folder_name, 'mimeType' : 'application/vnd.google-apps.folder'})
                folder.Upload()

            # Set image name to be current date
            date_DIR = os.path.join(quad_DIR, cur_date)

            # Get list of all quad folders already
            # generated for this bbox
            for quad_DIR in quads_list:
                if quad_DIR['title'] == name:
                    quad_id = quad_DIR['id']
                    quad_dir_query = "'"+quad_id+"' in parents and trashed=false"
                    break

            dates_list = drive.ListFile({'q': biome_dir_query}).GetList()
            date_DIRs = [file['title'] for file in dates_list]

            # Check this date has not already been downloaded
            if not cur_date in date_DIRs:
                print ("MAKING NEW DATE DIR: {}".format(date))
                folder_name = input(date_DIR)
                folder = drive.CreateFile({'title' : folder_name, 'mimeType' : 'application/vnd.google-apps.folder'})
                folder.Upload()

            # Append filetype to filename
            filename = name+'.tiff'

            # Get all files in this date directory
            # after first getting the current date directory ID
            for date_DIR in dates_list:
                if date_DIR['title'] == cur_date:
                    date_id = date_DIR['id']
                    date_dir_query = "'"+date_id+"' in parents and trashed=false"
                    break

            files_list = drive.ListFile({'q': biome_dir_query}).GetList()
            file_DIRs = [file['title'] for file in dates_list]

            # If quad has not already been downloaded
            # download the quad
            if not filename+'.tiff' in file_DIRs:
                # Download quad to local machine
                print ("RETRIEVING: " + str(file_DIR))
                urllib.request.urlretrieve(download_link, filename)

                print ("UPLOADING: {}".format(filename))
                # Upload quad to date folder in GDrive
                gfile = drive.CreateFile({'parents': [{'id': date_id}]})
                gfile.SetContentFile(filename)
                gfile.Upload() # Upload the file

                count += 1

    print ("Progress: Complete")


def acquire_images(start_dates, end_dates, mosaics, API_URL, session,gauth, drive):

    # Go through each year
    for cur_date in start_dates:

        mosaic_id = get_mosaic_id(cur_date, mosaics)

        print ("YEAR: {} MOSAIC_ID: {}".format(cur_date, mosaic_id))

        for bbox in bbox_dict:

            print ("BBOX: " + str(bbox))

            search_parameters = {
                'bbox': bbox_dict[bbox], # Specified geometry
                 'minimal': True,
                "item_types": ["REOrthoTile"] # Satellite type
            }

            # Get all quads for the current bbox
            quads_url = "{}/{}/quads?_page_size=100000".format(API_URL, mosaic_id) # Page size set to arbitrarily high number

            res = session.get(quads_url, params=search_parameters, stream=True)

            quads = res.json()

            items = quads['items']

            print("QUAD LIST API STATUS: {} NUMBER OF QUADS: {}".format(str(res.status_code), len(items)))

            download_Quads(items, bbox, start_dates, cur_date, gauth, drive)

def main():

    #setup API KEY
    PLANET_API_KEY = '597ec59bdd5449498bd2e2ae37bd02eb' # <= insert API key here

    # Generic API URL
    API_URL = 'https://api.planet.com/basemaps/v1/mosaics'

    #setup session
    session = requests.Session()

    #authenticate
    session.auth = (PLANET_API_KEY, "") #<= change to match variable for API Key if needed


    # Set up session with Google drive to store the images:
    gauth = GoogleAuth()
    drive = GoogleDrive(gauth)

    # Get mosaics and sort them from oldest to latest
    start_dates, end_dates, mosaics = get_mosaics(API_URL, session)

    acquire_images(start_dates, end_dates, mosaics, API_URL, session, gauth, drive)

AmaQuad1_1 = "-64.0283203125, 0.5273363048115169, -59.853515625, 3.908098881894123"
AmaQuad1_2 = "-68.92822265625, 0.4833927027896987, -66.9287109375, 1.845383988573187"
AmaQuad1_3 = "-68.8623046875, -4.36832042087623, -58.7, 0.3076157096439005"

AmaQuad2_1 = "-58.447265625, -4.674979814820243, -50.9765625, 1.098565496040652"
AmaQuad2_2 = "-50.8447265625, -4.740675384778361, -46.845703125, -1.669685500986571"
AmaQuad2_3 = "-46.7578125, -3.9519408561575817, -45.3076171875, -1.7575368113083125"

AmaQuad3_1 = "-70.2685546875, -11.005904459659451, -60.380859375, -4.302591077119676"
AmaQuad3_2 = "-72.50976562499999, -9.579084335882534, -70.3125, -4.784468966579362"

AmaQuad4_1 = "-60.3369140625, -10.790140750321738, -53.96484375, -4.521666342614791"
AmaQuad4_2 = "-53.78906249999999, -7.798078531355303, -49.7900390625, -4.696879026871413"
AmaQuad4_3 = "-53.6572265625, -12.554563528593656, -52.82226562499999, -7.972197714386866"
AmaQuad4_4 = "-52.470703125, -9.88227549342994, -51.328125, -8.059229627200192"

CatQuad1_1 = "-39.17876581754097,-8.111804629213445, -42.23296503629097,-3.0618983616936823"

CatQuad2_1 = "-39.13330078125,-8.515835561202218, -35.595703125, -5.266007882805485"
CatQuad2_2 = "-35.57373046875, -8.515835561202218, -35.15625, -6.140554782450295"

CatQuad3_1 = " -40.078125, -10.09867012060338, -36.298828125, -8.537565350804018"
CatQuad3_2 = "-40.05615234375, -12.490213662533295, -38.3642578125, -10.163560279490476"
CatQuad3_3 = "-40.0341796875, -14.75363533154043, -39.5068359375, -12.490213662533295"
CatQuad3_4 = "-38.29833984375, -11.587669416896203, -37.8369140625, -10.18518740926906"

CatQuad4_1 = "-42.0556640625, -14.77488250651626, -40.0341796875, -8.146242825034385"
CatQuad4_2 = "-42.12158203124999,-15.866241564066616, -40.693359375, -14.84923123791421"

CerQuad1_1 = "-46.6259765625, -16.34122561920748, -42.84667968749999, -8.581021215641842"
CerQuad1_2 = "-45.6591796875, -8.363692651835823, -42.890625, -6.533645130567532"
CerQuad1_3 = "-44.384765625, -6.468151012664202, -42.91259765625, -5.156598738411155"

CerQuad2_1 = "-55.06347656249999, -17.056784609942543, -46.71386718749999, -14.349547837185362"
CerQuad2_2 = "-50.6689453125, -14.221788628397585, -46.6259765625,-11.73830237143684"
CerQuad2_3 = "-48.8232421875, -11.566143767762844, -46.669921875, -10.487811882056683"
CerQuad2_4 = "-47.197265625, -10.31491928581316,-46.58203125, -8.49410453755187"

CerQuad3_1 = "-49.482421875, -24.766784522874428, -48.2958984375, -17.182779056431826"
CerQuad3_2 = "-48.2080078125, -23.52370005882413, -46.845703125, -17.014767530557823"
CerQuad3_3 = "-46.669921875, -21.555284406923178, -45.3076171875, -16.425547506916725"
CerQuad3_4 = "-45.24169921875,-20.529933125170764, -43.81347656249999,-16.383391123608387"
CerQuad3_5 = "-43.70361328125, -19.476950206488414, -42.86865234375, -16.34122561920748"

CerQuad4_1 = "-54.9755859375, -22.75592068148639, -49.7900390625, -17.09879223767869"
CerQuad4_2 = "-50.625, -24.846565348219734, -49.658203125, -22.958393318086337"
CerQuad4_3 = "-57.52441406249999, -21.657428197370642, -55.1953125,-19.26966529650232"

bbox_dict = {
    "AmaQuad1_1" : AmaQuad1_1,
    "AmaQuad1_2" : AmaQuad1_2,
    "AmaQuad1_3" : AmaQuad1_3,
    "AmaQuad2_1" : AmaQuad2_1,
    "AmaQuad2_2" : AmaQuad2_2,
    "AmaQuad2_3" : AmaQuad2_3,
    "AmaQuad3_1" : AmaQuad3_1,
    "AmaQuad3_2" : AmaQuad3_2,
    "AmaQuad4_1" : AmaQuad4_1,
    "AmaQuad4_2" : AmaQuad4_2,
    "AmaQuad4_3" : AmaQuad4_3,
    "AmaQuad4_4" : AmaQuad4_4,

    "CatQuad1_1" : CatQuad1_1,
    "CatQuad2_1" : CatQuad2_1,
    "CatQuad2_2" : CatQuad2_2,
    "CatQuad3_1" : CatQuad3_1,
    "CatQuad3_2" : CatQuad3_2,
    "CatQuad3_3" : CatQuad3_3,
    "CatQuad3_4" : CatQuad3_4,
    "CatQuad4_1" : CatQuad4_1,
    "CatQuad4_2" : CatQuad4_2,

    "CerQuad1_1" : CerQuad1_1,
    "CerQuad1_2" : CerQuad1_2,
    "CerQuad1_3" : CerQuad1_3,
    "CerQuad2_1" : CerQuad2_1,
    "CerQuad2_2" : CerQuad2_2,
    "CerQuad2_3" : CerQuad2_3,
    "CerQuad2_4" : CerQuad2_4,
    "CerQuad3_1" : CerQuad3_1,
    "CerQuad3_2" : CerQuad3_2,
    "CerQuad3_3" : CerQuad3_3,
    "CerQuad3_4" : CerQuad3_4,
    "CerQuad3_5" : CerQuad3_5,
    "CerQuad4_1" : CerQuad4_1,
    "CerQuad4_2" : CerQuad4_2,
    "CerQuad4_3" : CerQuad4_3

}

if __name__ == "__main__":
    main()
