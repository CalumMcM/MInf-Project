from datetime import datetime
import requests
import json
import getpass
import zipfile
import tarfile
from ImagePrep import ImagePrep

username = 'CalumMcM'
password = getpass.getpass()

# Interacts with the API and will print the HTTP status of the order and
# return the data within
def espa_api(endpoint, verb='get', body=None, uauth=None):
    host = 'https://espa.cr.usgs.gov/api/v1/'

    """ Suggested simple way to interact with the ESPA JSON REST API """
    auth_tup = uauth if uauth else (username, password)
    response = getattr(requests, verb)(host + endpoint, auth=auth_tup, json=body)
    print('{} {}'.format(response.status_code, response.reason))
    data = response.json()
    if isinstance(data, dict):
        messages = data.pop("messages", None)
        if messages:
            print(json.dumps(messages, indent=4))
    try:
        response.raise_for_status()
    except Exception as e:
        print(e)
        return None
    else:
        return data

# Check the status of the order
def check_order_status(orderid):

    print ('Checking order status...')

    print('GET /api/v1/order-status/{}'.format(orderid))

    resp = espa_api('order-status/{}'.format(orderid))

    print(json.dumps(resp, indent=4))

# Prin the list of download urls
def get_image_list(orderid, resp):

    for item in resp[orderid]:
        print("URL: {0}".format(item.get('product_dload_url')))

# Takes the download url and returns the file name at the end
def seperate_url(url):

    filename = url.split('/')[-1]

    return filename.split('.')[0]

# Downloads images and extracts them
def download_images(orderid, resp):

    print ("Beginning Download...")
    now = datetime.now()

    # Time at start
    current_time = now.strftime("%H:%M:%S")
    print("Start Time =", current_time)

    for item in resp[orderid]:

        url = item.get('product_dload_url')

        r = requests.get(url, allow_redirects=True)

        file_name = seperate_url(url)

        # Save contents of url to local directory
        open('raw_images/' + file_name + '.tar.gz', 'wb').write(r.content)

        # Extract files in '.tar.gz' and save to new directory
        tar = tarfile.open('raw_images/' + file_name + '.tar.gz', "r:gz")
        tar.extractall('Extracted/' + file_name)
        tar.close()

        # Create a new ChannelMerge class and begin processing of the image
        imagePrep = ImagePrep(file_name)
        imagePrep.prepareImage();

    # Time at finish
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print("End Time =", current_time)

# Main wrapper function that also gets the password for the user
def main():

    orderid = input('Enter orderid: ') # espa-calummcmeekin@mac.com-08242020-042201-261
                                       # espa-P100fdrjxk6lm
    check_order_status(orderid)

    # Get order
    resp = espa_api('item-status/{0}'.format(orderid), body={'status': 'complete'})

    get_image_list(orderid, resp)

    download_images(orderid, resp)

    print ("Complete")

if __name__ == '__main__':
    main()
