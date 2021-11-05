from datetime import datetime
import time
import requests


url = "http://www.bbc.com"
timeout = 5

while (True):
    now = datetime.now()
    try:
        request = requests.get(url, timeout=timeout)
        
        current_time = now.strftime("%H:%M:%S")
        print("Connected to the Internet @ {}".format(current_time))
        time.sleep(4)
    except (requests.ConnectionError, requests.Timeout) as exception:
        current_time = now.strftime("%H:%M:%S")
        print("NO INTERNET @ {}".format(current_time))
        time.sleep(4)

