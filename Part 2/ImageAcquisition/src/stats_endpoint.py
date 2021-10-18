import os
import requests
import json
from requests.auth import HTTPBasicAuth

# our demo filter that filters by geometry, date and cloud cover
from demo_filters import filters_combined


_QUERY = "SEARCH" # "STATS" or "SEARCH" or "ASSETS"
ids = []

if _QUERY == "STATS":

    # Stats API request object
    stats_endpoint_request = {
      "interval": "day",
      "item_types": ["REOrthoTile"],
      "filter": filters_combined
    }

    # fire off the POST request
    result = \
      requests.post(
        'https://api.planet.com/data/v1/stats',
        auth=HTTPBasicAuth('597ec59bdd5449498bd2e2ae37bd02eb', ''),
        json=stats_endpoint_request)

    parsed = json.loads(result.text)
    print ()
    print (json.dumps(parsed, indent=4, sort_keys=True))

if _QUERY == "SEARCH":

    # Search API request object
    search_endpoint_request = {
      "item_types": ["REOrthoTile"],
      "filter": filters_combined
    }

    # fire off the POST request
    result = \
      requests.post(
        'https://api.planet.com/data/v1/quick-search',
        auth=HTTPBasicAuth('597ec59bdd5449498bd2e2ae37bd02eb', ''),
        json=search_endpoint_request)

    parsed = json.loads(result.text)
    print (parsed.keys())
    features = parsed.get("features")
    for feature in features:
        print ("ID: " + str(feature["id"]) + "\n")
        ids.append(feature["id"])
    #print (json.dumps(parsed.get("features")[0]["id"], indent=4, sort_keys=True))

_QUERY = "ASSETS"

if _QUERY == "ASSETS":

    item_id = ids[0]
    print (ids[0])
    item_type = "REOrthoTile"
    asset_type = "visual"

    result = \
      requests.get(
        'https://api.planet.com/data/v1/item-types/{}/items/{}/assets'.format(item_type, item_id),
        auth=HTTPBasicAuth('597ec59bdd5449498bd2e2ae37bd02eb', '')
      )

    print(result.json().keys())
    # extract the activation url from the item for the desired asset
    print (item.text)
    item_activation_url = item.json()[asset_type]["_links"]["activate"]

    # request activation
    response = session.post(item_activation_url)

    print (response["status_code"])
