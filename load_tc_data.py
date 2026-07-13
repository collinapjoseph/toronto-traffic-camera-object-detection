import requests
import pandas as pd
from io import StringIO

# Toronto Open Data - Traffic Cameras Dataset. 
# Doc: https://open.toronto.ca/dataset/traffic-cameras/

def get_toronto_traffic_cameras():
    base_url = "https://ckan0.cf.opendata.inter.prod-toronto.ca"
    url = base_url + "/api/3/action/package_show"
    params = { "id": "traffic-cameras"}
    package = requests.get(url, params = params).json()

    tc_dataset = pd.DataFrame()
    for resource in package["result"]["resources"]:

        # Find active datastore.
        if resource["datastore_active"]:
            print("Loading:", resource["name"], " ", resource["last_modified"])
            # Get all records in CSV format:
            url = base_url + "/datastore/dump/" + resource["id"]
            resource_dump_data = requests.get(url).text
            tc_dataset = pd.read_csv(StringIO(resource_dump_data))
            tc_dataset["INTERSECTION"] = tc_dataset["MAINROAD"] + " / " + tc_dataset["CROSSROAD"]

    return tc_dataset
        