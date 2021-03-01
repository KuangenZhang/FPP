import json
import numpy as np

def read_dic_from_json():
    with open('rgbd_tracker_paras.json', 'r') as json_file:
        data = json.load(json_file)
        for key in data.keys():
            data[key] = np.asarray(data[key])
        return data

rgbd_tracker_paras = read_dic_from_json()
print(rgbd_tracker_paras)