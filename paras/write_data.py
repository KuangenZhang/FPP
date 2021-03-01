import numpy as np
import json
rgbd_tracker_paras = np.load('rgbd_tracker_paras.npy', allow_pickle=True).item()

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

def save_dict_to_file(rgbd_tracker_paras):
    f = open('rgbd_tracker_paras.txt','w')
    f.write(str(rgbd_tracker_paras))
    f.close()

def save_dict_to_json(rgbd_tracker_paras):
    with open('rgbd_tracker_paras.json', 'w') as result_file:
        json.dump(rgbd_tracker_paras, result_file, cls=NumpyEncoder)

save_dict_to_json(rgbd_tracker_paras)

