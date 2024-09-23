import json
import numpy as np
import os

def sample(args):
    d = None
    with open(args['data']) as f:
        d = json.load(f)

    cats = {'poses': [], 'shapes': [], 'supercategory': []}
    for i in d['data']:
        cats['poses'].append(i['pose'][3:])
        cats['shapes'].append(i['shape'])
        cats['supercategory'].append(i['supercategory'])
    cats['poses'] = np.array(cats['poses']).reshape(len(d['data']), -1, 3)
    cats['shapes'] = np.array(cats['shapes'])
    cats['supercategory'] = np.array(cats['supercategory'])

    np.savez(args['save'] + '.npz', pose_body=cats['poses'], betas = cats['shapes'], categories = cats['supercategory'])




if __name__ == '__main__':
    args = {
        'save': 'dataset/animal3d/SAMPLED_POSES/test',
        'data': 'dataset/animal3d/test.json'
    }
    os.mkdir("dataset/animal3d/SAMPLED_POSES", exist_ok=True)
    os.mkdir(args['save'], exist_ok=True)

    sample(args)