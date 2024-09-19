import json
import numpy as np

def sample(args):
    d = None
    with open(args['directory'] + args['data']) as f:
        d = json.load(f)

    cats = {'poses': [], 'shapes': [], 'supercategory': []}
    for i in d['data']:
        cats['poses'].append(i['pose'][3:])
        cats['shapes'].append(i['shape'])
        cats['supercategory'].append(i['supercategory'])
    cats['poses'] = np.array(cats['poses']).reshape(len(d['data']), -1, 3)
    cats['shapes'] = np.array(cats['shapes'])
    cats['supercategory'] = np.array(cats['supercategory'])

    np.savez(args['directory'] + args['save'] + '.npz', pose_body=cats['poses'], betas = cats['shapes'], categories = cats['supercategory'])




if __name__ == '__main__':
    args = {

        # 'directory': '/vol/bitbucket/mew23/individual_project_diffusion_example/',
        'directory': '/vol/bitbucket/mew23/individual-project/',

        'save': 'dataset/animal3d/SAMPLED_POSES/test',
        'data': 'dataset/animal3d/test.json'
    }

    sample(args)