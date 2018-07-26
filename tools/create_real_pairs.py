import numpy as np
import os
import pickle
from tqdm import tqdm

from scipy.ndimage import imread

image_folder = '../data/image'
depthmap_folder = '../data/depth'
data = {'image':[],'depth':[]}

image_paths = sorted([os.path.join(image_folder, x) for x in os.listdir(image_folder) if x.endswith('.jpg')])
depthmap_paths = sorted([os.path.join(depthmap_folder, x) for x in os.listdir(depthmap_folder) if x.endswith('.pkl')])


for i in tqdm(range(len(image_paths))):
    with open(depthmap_paths[i], 'rb') as fp:

        depth = pickle.load(fp)
        depth = depth.astype(np.float32)
        depth = np.expand_dims(depth, axis=-1)
        data['depth'] = depth

    image = imread(image_paths[i])
    data['image'] = image
    data['image'] = np.array(data['image'])
    data['depth'] = np.array(data['depth'])

    with open('../data/pairs/'+'pair'+str(i)+'.pkl', 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

