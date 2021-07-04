import os
import glob
import torch
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import cv2
import math
import random
import sklearn
from sklearn.neighbors import NearestNeighbors
from pyquaternion import Quaternion
from scipy.spatial.transform import Rotation as R


class Image_Motion_Data_Training(Dataset):
    def __init__(self, dataset_root_dir, image_filetype, motion_filetype, image_size, pos_number, neg_number, train=True):
        super().__init__()
        self.image_size = image_size
        self.train = train
        self.pos_number = pos_number
        self.neg_number = neg_number

        image_path = os.path.join(dataset_root_dir, 'image', '*.{}'.format(image_filetype))
        self.imfiles = sorted(glob.glob(image_path))
        position_path = os.path.join(dataset_root_dir, 'motion', 'position.{}'.format(motion_filetype))
        self.position = np.loadtxt(position_path, delimiter=',')
        orientation_path = os.path.join(dataset_root_dir, 'motion', 'orientation.{}'.format(motion_filetype))
        self.orientation = np.loadtxt(orientation_path, delimiter=',')

        self.nbrs = NearestNeighbors(algorithm='auto', n_jobs=-1).fit(self.position)
        # self.distance, self.indices = self.nbrs.kneighbors(self.position, n_neighbors=len(self.imfiles), return_distance=True)
        self.distance, self.indices = self.nbrs.radius_neighbors(self.position, radius=100, return_distance=True, sort_results=True)
        return

    def __len__(self):
        return len(self.imfiles)

    def __getitem__(self, item):
        position = self.position[item]
        orientation = self.orientation[item]
        ##############################################################################
        pos_image = []
        neg_image = []

        query_image = self.get_image(item)
        query_image = np.expand_dims(query_image, axis=0)
        ##############################################################################
        query_index = self.indices[item]
        pos_index = query_index[1: 1+self.pos_number]
        # pos_index = query_index[: self.pos_number]
        for i in pos_index:
            pos_image.append(self.get_image(i))
        pos_image = np.asarray(pos_image)

        pos_distance = self.distance[item]
        # pos_distance = pos_distance[1: 1+self.pos_number]
        pos_distance = pos_distance[1: 50]
        # pos_distance = pos_distance[: self.pos_number]
        ##############################################################################
        neg_index = np.setdiff1d(np.arange(self.position.shape[0]), query_index)
        neg_index = np.random.choice(neg_index, self.neg_number)
        neg_distance = []
        for i in neg_index:
            neg_image.append(self.get_image(i))
            neg_position = self.position[i]
            neg_distance.append(np.linalg.norm(position-neg_position))
        neg_image = np.asarray(neg_image)
        neg_distance = np.asarray(neg_distance)
        ##############################################################################
        return query_image, pos_image, neg_image, pos_distance, neg_distance, position, orientation

    def get_image(self, index):
        imfile = self.imfiles[index]
        image = cv2.imread(imfile, cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image.astype('float32'), self.image_size) # cv2.resize() uses (width, height) for resize
        image = image / 255.
        return image


if __name__ == "__main__":
    data_train = Image_Motion_Data_Training(dataset_root_dir='/home/engs2191/simulation_PULSE_data',
                                    image_filetype='png',
                                    motion_filetype='csv',
                                    image_size=(400, 274), # (801, 547)
                                    pos_number=3,
                                    neg_number=10,
                                    train=True)
    dl_train = DataLoader(data_train, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)

    print('The size of train dataset loader: ', len(dl_train))
    for batch_idx, (query_image, pos_image, neg_image, position, orientation) in enumerate(dl_train):
        print('batch_idx:', batch_idx)
        print('image.size:', query_image.size())
        print('pos_image.size:', pos_image.size())
        print('neg_image.size:', neg_image.size())
        print('position.size:', position.size())
        print('orientation.size:', orientation.size())
    print('Finish testing!')
