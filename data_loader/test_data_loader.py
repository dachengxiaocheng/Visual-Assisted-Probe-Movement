import os
import glob
import torch
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import cv2
import math
import random
from sklearn.neighbors import NearestNeighbors
from pyquaternion import Quaternion
from scipy.spatial.transform import Rotation as R


class Image_Motion_Data_Evaluation(Dataset):
    def __init__(self, dataset_root_dir, image_filetype, motion_filetype, image_size, topN=20):
        super().__init__()
        self.image_size = image_size
        self.topN = topN

        query_image_path = os.path.join(dataset_root_dir, 'image_query', '*.{}'.format(image_filetype))
        self.query_imfiles = sorted(glob.glob(query_image_path))
        query_position_path = os.path.join(dataset_root_dir, 'motion_query', 'position.{}'.format(motion_filetype))
        self.query_position = np.loadtxt(query_position_path, delimiter=',')
        query_orientation_path = os.path.join(dataset_root_dir, 'motion_query', 'orientation.{}'.format(motion_filetype))
        self.query_orientation = np.loadtxt(query_orientation_path, delimiter=',')
        self.query_len = len(self.query_imfiles)

        database_image_path = os.path.join(dataset_root_dir, 'image_database', '*.{}'.format(image_filetype))
        self.database_imfiles = sorted(glob.glob(database_image_path))
        database_position_path = os.path.join(dataset_root_dir, 'motion_database', 'position.{}'.format(motion_filetype))
        self.database_position = np.loadtxt(database_position_path, delimiter=',')
        database_orientation_path = os.path.join(dataset_root_dir, 'motion_database', 'orientation.{}'.format(motion_filetype))
        self.database_orientation = np.loadtxt(database_orientation_path, delimiter=',')
        self.database_len = len(self.database_imfiles)

        self.nbrs = NearestNeighbors(algorithm='auto', n_jobs=-1).fit(self.database_position)
        # self.distance, self.indices = self.nbrs.kneighbors(self.query_position, return_distance=True)
        self.distance, self.indices = self.nbrs.radius_neighbors(self.query_position, radius=100, return_distance=True, sort_results=True)
        return

    def __len__(self):
        return self.database_len

    def __getitem__(self, item):
        if item < self.query_len:
            flag = 0
            query_image = self.get_image(self.query_imfiles[item])
            query_image = np.expand_dims(query_image, axis=0)

            database_image = self.get_image(self.database_imfiles[item])
            database_image = np.expand_dims(database_image, axis=0)

            indices = self.indices[item][: self.topN]
            distance = self.distance[item][: self.topN]
            return flag, query_image, database_image, indices, distance
        else:
            flag = 1
            database_image = self.get_image(self.database_imfiles[item])
            database_image = np.expand_dims(database_image, axis=0)
            return flag, [], database_image, [], []

    def get_image(self, imfile):
        image = cv2.imread(imfile, cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image.astype('float32'), self.image_size) # cv2.resize() uses (width, height) for resize
        image = image / 255.
        return image


if __name__ == "__main__":
    data_train = Image_Motion_Data_Evaluation(dataset_root_dir='/home/engs2191/simulation_PULSE_data',
                                    image_filetype='png',
                                    motion_filetype='csv',
                                    image_size=(400, 274), # (801, 547)
                                    topN=20
                                    )

    dl_train = DataLoader(data_train, batch_size=10, shuffle=False, num_workers=0, pin_memory=True)

    print('The size of train dataset loader: ', len(dl_train))
    for batch_idx, (flag, query_image, database_image, indices, distance) in enumerate(dl_train):
        print('batch_idx:', batch_idx)
        print('query_image.size:', query_image.size())
        print('database_image.size:', database_image.size())
        print('indices.size:', indices.size())
        print('distance.size:', distance.size())

    print('Finish testing!')
