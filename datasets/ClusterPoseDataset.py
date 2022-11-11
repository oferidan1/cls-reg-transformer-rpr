import random
from skimage.io import imread
from torch.utils.data import Dataset
import pandas as pd
from os.path import join
import numpy as np
from collections import defaultdict
import joblib
import torch

class ClusterPoseDataset(Dataset):
    def __init__(self, data_path, pairs_file, transform=None):
        self.img_path1, self.img_path2, self.poses, self.scenes, self.scene_ids, \
        self.orientation_cluster_ids , self.position_cluster_ids = \
            read_pairs_file(data_path, pairs_file)
        self.transform = transform
        self.dataset_size = self.poses.shape[0]
        self.num_clusters_q = np.max(self.orientation_cluster_ids) + 1
        self.num_clusters_x = np.max(self.position_cluster_ids) + 1

    def __len__(self):
        return len(self.img_path1)

    def __getitem__(self, idx):
        img1 = imread(self.img_path1[idx])
        img2 = imread(self.img_path2[idx])
        pose = self.poses[idx]
        scene = self.scene_ids[idx]
        position_cluster_id = int(self.position_cluster_ids[idx])
        orientation_cluster_id = int(self.orientation_cluster_ids[idx])
        #position_centroids = self.position_centroids#[scene1]
        #orientation_centroids = self.orientation_centroids#[scene1]

        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        sample = {'img1': img1, 'img2': img2, 'pose': pose, 'scene': scene,
                  'position_cluster_id': position_cluster_id, 'orientation_cluster_id': orientation_cluster_id
                  }
        return sample

    def calc_cluster_weights(self):
        cluser_x_cnt = []
        total = 0
        for i in range(self.num_clusters_x):
            idx_in_cluster = np.where(self.position_cluster_ids == i)
            cluser_x_cnt.append(len(idx_in_cluster[0]))
            total += len(idx_in_cluster[0])

        cluser_x_weight = []
        for i in range(len(cluser_x_cnt)):
            weight = cluser_x_cnt[self.num_clusters_x-1] / cluser_x_cnt[i]
            cluser_x_weight.append(weight)

        cluser_q_cnt = []
        for i in range(self.num_clusters_q):
            idx_in_cluster = np.where(self.orientation_cluster_ids == i)
            cluser_q_cnt.append(len(idx_in_cluster[0]))
            total += len(idx_in_cluster[0])

        cluser_q_weight = []
        for i in range(len(cluser_q_cnt)):
            weight = cluser_q_cnt[self.num_clusters_q - 1] / cluser_q_cnt[i]
            cluser_q_weight.append(weight)

        return cluser_x_weight, cluser_q_weight


def read_pairs_file(dataset_path, labels_file):
    df = pd.read_csv(labels_file)
    cluster_q = df['cluster_q'].values
    cluster_x = df['cluster_x'].values
    imgs_paths1 = [join(dataset_path, path) for path in df['img_path0'].values]
    imgs_paths2 = [join(dataset_path, path) for path in df['img_path1'].values]
    scenes = df['scene_a'].values
    scene_unique_names = np.unique(scenes)
    scene_name_to_id = dict(zip(scene_unique_names, list(range(len(scene_unique_names)))))
    scenes_ids = [scene_name_to_id[s] for s in scenes]
    n = df.shape[0]
    poses = np.zeros((n, 7))
    poses[:, 0] = df['x1_ab'].values
    poses[:, 1] = df['x2_ab'].values
    poses[:, 2] = df['x3_ab'].values
    poses[:, 3] = df['q1_ab'].values
    poses[:, 4] = df['q2_ab'].values
    poses[:, 5] = df['q3_ab'].values
    poses[:, 6] = df['q4_ab'].values
    return imgs_paths1, imgs_paths2, poses, scenes, scenes_ids, cluster_q, cluster_x