import random
from skimage.io import imread
from torch.utils.data import Dataset
import pandas as pd
from os.path import join
import numpy as np
from collections import defaultdict
import joblib

class RelPoseDataset(Dataset):
    def __init__(self, data_path, pairs_file, kmeans_position_file, kmeans_orientation_file, transform=None, num_position_clusters=4, num_orientation_clusters=4):
        self.img_path1, self.scenes1, self.scene_ids1, self.poses1, \
        self.img_path2, self.scenes2, self.scene_ids2, self.poses2 = \
            read_pairs_file(data_path, pairs_file)
        self.transform = transform
        self.num_scenes = np.max(self.scene_ids1) + 1
        self.dataset_size = self.poses1.shape[0]
        # Generate clusters for each scene
        self.position_centroids = {}
        self.position_cluster_ids = np.zeros(self.dataset_size)
        self.orientation_centroids = {}
        self.orientation_cluster_ids = np.zeros(self.dataset_size)

        #kmeans_position = joblib.load(kmeans_position_file)
        #kmeans_orientation = joblib.load(kmeans_orientation_file)

        #self.position_centroids = kmeans_position.cluster_centers_.astype(np.float32)
        #self.orientation_centroids = kmeans_orientation.cluster_centers_.astype(np.float32)

    def __len__(self):
        return len(self.img_path1)

    def __getitem__(self, idx):
        img1 = imread(self.img_path1[idx])
        img2 = imread(self.img_path2[idx])
        pose1 = self.poses1[idx]
        pose2 = self.poses2[idx]
        scene1 = self.scene_ids1[idx]
        scene2 = self.scene_ids2[idx]
        #position_cluster_id = int(self.position_cluster_ids[idx])
        #orientation_cluster_id = int(self.orientation_cluster_ids[idx])
        position_centroids = 0 #self.position_centroids#[scene1]
        orientation_centroids = 0 #self.orientation_centroids#[scene1]

        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        sample = {'img1': img1, 'img2': img2, 'pose1': pose1, 'pose2': pose2, 'scene': scene1,
                  'position_centroids': position_centroids, 'orientation_centroids': orientation_centroids,
                  }
        return sample

def read_pairs_file(dataset_path, labels_file):
    df = pd.read_csv(labels_file)
    img_paths = []
    scenes = []
    scene_ids = []
    all_poses = []
    n = df.shape[0]
    scenes_dict = defaultdict(str)
    for i, scene in enumerate(['chess', 'fire', 'heads', 'office', 'pumpkin', 'redkitchen', 'stairs']):
        scenes_dict[i] = scene
    for suffix in ["a", "b"]:
        scenes.append(df['scene_{}'.format(suffix)].values)
        scene_ids.append( df['scene_id_{}'.format(suffix)].values)
        #img_paths.append([join(dataset_path, path) for path in df['img_path_{}'.format(suffix)].values])
        poses = np.zeros((n, 7))
        poses[:, 0] = df['x1_{}'.format(suffix)].values
        poses[:, 1] = df['x2_{}'.format(suffix)].values
        poses[:, 2] = df['x3_{}'.format(suffix)].values
        poses[:, 3] = df['q1_{}'.format(suffix)].values
        poses[:, 4] = df['q2_{}'.format(suffix)].values
        poses[:, 5] = df['q3_{}'.format(suffix)].values
        poses[:, 6] = df['q4_{}'.format(suffix)].values
        all_poses.append(poses)

    #img_paths1, img_paths2 = img_paths
    scenes1, scenes2 = scenes
    scene_ids1, scene_ids2 = scene_ids
    poses1, poses2 = all_poses
    img_paths1 = []
    img_paths2 = []
    for i in range(len(df['img_path_a'])):
        img_paths1.append(join(dataset_path, scenes1[i] + df['img_path_a'][i]))
        img_paths2.append(join(dataset_path, scenes2[i] + df['img_path_b'][i]))

    return img_paths1, scenes1, scene_ids1, poses1, img_paths2, scenes2, scene_ids2, poses2