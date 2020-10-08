import os
import os.path as osp
from itertools import repeat, product
import numpy as np
import h5py
import torch
import random
import glob
from plyfile import PlyData, PlyElement
from torch_geometric.data import InMemoryDataset, Data, extract_zip, Dataset
from torch_geometric.data.dataset import files_exist
from torch_geometric.data import DataLoader
from torch_geometric.datasets import S3DIS as S3DIS1x1
import torch_geometric.transforms as T
import logging
from sklearn.neighbors import NearestNeighbors, KDTree
from tqdm.auto import tqdm as tq
import csv
import pandas as pd
import pickle
import gdown
import shutil
import json
from PIL import Image

from torch_points3d.datasets.samplers import BalancedRandomSampler
import torch_points3d.core.data_transform as cT
from torch_points3d.datasets.base_dataset import BaseDataset

from torch_points3d.datasets.segmentation.s3dis import *
from torch_points3d.projection.projection import compute_index_map

DIR = osp.dirname(osp.realpath(__file__))
log = logging.getLogger(__name__)



################################### UTILS #######################################


def read_s3dis_pose(json_file):
    
    # Area 5b poses need a special treatment
    # Need to see the file comes from Area i in the provided filepath
    area_5b = 'area_5b' in json_file.lower()
    
    # Loading the Stanford pose json file
    with open(json_file) as f:
        pose_data = json.load(f)
    
    # XYZ camera position
    xyz = np.array(pose_data['camera_location'])
    
    # Omega, Phi, Kappa camera pose
    # We define a different pose coordinate system 
    omega, phi, kappa = [np.double(i) for i in pose_data['final_camera_rotation']]
#     opk = np.array([omega - (np.pi / 2), -phi, -kappa])
    opk = np.array([omega - (np.pi / 2), -phi, -kappa - (np.pi / 2)])

    # Area 5b poses require some rotation and offset corrections
    if area_5b:
        M = np.array([[0, 1, 0],
                      [-1, 0, 0],
                      [0, 0, 1]])
        xyz = M.dot(xyz) + np.array([-4.10, 6.25, 0.0])
        angles = opk + np.array([0, 0, np.pi/2])
    
    return xyz, opk


def non_static_pixels_mask(img_list, img_size=(1024,512), sample=5):
    """Find the mask of identical pixels accross a list of images."""
    
    # Iteratively update the mask w.r.t. a reference image
    mask = np.ones(img_size, dtype='bool')
    img_1 = np.array(Image.open(img_list[0]).convert('RGB').resize(img_size, Image.LANCZOS))
        
    for img_path in np.random.choice(img_list[1:], size=min(sample, len(img_list)-1), replace=False):
        img_2 = np.array(Image.open(img_path).convert('RGB').resize(img_size, Image.LANCZOS))
        
        mask_equal = np.all(img_1 == img_2, axis=2).transpose()
        mask[np.logical_and(mask, mask_equal)] = 0
    
    return mask 


def image_room(image_path):
    return '_'.join(osp.basename(image_path).split('_')[2:4])



################################### Used for fused s3dis radius sphere ###################################


class S3DISOriginalFusedMultimodal(S3DISOriginalFused):
    """ Original S3DIS dataset. Each area is loaded individually and can be processed using a pre_collate transform. 
    This transform can be used for example to fuse the area into a single space and split it into 
    spheres or smaller regions. If no fusion is applied, each element in the dataset is a single room by default.

    http://buildingparser.stanford.edu/dataset.html

    Parameters
    ----------
    root: str
        path to the directory where the data will be saved
    test_area: int
        number between 1 and 6 that denotes the area used for testing
    split: str
        can be one of train, trainval, val or test
    pre_collate_transform:
        Transforms to be applied before the data is assembled into samples (apply fusing here for example)
    keep_instance: bool
        set to True if you wish to keep instance data
    pre_transform
    transform
    pre_filter
    """

    form_url = "https://docs.google.com/forms/d/e/1FAIpQLScDimvNMCGhy_rmBA2gHfDu3naktRm6A8BPwAWWDv-Uhm6Shw/viewform?c=0&w=1"
    download_url = "https://drive.google.com/uc?id=0BweDykwS9vIobkVPN0wzRzFwTDg&export=download"
    zip_name = "Stanford3dDataset_v1.2_Version.zip"
    path_file = osp.join(DIR, "s3dis.patch")
    file_name = "Stanford3dDataset_v1.2"
    folders = ["Area_{}".format(i) for i in range(1, 7)]
    num_classes = S3DIS_NUM_CLASSES

    def __init__(
        self,
        root,
        test_area=6,
        split="train",
        transform=None,
        pre_transform=None,
        pre_collate_transform=None,
        pre_filter=None,
        multimodal_mapping=None,
        keep_instance=False,
        verbose=False,
        debug=False,
    ):
        assert test_area in list(range(1,7))
        self.transform = transform
        self.pre_collate_transform = pre_collate_transform
        self.multimodal_mapping = multimodal_mapping
        self.test_area = test_area
        self.keep_instance = keep_instance
        self.verbose = verbose
        self.debug = debug
        self._split = split
        super(S3DISOriginalFusedMultimodal, self).__init__(
            root, transform, pre_transform, pre_filter)
        if split == "train":
            path = self.processed_paths[0]
        elif split == "val":
            path = self.processed_paths[1]
        elif split == "test":
            path = self.processed_paths[2]
        elif split == "trainval":
            path = self.processed_paths[3]
        else:
            raise ValueError(
                (f"Split {split} found, but expected either " "train, val, trainval or test"))
        self._load_data(path)

        if split == "test":
            self.raw_test_data = torch.load(
                self.raw_areas_paths[test_area - 1])

    @property
    def pre_processed_path(self):
        pre_processed_file_names = "preprocessed.pt"
        return osp.join(self.processed_dir, pre_processed_file_names)

    @property
    def processed_file_names(self):
        test_area = self.test_area
        return (
            ["{}_{}.pt".format(s, test_area)
             for s in ["train", "val", "test", "trainval"]]
            + self.raw_areas_paths
            + [self.pre_processed_path]
        )

    def download(self):
        raw_folders = os.listdir(self.raw_dir)
        if len(raw_folders) == 0:
            if not osp.exists(osp.join(self.root, self.zip_name)):
        ########################################################################
        # Here download and unzip images
        ########################################################################
                log.info("WARNING: You are downloading S3DIS dataset")
                log.info(
                    "Please, register yourself by filling up the form at {}".format(
                        self.form_url)
                )
                log.info("***")
                log.info(
                    "Press any key to continue, or CTRL-C to exit. By continuing, you confirm filling up the form.")
                input("")
                gdown.download(self.download_url, osp.join(self.root, self.zip_name), quiet=False)
            extract_zip(osp.join(self.root, self.zip_name), self.root)
            shutil.rmtree(self.raw_dir)
            os.rename(osp.join(self.root, self.file_name), self.raw_dir)   
            shutil.copy(self.path_file, self.raw_dir)
            cmd = "patch -ruN -p0 -d  {} < {}".format(self.raw_dir, osp.join(self.raw_dir, "s3dis.patch"))
            os.system(cmd) 
        else:
            intersection = len(set(self.folders).intersection(set(raw_folders)))
            if intersection != 6:
                shutil.rmtree(self.raw_dir)
                os.makedirs(self.raw_dir)
                self.download()      

    def process(self):
        if not osp.exists(self.pre_processed_path):
            train_areas = [f for f in self.folders if str(self.test_area) not in f]
            test_areas = [f for f in self.folders if str(self.test_area) in f]

            train_files = [
                (f, room_name, osp.join(self.raw_dir, f, room_name))
                for f in train_areas
                for room_name in os.listdir(osp.join(self.raw_dir, f))
                if osp.isdir(osp.join(self.raw_dir, f, room_name))
            ]

            train_images = #############################

            test_files = [
                (f, room_name, osp.join(self.raw_dir, f, room_name))
                for f in test_areas
                for room_name in os.listdir(osp.join(self.raw_dir, f))
                if osp.isdir(osp.join(self.raw_dir, f, room_name))
            ]

            test_images = #############################

            # Gather data per area
            data_list = [[] for _ in range(6)]
            if self.debug:
                areas = np.zeros(7)
            for (area, room_name, file_path) in tq(train_files + test_files):
                if self.debug:
                    area_idx = int(area.split('_')[-1])
                    if areas[area_idx] == 5:
                        continue
                    else:
                        print(area_idx)
                        areas[area_idx] += 1
                
                area_num = int(area[-1]) - 1
                if self.debug:
                    read_s3dis_format(
                        file_path, room_name, label_out=True, verbose=self.verbose, debug=self.debug)
                    continue
                else:
                    xyz, rgb, room_labels, instance_labels, room_label = read_s3dis_format(
                        file_path, room_name, label_out=True, verbose=self.verbose, debug=self.debug
                    )

                    rgb_norm = rgb.float() / 255.0
                    data = Data(pos=xyz, y=room_labels, room_label=room_label, rgb=rgb_norm)
                    if room_name in VALIDATION_ROOMS:
                        data.validation_set = True
                    else:
                        data.validation_set = False

                    if self.keep_instance:
                        data.instance_labels = instance_labels

                    if self.pre_filter is not None and not self.pre_filter(data):
                        continue

                    data_list[area_num].append(data)

            raw_areas = cT.PointCloudFusion()(data_list)
            for i, area in enumerate(raw_areas):
                torch.save(area, self.raw_areas_paths[i])

            for area_datas in data_list:
                # Apply pre_transform
                if self.pre_transform is not None:
                    for data in area_datas:
                        data = self.pre_transform(data)
            torch.save(data_list, self.pre_processed_path)
        else:
            data_list = torch.load(self.pre_processed_path)

        if self.debug:
            return

        train_data_list = {}
        val_data_list = {}
        trainval_data_list = {}
        for i in range(6):
            if i != self.test_area - 1:
                train_data_list[i] = []
                val_data_list[i] = []
                for data in data_list[i]:
                    validation_set = data.validation_set
                    del data.validation_set
                    if validation_set:
                        val_data_list[i].append(data)
                    else:
                        train_data_list[i].append(data)
                trainval_data_list[i] = val_data_list[i] + train_data_list[i]

        train_data_list = list(train_data_list.values())
        val_data_list = list(val_data_list.values())
        trainval_data_list = list(trainval_data_list.values())
        test_data_list = data_list[self.test_area - 1]

        if self.pre_collate_transform:
            log.info("pre_collate_transform ...")
            log.info(self.pre_collate_transform)
            train_data_list = self.pre_collate_transform(train_data_list)
            val_data_list = self.pre_collate_transform(val_data_list)
            test_data_list = self.pre_collate_transform(test_data_list)
            trainval_data_list = self.pre_collate_transform(trainval_data_list)

        ######################################################################## 
        # MAPPING COMPUTATION HERE
        self.multimodal_mapping()
        ########################################################################

        self._save_data(train_data_list, val_data_list,
                        test_data_list, trainval_data_list)

    def _save_data(self, train_data_list, val_data_list, test_data_list, trainval_data_list):
        torch.save(self.collate(train_data_list), self.processed_paths[0])
        torch.save(self.collate(val_data_list), self.processed_paths[1])
        torch.save(self.collate(test_data_list), self.processed_paths[2])
        torch.save(self.collate(trainval_data_list), self.processed_paths[3])

    def _load_data(self, path):
        self.data, self.slices = torch.load(path)



class S3DISSphereMultimodal(S3DISOriginalFusedMultimodal):
    """ Small variation of S3DISOriginalFusedMultimodal that allows random sampling of spheres 
    within an Area during training and validation. Spheres have a radius of 2m. If sample_per_epoch
    is not specified, spheres are taken on a 2m grid.

    http://buildingparser.stanford.edu/dataset.html

    Parameters
    ----------
    root: str
        path to the directory where the data will be saved
    test_area: int
        number between 1 and 6 that denotes the area used for testing
    train: bool
        Is this a train split or not
    pre_collate_transform:
        Transforms to be applied before the data is assembled into samples (apply fusing here for example)
    keep_instance: bool
        set to True if you wish to keep instance data
    sample_per_epoch
        Number of spheres that are randomly sampled at each epoch (-1 for fixed grid)
    radius
        radius of each sphere
    pre_transform
    transform
    pre_filter
    """

    def __init__(self, root, sample_per_epoch=100, radius=2, *args, **kwargs):
        self._sample_per_epoch = sample_per_epoch
        self._radius = radius
        self._grid_sphere_sampling = cT.GridSampling3D(size=radius / 10.0)
        super().__init__(root, *args, **kwargs)

    def __len__(self):
        if self._sample_per_epoch > 0:
            return self._sample_per_epoch
        else:
            return len(self._test_spheres)

    def get(self, idx):
        if self._sample_per_epoch > 0:
            return self._get_random()
        else:
            return self._test_spheres[idx].clone()

    def process(self):  # We have to include this method, otherwise the parent class skips processing
        super().process()

    def download(self):  # We have to include this method, otherwise the parent class skips download
        super().download()

    def _get_random(self):
        # Random spheres biased towards getting more low frequency classes
        chosen_label = np.random.choice(self._labels, p=self._label_counts)
        valid_centres = self._centres_for_sampling[self._centres_for_sampling[:, 4] == chosen_label]
        centre_idx = int(random.random() * (valid_centres.shape[0] - 1))
        centre = valid_centres[centre_idx]
        area_data = self._datas[centre[3].int()]
        sphere_sampler = cT.SphereSampling(
            self._radius, centre[:3], align_origin=False)
        return sphere_sampler(area_data)

    def _save_data(self, train_data_list, val_data_list, test_data_list, trainval_data_list):
        torch.save(train_data_list, self.processed_paths[0])
        torch.save(val_data_list, self.processed_paths[1])
        torch.save(test_data_list, self.processed_paths[2])
        torch.save(trainval_data_list, self.processed_paths[3])

    def _load_data(self, path):
        self._datas = torch.load(path)
        if not isinstance(self._datas, list):
            self._datas = [self._datas]
        if self._sample_per_epoch > 0:
            self._centres_for_sampling = []
            for i, data in enumerate(self._datas):
                assert not hasattr(data, cT.SphereSampling.KDTREE_KEY)  # Just to make we don't have some out of date data in there
                low_res = self._grid_sphere_sampling(data.clone())
                centres = torch.empty((low_res.pos.shape[0], 5), dtype=torch.float)
                centres[:, :3] = low_res.pos
                centres[:, 3] = i
                centres[:, 4] = low_res.y
                self._centres_for_sampling.append(centres)
                tree = KDTree(np.asarray(data.pos), leaf_size=10)
                setattr(data, cT.SphereSampling.KDTREE_KEY, tree)

            self._centres_for_sampling = torch.cat(self._centres_for_sampling, 0)
            uni, uni_counts = np.unique(np.asarray(self._centres_for_sampling[:, -1]),
                return_counts=True)
            uni_counts = np.sqrt(uni_counts.mean() / uni_counts)
            self._label_counts = uni_counts / np.sum(uni_counts)
            self._labels = uni
        else:
            grid_sampler = cT.GridSphereSampling(self._radius, self._radius, center=False)
            self._test_spheres = grid_sampler(self._datas)



class S3DISFusedMultimodalDataset(BaseDataset):
    """ Wrapper around S3DISSphereMultimodal that creates train and test datasets.

    http://buildingparser.stanford.edu/dataset.html

    Parameters
    ----------
    dataset_opt: omegaconf.DictConfig
        Config dictionary that should contain

            - dataroot
            - fold: test_area parameter
            - pre_collate_transform
            - train_transforms
            - test_transforms
    """

    INV_OBJECT_LABEL = INV_OBJECT_LABEL

    def __init__(self, dataset_opt):
        super().__init__(dataset_opt)

        sampling_format = dataset_opt.get('sampling_format', 'sphere')
        assert sampling_format == 'sphere', f"Sampling format '{sampling_format}' is not supported"

        self.train_dataset = S3DISSphereMultimodal(
            self._data_path,
            sample_per_epoch=3000,
            test_area=self.dataset_opt.fold,
            split="train",
            pre_collate_transform=self.pre_collate_transform,
            transform=self.train_transform,
            multimodal_mapping=self.dataset_opt.multimodal_mapping,
        )

        self.val_dataset = S3DISSphereMultimodal(
            self._data_path,
            sample_per_epoch=-1,
            test_area=self.dataset_opt.fold,
            split="val",
            pre_collate_transform=self.pre_collate_transform,
            transform=self.val_transform,
            multimodal_mapping=self.dataset_opt.multimodal_mapping,
        )
        self.test_dataset = S3DISSphereMultimodal(
            self._data_path,
            sample_per_epoch=-1,
            test_area=self.dataset_opt.fold,
            split="test",
            pre_collate_transform=self.pre_collate_transform,
            transform=self.test_transform,
            multimodal_mapping=self.dataset_opt.multimodal_mapping,
        )

        if dataset_opt.class_weight_method:
            self.train_dataset = add_weights(
                self.train_dataset, True, dataset_opt.class_weight_method)

    @property
    def test_data(self):
        return self.test_dataset[0].raw_test_data

    @staticmethod
    def to_ply(pos, label, file):
        """ Allows to save s3dis predictions to disk using s3dis color scheme

        Parameters
        ----------
        pos : torch.Tensor
            tensor that contains the positions of the points
        label : torch.Tensor
            predicted label
        file : string
            Save location
        """
        to_ply(pos, label, file)

    def get_tracker(self, wandb_log: bool, tensorboard_log: bool):
        """Factory method for the tracker

        Arguments:
            wandb_log - Log using weight and biases
            tensorboard_log - Log using tensorboard
        Returns:
            [BaseTracker] -- tracker
        """
        from torch_points3d.metrics.s3dis_tracker import S3DISTracker

        return S3DISTracker(self, wandb_log=wandb_log, use_tensorboard=tensorboard_log)
