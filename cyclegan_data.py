import os
import errno
import numpy as np
from PIL import Image
import torch
import torch.utils.data as data
import collections
from tqdm import tqdm
import random
import glob
import cv2

class cyclegan_data(data.Dataset):
    def __init__(self):
        self.image_dir = './datasets'
        self.data = None
        self.labels = None
        data_file = 'zebra_train.pt'

        self.cache_data(data_file)
        data, labels = torch.load(data_file)

        self.data = data
        self.labels = labels

        self.data = torch.ByteTensor(self.data)
        self.labels = torch.LongTensor(self.labels)

    def __getitem__(self, index):
        return self.data[index], self.labels[index]

    def __len__(self):
        return self.data.shape[0]
    
    def _check_datafile_exists(self,data_file):
        return os.path.exists(data_file)
    
    def cache_data(self,data_file):
        if self._check_datafile_exists(data_file):
            print("# Found data file")
            return
        
        print("# Caching data")
        dataset = (
            self.read_image_file()
        )
        with open(data_file, 'wb') as f:
            torch.save(dataset, f)


    def read_image_file(self):
        """Return a Tensor containing the patches
        """
        image_list = []
        filename_list = []
        label_list = []
        #load all possible jpg or png images
        search_str = './datasets/real/zebra/testB/*.jpg'

        # real images
        search_str = './datasets/real/zebra/testB/*.jpg'

        for filename in glob.glob(search_str):
            image = cv2.imread(filename)
            if image.shape[0]!=256:
                image = cv2.resize(image, (256,256))
            image_list.append(image)
            label_list.append(1)
        
        # fake images
        search_str = './datasets/fake/zebra/testB/*.jpg'

        for filename in glob.glob(search_str):
            image = cv2.imread(filename)
            if image.shape[0]!=256:
                image = cv2.resize(image, (256,256)) 
            image_list.append(image)
            label_list.append(0)

        return np.array(image_list), np.array(label_list)
