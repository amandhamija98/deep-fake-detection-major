import sys
from copy import deepcopy
import argparse
import torch
import torch.nn as nn
import torchvision.datasets as dset
from torch.autograd import Variable
import os
from tqdm import tqdm
import numpy as np
import random
import cv2
import copy
from deep_fake_detection import gan_data
from collections import OrderedDict
import csv
from torchvision import transforms, models

dataset_names = ['orange']

SEED = 5
TEST_BATCH_SIZE = 16
EPOCHS = 20
START_EPOCH = 1
result_dir = './orange_test_results'
model_dir = './models2'

torch.manual_seed(SEED)
np.random.seed(SEED)

try:
    os.stat('{}/'.format(result_dir))
except:
    os.makedirs('{}/'.format(result_dir))

def create_loaders():

    test_dataset_names = copy.copy(dataset_names)

    kwargs = {}

    print(test_dataset_names) 
    test_loaders = [{'name': name,
                     'dataloader': torch.utils.data.DataLoader(
             gan_data(train=False),
                        batch_size=TEST_BATCH_SIZE,
                        shuffle=False, **kwargs)}
                    for name in test_dataset_names]

    return test_loaders

def test(test_loader, model, epoch, logger_test_name):
    # switch to evaluate mode
    model.eval()

    labels, predicts = [], []

    pbar = tqdm(enumerate(test_loader))
    for batch_idx, (image_pair, label) in pbar:

        with torch.no_grad():
            image_pair, label = Variable(image_pair), Variable(label)

        out = model(image_pair)
        _, pred = torch.max(out,1)
        ll = label.data.cpu().numpy().reshape(-1, 1)
        pred = pred.data.cpu().numpy().reshape(-1, 1)
        labels.append(ll)
        predicts.append(pred)

    num_tests = test_loader.dataset.labels.size(0)
    labels = np.vstack(labels).reshape(num_tests)
    predicts = np.vstack(predicts).reshape(num_tests)
    
    print('\33[91mTest set: {}\n\33[0m'.format(logger_test_name))

    acc = np.sum(labels == predicts)/float(num_tests)
    print('\33[91mTest set: Accuracy: {:.8f}\n\33[0m'.format(acc))
    
    pos_label = labels[labels==1]
    pos_pred = predicts[labels==1]
    TPR = np.sum(pos_label == pos_pred)/float(pos_label.shape[0])
    print('\33[91mTest set: TPR: {:.8f}\n\33[0m'.format(TPR))

    neg_label = labels[labels==0]
    neg_pred = predicts[labels==0]
    TNR = np.sum(neg_label == neg_pred)/float(neg_label.shape[0])
    print('\33[91mTest set: TNR: {:.8f}\n\33[0m'.format(TNR))

    return acc

def main(test_loaders, model):
    acc_list = []
    
    csv_file = csv.writer(open('{}/{}.csv'.format(result_dir, 'standard'), 'w'), delimiter=',')
    csv_file.writerow(dataset_names) 
    # if not args.leave_one_out:
    #     csv_file = csv.writer(open('{}/{}.csv'.format(result_dir, suffix), 'w'), delimiter=',')
    #     csv_file.writerow(dataset_names) 
    # else:
    #     result_dict = OrderedDict()
    #     try:
    #         read_result_dict = load_csv('{}/leave_one_out.csv'.format(args.result_dir),',')
    #         read_result_dict = read_result_dict.to_dict()
    #         for key in read_result_dict:
    #             result_dict[key] = read_result_dict[key][0]
    #     except Exception as e:
    #         print(str(e))
    #     if result_dict is None:
    #         result_dict = OrderedDict()

    start = START_EPOCH
    end = start + EPOCHS
    for test_loader in test_loaders:
        acc = test(test_loader['dataloader'], model, 0, test_loader['name'])*100
        acc_list.append(str(acc))
        # if args.leave_one_out:
        #     result_dict[test_loader['name']] = acc
    
    #write csv file

    csv_file.writerow(acc_list) 

    # if not args.leave_one_out:
    #     csv_file.writerow(acc_list) 
    # else:
    #     csv_file = csv.writer(open('{}/leave_one_out.csv'.format(args.result_dir), 'w'), delimiter=',')
    #     name_list = []
    #     acc_list = []
    #     for key in result_dict:
    #         name_list.append(key)
    #         acc_list.append(result_dict[key])
    #     csv_file.writerow(name_list)  
    #     csv_file.writerow(acc_list) 

if __name__ == '__main__':

    model = models.resnet34(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)
    # if args.model == 'resnet':
    #     model = models.resnet34(pretrained=True)
    #     num_ftrs = model.fc.in_features
    #     model.fc = nn.Linear(num_ftrs, 2)
    # elif args.model == 'pggan':
    #     model = pggan_dnet.SimpleDiscriminator(3, label_size=1, mbstat_avg='all', 
    #             resolution=256, fmap_max=128, fmap_base=2048, sigmoid_at_end=False)
    # elif args.model == 'densenet':
    #     model = models.densenet121(pretrained=True)
    #     num_ftrs = model.classifier.in_features
    #     model.classifier = nn.Linear(num_ftrs, 2)

    print('{}/checkpoint_{}.pth'.format(model_dir,EPOCHS))
    load_model = torch.load('{}/checkpoint_{}.pth'.format(model_dir,EPOCHS))
    model.load_state_dict(load_model['state_dict'])

    test_loaders = create_loaders()
    main(test_loaders, model)