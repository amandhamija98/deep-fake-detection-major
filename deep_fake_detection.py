from __future__ import division, print_function
import sys
from copy import deepcopy
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import os
from tqdm import tqdm
import numpy as np
import random
import cv2
import copy
import cyclegan_data
import torch.nn as nn

from torchvision import transforms, models

dataset_names = ['zebra']
BATCH_SIZE = 16
TEST_BATCH_SIZE = 16
LR = 0.01
WD = 1e-4
EPOCHS = 20
START_EPOCH = 0
ENABLE_LOGGING = True
SEED = 5
model_dir = './models2/'
log_dir = './logs'

torch.manual_seed(SEED)
np.random.seed(SEED)

class gan_data(cyclegan_data.cyclegan_data):
    
    def __init__(self,train = True):
        super(gan_data,self).__init__()
        self.batch_size = BATCH_SIZE
        self.train = train
    
    def __getitem__(self, index):
        img = self.data[index]
        label = self.labels[index]

        if self.train:
            im = deepcopy(img.numpy()[16:240,16:240,:])
        else:
            im = deepcopy(img.numpy()[16:240,16:240,:])
        
        #fft

        im = im.astype(np.float32)
        im = im/255.0

        for i in range(3):
            img = im[:,:,i]
            fft_img = np.fft.fft2(img)
            fft_img = np.log(np.abs(fft_img)+1e-3)
            fft_min = np.percentile(fft_img,5)
            fft_max = np.percentile(fft_img,95)
            fft_img = (fft_img - fft_min)/(fft_max - fft_min)
            fft_img = (fft_img-0.5)*2
            fft_img[fft_img<-1] = -1
            fft_img[fft_img>1] = 1

            #take the whole band
            im[:,:,i] = fft_img

        im = np.transpose(im, (2,0,1))
        return (im, label)
    
    def __len__(self):
        return self.labels.size(0)


def create_loaders():
    test_dataset_names = copy.copy(dataset_names)
    kwargs = {}
    
    train_loader = torch.utils.data.DataLoader(gan_data(train=True),batch_size=BATCH_SIZE, shuffle=True, **kwargs)

    test_loaders = [{'name': name,
                     'dataloader': torch.utils.data.DataLoader(
                    gan_data(train=False), batch_size=TEST_BATCH_SIZE,shuffle=False, **kwargs)}
                    for name in test_dataset_names]
    
    return train_loader, test_loaders

def train(train_loader, model, optimizer, criterion,  epoch, logger):
    # switch to train mode
    model.train()
    pbar = tqdm(enumerate(train_loader))
    for batch_idx, data in pbar:
        image_pair, label = data

        # if args.cuda:
        #     image_pair, label  = image_pair.cuda(), label.cuda()
        #     image_pair, label = Variable(image_pair), Variable(label)
        #     out= model(image_pair)

        out= model(image_pair)
        loss = criterion(out, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    adjust_learning_rate(optimizer)

    # if (args.enable_logging):
    logger.log_value('loss', loss.data.item()).step()

    try:
        os.stat('{}'.format(model_dir))
    except:
        os.makedirs('{}'.format(model_dir))

    if ((epoch+1)%10)==0:
        torch.save({'epoch': epoch, 'state_dict': model.state_dict()},
               '{}/checkpoint_{}.pth'.format(model_dir, epoch+1))

def test(test_loader, model, epoch, logger, logger_test_name):
    # switch to evaluate mode, caculate test accuracy
    model.eval()

    labels, predicts = [], []
    outputs = []
    pbar = tqdm(enumerate(test_loader))
    for batch_idx, (image_pair, label) in pbar:
        # if args.cuda:
        #     image_pair = image_pair.cuda()
        with torch.no_grad():
            image_pair, label = Variable(image_pair), Variable(label)
        out = model(image_pair)
        _, pred = torch.max(out,1)
        ll = label.data.cpu().numpy().reshape(-1, 1)
        pred = pred.data.cpu().numpy().reshape(-1, 1)
        out = out.data.cpu().numpy().reshape(-1, 2)
        labels.append(ll)
        predicts.append(pred)
        outputs.append(out)

    num_tests = test_loader.dataset.labels.size(0)
    labels = np.vstack(labels).reshape(num_tests)
    predicts = np.vstack(predicts).reshape(num_tests)
    outputs = np.vstack(outputs).reshape(num_tests,2)

    print('\33[91mTest set: {}\n\33[0m'.format(logger_test_name))

    acc = np.sum(labels == predicts)/float(num_tests)
    print('\33[91mTest set: Accuracy: {:.8f}\n\33[0m'.format(acc))
    
    # if (args.enable_logging):
    logger.log_value(logger_test_name+' Acc', acc)
    return

def adjust_learning_rate(optimizer):
    """Updates the learning rate given the learning rate decay.
    The routine has been implemented according to the original Lua SGD optimizer
    """
    for group in optimizer.param_groups:
        if 'step' not in group:
            group['step'] = 0.
        else:
            group['step'] += 1.
        #group['lr'] = args.lr*((1-args.lr_decay)**group['step'])
        group['lr'] = LR
        
    return

def create_optimizer(model, new_lr):
    # setup optimizer

    optimizer = optim.SGD(model.parameters(), lr=new_lr,
                              momentum=0.9, dampening=0.9,
                              weight_decay=WD)
    # if args.optimizer == 'sgd':
        # optimizer = optim.SGD(model.parameters(), lr=new_lr,
        #                       momentum=0.9, dampening=0.9,
        #                       weight_decay=args.wd)
    # elif args.optimizer == 'adam':
    #     optimizer = optim.Adam(model.parameters(), lr=new_lr,
    #                            weight_decay=args.wd)
    # else:
    #     raise Exception('Not supported optimizer: {0}'.format(args.optimizer))
    return optimizer


def main(train_loader, test_loaders, model, logger):
    #print('\nparsed options:\n{}\n'.format(vars(args)))

    optimizer1 = create_optimizer(model, LR)
    criterion = nn.CrossEntropyLoss()
    # if args.cuda:
    #     model.cuda()
    #     criterion.cuda()

    # optionally resume from a checkpoint
    # if args.resume:
    #     if os.path.isfile(args.resume):
    #         print('=> loading checkpoint {}'.format(args.resume))
    #         checkpoint = torch.load(args.resume)
    #         args.start_epoch = checkpoint['epoch']
    #         checkpoint = torch.load(args.resume)
    #         model.load_state_dict(checkpoint['state_dict'])
    #     else:
    #         print('=> no checkpoint found at {}'.format(args.resume))
            
    start = START_EPOCH
    end = EPOCHS
    for test_loader in test_loaders:
        test(test_loader['dataloader'], model, 0, logger, test_loader['name'])
    for epoch in range(start, end):
        # iterate over test loaders and test results
        train(train_loader, model, optimizer1, criterion, epoch, logger)
        if ((epoch+1)%5)==0:
            for test_loader in test_loaders:
                test(test_loader['dataloader'], model, epoch+1, logger, test_loader['name'])

if __name__ == '__main__':
    LOG_DIR = log_dir
    if not os.path.isdir(LOG_DIR):
        os.makedirs(LOG_DIR)
    # LOG_DIR = log_dir + suffix
    logger, file_logger = None, None

    # pretrain_flag = not args.feature=='comatrix'

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

    if(ENABLE_LOGGING):
        from Loggers import Logger
        logger = Logger(LOG_DIR)
    train_loader, test_loaders = create_loaders()
    main(train_loader, test_loaders, model, logger)