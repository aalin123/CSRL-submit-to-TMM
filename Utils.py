import os
import argparse
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim

import sklearn
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE

import torch.utils.data as data
#import torchvision.transforms as transforms
from torchvision import datasets, transforms

from dataset import *

from model import *
numOfAU = 17
import torch.utils.model_zoo as model_zoo
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches



class AverageMeter(object):
    '''Computes and stores the sum, count and average'''

    def __init__(self):
        self.reset()

    def reset(self):    
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, count=1):
        self.val = val
        self.sum += val 
        self.count += count
        if self.count==0:
            self.avg = 0
        else:
            self.avg = float(self.sum) / self.count


def str2bool(input):
    if isinstance(input, bool):
       return input
    if input.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif input.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')



#args, flag1='train', flag2='source'
def BulidDataloader(args, flag1='train', flag2='source'):
    """Bulid data loader."""

    assert flag1 in ['train', 'test'], 'Function BuildDataloader : function parameter flag1 wrong.'
    assert flag2 in ['source', 'target'], 'Function BuildDataloader : function parameter flag2 wrong.'

    # Set Transform

    trans = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]),
            ])
    target_trans = None


    # Basic Notes:
    # 0: Surprised
    # 1: Fear
    # 2: Disgust
    # 3: Happy
    # 4: Sad
    # 5: Angry
    # 6: Neutral

    dataPath_prefix = '/home/b3432/Code/experiment/databases/Datasets_AGRA_2020'

    data_imgs, data_labels, data_bboxs, data_landmarks = [], [], [], []
    if flag1 == 'train':
        if flag2 == 'source':
            if args.sourceDataset=='RAF': # RAF Train Set
                
                list_patition_label = pd.read_csv(dataPath_prefix+'/RAF/RAF/basic/EmoLabel/list_patition_label.txt', header=None, delim_whitespace=True)
                list_patition_label = np.array(list_patition_label)

                for index in range(list_patition_label.shape[0]):
                    if list_patition_label[index,0][:5] == "train":
                        data_imgs.append(dataPath_prefix+'/RAF/RAF/basic/Image/aligned/'+list_patition_label[index, 0].split('.')[0]+'_aligned.jpg')
                        #data_imgs.append(dataPath_prefix+'/RAF/basic/Image/original/'+list_patition_label[index,0])
                        ##append()接受一个对象参数，把对象添加到列表的尾部
                        data_labels.append(list_patition_label[index,1]-1)
                        
            elif args.sourceDataset=='AFED': # AFED Train Set

                AsiantoLabel = { 3:0, 6:1, 5:2, 1:3, 4:4, 9:5, 0:6 }
                list_patition_label = pd.read_csv(dataPath_prefix+'/Asian_Facial_Expression/AsianMovie_0725_0730/list/train_list.txt', header=None, delim_whitespace=True)
                list_patition_label = np.array(list_patition_label)

                for index in range(list_patition_label.shape[0]):

                    if list_patition_label[index,-1] not in AsiantoLabel.keys():
                        continue

                    bbox = list_patition_label[index,1:5].astype(np.int)
                    landmark = np.loadtxt(dataPath_prefix+'/Asian_Facial_Expression/AsianMovie_0725_0730/annos/landmark_5/'+list_patition_label[index,0][:-3]+'txt').astype(np.int)
                    
                    data_imgs.append(dataPath_prefix+'/Asian_Facial_Expression/AsianMovie_0725_0730/images/'+list_patition_label[index,0])
                    data_labels.append(AsiantoLabel[list_patition_label[index,-1]])
                    data_bboxs.append(bbox)  
                    data_landmarks.append(landmark)

        elif flag2 == 'target':

            if args.targetDataset=='CK+': # CK+ Train Set
                
                for index, expression in enumerate(['Surprised','Fear','Disgust','Happy','Sad','Angry','Neutral']):
                    Dirs = os.listdir(os.path.join(dataPath_prefix+'/CK/CK+_Emotion/Train/CK+_Train_crop',expression))
                    for imgFile in Dirs:
                        imgPath = os.path.join(dataPath_prefix+'/CK/CK+_Emotion/Train/CK+_Train_crop',expression,imgFile)
                    
                        data_imgs.append(imgPath)
                        data_labels.append(index)
            
            elif args.targetDataset=='CK_all': # CK+ Train Set
                
                #/home/b3432/Code/experiment/databases/Datasets_AGRA_2020/CK/CK+
                for index, expression in enumerate(['Surprised','Fear','Disgust','Happy','Sad','Angry','Neutral']):
                    Dirs = os.listdir(os.path.join(dataPath_prefix+'/CK/CK+',expression))
                    for imgFile in Dirs:
                        imgPath = os.path.join(dataPath_prefix+'/CK/CK+',expression,imgFile)
                       
                        data_imgs.append(imgPath)
                        data_labels.append(index)

            elif args.targetDataset=='JAFFE': # JAFFE Dataset

                list_patition_label = pd.read_csv(dataPath_prefix+'/JAFFE/list/list_putao.txt', header=None, delim_whitespace=True)
                list_patition_label = np.array(list_patition_label)

                for index in range(list_patition_label.shape[0]):

                    data_imgs.append(dataPath_prefix+'/JAFFE/images_crop/'+list_patition_label[index,0])
                    data_labels.append(list_patition_label[index,1])
                   
            elif args.targetDataset=='Oulu-CASIA': # Oulu-CASIA Dataset

                list_patition_label = pd.read_csv(dataPath_prefix+'/Oulu-CASIA/list/list.txt', header=None, delim_whitespace=True)
                list_patition_label = np.array(list_patition_label)

                for index in range(list_patition_label.shape[0]):

                    if not os.path.exists(dataPath_prefix+'/Oulu-CASIA/annos/landmark_5/VL_Acropped/Strong/'+list_patition_label[index,0][:-4]+'txt'): 
                        continue
                    
                    img = Image.open(dataPath_prefix+'/Oulu-CASIA/images/'+list_patition_label[index,0]).convert('RGB')
                    ori_img_w, ori_img_h = img.size

                    landmark = np.loadtxt(dataPath_prefix+'/Oulu-CASIA/annos/landmark_5/VL_Acropped/Strong/'+list_patition_label[index,0][:-4]+'txt').astype(np.int)

                    data_imgs.append(dataPath_prefix+'/Oulu-CASIA/images/'+list_patition_label[index,0])
                    data_labels.append(list_patition_label[index,1])
                    data_bboxs.append((0,0,ori_img_w,ori_img_h)) 
                    data_landmarks.append(landmark)

            elif args.targetDataset=='SFEW': # SFEW Train Set

                for index, expression in enumerate(['Surprise','Fear','Disgust','Happy','Sad','Angry','Neutral']):
                    Dirs = os.listdir(os.path.join('/home/b3432/Code/experiment/databases/Datasets_AGRA_2020/SFEW/SFEW/Train/imgs',expression))
                    for imgFile in Dirs:
                        
                        imgPath = os.path.join('/home/b3432/Code/experiment/databases/Datasets_AGRA_2020/SFEW/SFEW/Train/imgs',expression,imgFile)
                        data_imgs.append(imgPath)
                        data_labels.append(index)
                       
            elif args.targetDataset=='FER2013': # FER2013 Train Set
                for index, expression in enumerate(['1surprised','2fear','3disgust','4happy','5sad','6anger','7normal']):
                    Dirs = os.listdir(os.path.join(dataPath_prefix+'/fer2013/Training',expression))
                    for imgFile in Dirs:
                        imgPath = os.path.join(dataPath_prefix+'/fer2013/Training',expression,imgFile)
                        data_imgs.append(imgPath)
                        data_labels.append(index)
                   
            elif args.targetDataset=='ExpW': # ExpW Train Set
                
                ExpWtoLabel = { 5:0, 2:1, 1:2, 3:3, 4:4, 0:5, 6:6 }
                list_patition_label = pd.read_csv(dataPath_prefix+'/ExpW/list/Landmarks_5/train_list_5landmarks.txt', header=None, delim_whitespace=True)
                list_patition_label = np.array(list_patition_label)

                for index in range(list_patition_label.shape[0]):

                    bbox = list_patition_label[index,2:6].astype(np.int)
                    landmark = np.array(list_patition_label[index,7:]).astype(np.int).reshape(-1,2)
                    
                    data_imgs.append(dataPath_prefix+'/ExpW/data/image/origin/'+list_patition_label[index,0])
                    data_labels.append(ExpWtoLabel[list_patition_label[index,6]])
                    data_bboxs.append(bbox)
                    data_landmarks.append(landmark)
            
            elif args.targetDataset=='AFED': # AFED Train Set

                AsiantoLabel = { 3:0, 6:1, 5:2, 1:3, 4:4, 9:5, 0:6 }
                list_patition_label = pd.read_csv(dataPath_prefix+'/Asian_Facial_Expression/AsianMovie_0725_0730/list/train_list.txt', header=None, delim_whitespace=True)
                list_patition_label = np.array(list_patition_label)

                for index in range(list_patition_label.shape[0]):

                    if list_patition_label[index,-1] not in AsiantoLabel.keys():
                        continue

                    bbox = list_patition_label[index,1:5].astype(np.int)
                    landmark = np.loadtxt(dataPath_prefix+'/Asian_Facial_Expression/AsianMovie_0725_0730/annos/landmark_5/'+list_patition_label[index,0][:-3]+'txt').astype(np.int)
                    
                    data_imgs.append(dataPath_prefix+'/Asian_Facial_Expression/AsianMovie_0725_0730/images/'+list_patition_label[index,0])
                    data_labels.append(AsiantoLabel[list_patition_label[index,-1]])
                    data_bboxs.append(bbox)  
                    data_landmarks.append(landmark)

            elif args.targetDataset=='RAF': # RAF Train Set

                list_patition_label = pd.read_csv(dataPath_prefix+'/RAF/RAF/basic/EmoLabel/list_patition_label.txt', header=None, delim_whitespace=True)
                list_patition_label = np.array(list_patition_label)

                for index in range(list_patition_label.shape[0]):
                    if list_patition_label[index,0][:5] == "train":
                        data_imgs.append(dataPath_prefix+'/RAF/RAF/basic/Image/aligned/'+list_patition_label[index, 0].split('.')[0]+'_aligned.jpg')
                        #data_imgs.append(dataPath_prefix+'/RAF/basic/Image/original/'+list_patition_label[index,0])
                        ##append()接受一个对象参数，把对象添加到列表的尾部
                        data_labels.append(list_patition_label[index,1]-1)
                        
           
    elif flag1 == 'test':
        if flag2 =='source':
            if args.sourceDataset=='CK+': # CK+ Val Set

                for index, expression in enumerate(['Surprised','Fear','Disgust','Happy','Sad','Angry','Neutral']):
                    Dirs = os.listdir(os.path.join(dataPath_prefix+'/CK/CK+_Emotion/Val/CK+_Val_crop',expression))
                    for imgFile in Dirs:
                        imgPath = os.path.join(dataPath_prefix+'/CK/CK+_Emotion/Val/CK+_Val_crop',expression,imgFile)
                        
                        data_imgs.append(imgPath)
                        data_labels.append(index)
                       
            elif args.sourceDataset=='JAFFE': # JAFFE Dataset

                list_patition_label = pd.read_csv(dataPath_prefix+'/JAFFE/list/list_putao.txt', header=None, delim_whitespace=True)
                list_patition_label = np.array(list_patition_label)

                for index in range(list_patition_label.shape[0]):

                    data_imgs.append(dataPath_prefix+'/JAFFE/images_crop/'+list_patition_label[index,0])
                    data_labels.append(list_patition_label[index,1])
          
            elif args.sourceDataset=='Oulu-CASIA': # Oulu-CASIA Dataset

                list_patition_label = pd.read_csv(dataPath_prefix+'/Oulu-CASIA/list/list.txt', header=None, delim_whitespace=True)
                list_patition_label = np.array(list_patition_label)

                for index in range(list_patition_label.shape[0]):

                    if not os.path.exists(dataPath_prefix+'/Oulu-CASIA/annos/landmark_5/VL_Acropped/Strong/'+list_patition_label[index,0][:-4]+'txt'): 
                        continue
                    
                    img = Image.open(dataPath_prefix+'/Oulu-CASIA/images/'+list_patition_label[index,0]).convert('RGB')
                    ori_img_w, ori_img_h = img.size

                    landmark = np.loadtxt(dataPath_prefix+'/Oulu-CASIA/annos/landmark_5/VL_Acropped/Strong/'+list_patition_label[index,0][:-4]+'txt').astype(np.int)

                    data_imgs.append(dataPath_prefix+'/Oulu-CASIA/images/'+list_patition_label[index,0])
                    data_labels.append(list_patition_label[index,1])
                    data_bboxs.append((0,0,ori_img_w,ori_img_h)) 
                    data_landmarks.append(landmark)

            elif args.sourceDataset=='SFEW': # SFEW 2.0 Val Set

                for index, expression in enumerate(['Surprise','Fear','Disgust','Happy','Sad','Angry','Neutral']):
                    Dirs = os.listdir(os.path.join(dataPath_prefix + '/SFEW/SFEW/Val/imgs',expression))
                    for imgFile in Dirs:
                        imgPath = os.path.join(dataPath_prefix + '/SFEW/SFEW/Val/imgs',expression, imgFile)
                        data_imgs.append(imgPath)
                        data_labels.append(index)


            elif args.sourceDataset=='FER2013': # FER2013 Val Set

                  for index, expression in enumerate(['1surprised','2fear','3disgust','4happy','5sad','6anger','7normal']):
                    Dirs = os.listdir(os.path.join(dataPath_prefix+'/fer2013/PublicTest',expression))
                    for imgFile in Dirs:
                        imgPath = os.path.join(dataPath_prefix+'/fer2013/PublicTest',expression,imgFile)
                        data_imgs.append(imgPath)
                        data_labels.append(index)

            elif args.sourceDataset=='ExpW': # ExpW Val Set

                ExpWtoLabel = { 5:0, 2:1, 1:2, 3:3, 4:4, 0:5, 6:6 }
                list_patition_label = pd.read_csv(dataPath_prefix+'/ExpW/list/Landmarks_5/val_list_5landmarks.txt', header=None, delim_whitespace=True)
                list_patition_label = np.array(list_patition_label)

                for index in range(list_patition_label.shape[0]):

                    bbox = list_patition_label[index,2:6].astype(np.int)
                    landmark = np.array(list_patition_label[index,7:]).astype(np.int).reshape(-1,2)
                    
                    data_imgs.append(dataPath_prefix+'/ExpW/data/image/origin/'+list_patition_label[index,0])
                    data_labels.append(ExpWtoLabel[list_patition_label[index,6]])
                    data_bboxs.append(bbox)
                    data_landmarks.append(landmark)
           
            elif args.sourceDataset=='AFED': # AFED Val Set

                AsiantoLabel = { 3:0, 6:1, 5:2, 1:3, 4:4, 9:5, 0:6 }
                list_patition_label = pd.read_csv(dataPath_prefix+'/Asian_Facial_Expression/AsianMovie_0725_0730/list/val_list.txt', header=None, delim_whitespace=True)
                list_patition_label = np.array(list_patition_label)

                for index in range(list_patition_label.shape[0]):

                    if list_patition_label[index,-1] not in AsiantoLabel.keys():
                        continue

                    bbox = list_patition_label[index,1:5].astype(np.int)
                    landmark = np.loadtxt(dataPath_prefix+'/Asian_Facial_Expression/AsianMovie_0725_0730/annos/landmark_5/'+list_patition_label[index,0][:-3]+'txt').astype(np.int)
                    
                    data_imgs.append(dataPath_prefix+'/Asian_Facial_Expression/AsianMovie_0725_0730/images/'+list_patition_label[index,0])
                    data_labels.append(AsiantoLabel[list_patition_label[index,-1]])
                    data_bboxs.append(bbox)
                    data_landmarks.append(landmark)

            elif args.sourceDataset=='RAF': # RAF Test Set

                list_patition_label = pd.read_csv(dataPath_prefix+'/RAF/RAF/basic/EmoLabel/list_patition_label.txt', header=None, delim_whitespace=True)
                list_patition_label = np.array(list_patition_label)

                for index in range(list_patition_label.shape[0]):
                    if list_patition_label[index,0][:4] == "test":
                        data_imgs.append(dataPath_prefix+'/RAF/RAF/basic/Image/aligned/'+list_patition_label[index, 0].split('.')[0]+'_aligned.jpg')
                        data_labels.append(list_patition_label[index,1]-1)

        elif flag2=='target':

            if args.targetDataset=='CK+': # CK+ Val Set

                for index, expression in enumerate(['Surprised','Fear','Disgust','Happy','Sad','Angry','Neutral']):
                    Dirs = os.listdir(os.path.join(dataPath_prefix+'/CK/CK+_Emotion/Val/CK+_Val_crop',expression))
                    for imgFile in Dirs:
                        imgPath = os.path.join(dataPath_prefix+'/CK/CK+_Emotion/Val/CK+_Val_crop',expression,imgFile)
                        
                        data_imgs.append(imgPath)
                        data_labels.append(index)

            elif args.targetDataset=='CK_all':
                for index, expression in enumerate(['Surprised','Fear','Disgust','Happy','Sad','Angry','Neutral']):
                    Dirs = os.listdir(os.path.join(dataPath_prefix+'/CK/CK+',expression))
                    for imgFile in Dirs:
                        imgPath = os.path.join(dataPath_prefix+'/CK/CK+',expression,imgFile)
                        data_imgs.append(imgPath)
                        data_labels.append(index)

            elif args.targetDataset=='JAFFE': # JAFFE Dataset

                list_patition_label = pd.read_csv(dataPath_prefix+'/JAFFE/list/list_putao.txt', header=None, delim_whitespace=True)
                list_patition_label = np.array(list_patition_label)

                for index in range(list_patition_label.shape[0]):

                    data_imgs.append(dataPath_prefix+'/JAFFE/images_crop/'+list_patition_label[index,0])
                    data_labels.append(list_patition_label[index,1])
          
            elif args.targetDataset=='Oulu-CASIA': # Oulu-CASIA Dataset

                list_patition_label = pd.read_csv(dataPath_prefix+'/Oulu-CASIA/list/list.txt', header=None, delim_whitespace=True)
                list_patition_label = np.array(list_patition_label)

                for index in range(list_patition_label.shape[0]):

                    if not os.path.exists(dataPath_prefix+'/Oulu-CASIA/annos/landmark_5/VL_Acropped/Strong/'+list_patition_label[index,0][:-4]+'txt'): 
                        continue
                    
                    img = Image.open(dataPath_prefix+'/Oulu-CASIA/images/'+list_patition_label[index,0]).convert('RGB')
                    ori_img_w, ori_img_h = img.size

                    landmark = np.loadtxt(dataPath_prefix+'/Oulu-CASIA/annos/landmark_5/VL_Acropped/Strong/'+list_patition_label[index,0][:-4]+'txt').astype(np.int)

                    data_imgs.append(dataPath_prefix+'/Oulu-CASIA/images/'+list_patition_label[index,0])
                    data_labels.append(list_patition_label[index,1])
                    data_bboxs.append((0,0,ori_img_w,ori_img_h)) 
                    data_landmarks.append(landmark)

            elif args.targetDataset=='SFEW': # SFEW 2.0 Val Set

                 for index, expression in enumerate(['Surprise','Fear','Disgust','Happy','Sad','Angry','Neutral']):
                    Dirs = os.listdir(os.path.join(dataPath_prefix + '/SFEW/SFEW/Val/imgs',expression))
                    for imgFile in Dirs:
                        imgPath = os.path.join(dataPath_prefix + '/SFEW/SFEW/Val/imgs',expression, imgFile)
                        data_imgs.append(imgPath)
                        data_labels.append(index)

            elif args.targetDataset=='FER2013': # FER2013 Val Set

                 for index, expression in enumerate(['1surprised','2fear','3disgust','4happy','5sad','6anger','7normal']):
                    Dirs = os.listdir(os.path.join(dataPath_prefix+'/fer2013/PublicTest',expression))
                    for imgFile in Dirs:
                        imgPath = os.path.join(dataPath_prefix+'/fer2013/PublicTest',expression,imgFile)
                        data_imgs.append(imgPath)
                        data_labels.append(index)

            elif args.targetDataset=='ExpW': # ExpW Val Set

                ExpWtoLabel = { 5:0, 2:1, 1:2, 3:3, 4:4, 0:5, 6:6 }
                list_patition_label = pd.read_csv(dataPath_prefix+'/ExpW/list/Landmarks_5/val_list_5landmarks.txt', header=None, delim_whitespace=True)
                list_patition_label = np.array(list_patition_label)

                for index in range(list_patition_label.shape[0]):

                    bbox = list_patition_label[index,2:6].astype(np.int)
                    landmark = np.array(list_patition_label[index,7:]).astype(np.int).reshape(-1,2)
                    
                    data_imgs.append(dataPath_prefix+'/ExpW/data/image/origin/'+list_patition_label[index,0])
                    data_labels.append(ExpWtoLabel[list_patition_label[index,6]])
                    data_bboxs.append(bbox)
                    data_landmarks.append(landmark)
           
            elif args.targetDataset=='AFED': # AFED Val Set

                AsiantoLabel = { 3:0, 6:1, 5:2, 1:3, 4:4, 9:5, 0:6 }
                list_patition_label = pd.read_csv(dataPath_prefix+'/Asian_Facial_Expression/AsianMovie_0725_0730/list/val_list.txt', header=None, delim_whitespace=True)
                list_patition_label = np.array(list_patition_label)

                for index in range(list_patition_label.shape[0]):

                    if list_patition_label[index,-1] not in AsiantoLabel.keys():
                        continue

                    bbox = list_patition_label[index,1:5].astype(np.int)
                    landmark = np.loadtxt(dataPath_prefix+'/Asian_Facial_Expression/AsianMovie_0725_0730/annos/landmark_5/'+list_patition_label[index,0][:-3]+'txt').astype(np.int)
                    
                    data_imgs.append(dataPath_prefix+'/Asian_Facial_Expression/AsianMovie_0725_0730/images/'+list_patition_label[index,0])
                    data_labels.append(AsiantoLabel[list_patition_label[index,-1]])
                    data_bboxs.append(bbox)
                    data_landmarks.append(landmark)

            elif args.targetDataset=='RAF': # RAF Test Set

                list_patition_label = pd.read_csv(dataPath_prefix+'/RAF/RAF/basic/EmoLabel/list_patition_label.txt', header=None, delim_whitespace=True)
                list_patition_label = np.array(list_patition_label)

                for index in range(list_patition_label.shape[0]):
                    if list_patition_label[index,0][:5] == "test":
                        data_imgs.append(dataPath_prefix+'/RAF/RAF/basic/Image/aligned/'+list_patition_label[index, 0].split('.')[0]+'_aligned.jpg')
                        #data_imgs.append(dataPath_prefix+'/RAF/basic/Image/original/'+list_patition_label[index,0])
                        ##append()接受一个对象参数，把对象添加到列表的尾部
                        data_labels.append(list_patition_label[index,1]-1)


    # DataSet Distribute
    distribute_ = np.array(data_labels)
    print('The %s %s dataset quantity: %d' % ( flag1, flag2, len(data_imgs) ) )
    print('The %s %s dataset distribute: %d, %d, %d, %d, %d, %d, %d' % ( flag1, flag2,
                                                                               np.sum(distribute_==0), np.sum(distribute_==1), np.sum(distribute_==2), np.sum(distribute_==3),
                                                                               np.sum(distribute_==4), np.sum(distribute_==5), np.sum(distribute_==6) ))

    # DataSet
    data_set = MyDataset_EM_nobox(data_imgs, data_labels,  flag1, trans, target_trans)

    # DataLoader
    if flag1=='train':
        data_loader = data.DataLoader(dataset=data_set, batch_size=args.train_batch_size, shuffle=True, num_workers=args.Num_Workers, drop_last=True)
    elif flag1=='test':
        data_loader = data.DataLoader(dataset=data_set, batch_size=args.test_batch_size, shuffle=False, num_workers=args.Num_Workers, drop_last=False)

    return data_loader
   


def Bulid_Model_1(args):
    '''Bulid Model'''
    model_urls = {'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
                  'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',}

    #if args.Distribute == 'Basic':
    if  args.Backbone == 'ResNet18':
        model = ResNet18()
    elif args.Backbone == 'ResNet50':
        model = ResNet50()

    #elif args.Distribute == 'Compound':
    #    model = ResNet101_Compound(args.Dim)

    if args.Resume_Model != 'None':
        if args.Resume_Model =='imagenet':
            print('Resume Model: {}'.format(args.Resume_Model))
            if  args.Backbone == 'ResNet18':
                model.load_state_dict(model_zoo.load_url(model_urls['resnet18']),strict=False)
            elif args.Backbone == 'ResNet50':
                model.load_state_dict(model_zoo.load_url(model_urls['resnet50']),strict=False)
            # del checkpoint
            # torch.cuda.empty_cache()
            #torch.cuda.empty_cache()
        else:
            print('Resume Model: {}'.format(args.Resume_Model))
            checkpoint = torch.load(args.Resume_Model, map_location='cpu')
            model.load_state_dict(checkpoint, strict=False)
        #model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
        # Save GPU Memory
            del checkpoint
            torch.cuda.empty_cache()
    else:
        print('No Resume Model')

    if args.DataParallel:
        model = nn.DataParallel(model)

    if torch.cuda.is_available():
        model = model.cuda()

    return model



def Bulid_Model(args):
    '''Bulid Model'''
    # model_urls = {'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    #               'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',}
    model=TransNet_Dual(args)
    #model=TransNet(args)
    
    if args.DataParallel:
        model = nn.DataParallel(model)
    if args.Resume_Model_all != 'None':
        print('Resume Model: {}'.format(args.Resume_Model_all))
        checkpoint = torch.load(args.Resume_Model_all, map_location='cpu')
        model.load_state_dict(checkpoint, strict=False)
        #Save GPU Memory
        del checkpoint
        torch.cuda.empty_cache()
    else:
        print('Resume Model: {}'.format(args.Resume_Model_all))

    if args.DataParallel:
        model = nn.DataParallel(model)
       
    if torch.cuda.is_available():
        model = model.cuda()

    # #if args.Distribute == 'Basic':
    # if  args.Backbone == 'ResNet18':
    #     model = ResNet18()
    # elif args.Backbone == 'ResNet50':
    #     model = ResNet50()

    # #elif args.Distribute == 'Compound':
    # #    model = ResNet101_Compound(args.Dim)

    # if args.Resume_Model != 'None':
    #     if args.Resume_Model =='imagenet':
    #         print('Resume Model: {}'.format(args.Resume_Model))
    #         if  args.Backbone == 'ResNet18':
    #             model.load_state_dict(model_zoo.load_url(model_urls['resnet18']),strict=False)
    #         elif args.Backbone == 'ResNet50':
    #             model.load_state_dict(model_zoo.load_url(model_urls['resnet50']),strict=False)
    #         # del checkpoint
    #         # torch.cuda.empty_cache()
    #         #torch.cuda.empty_cache()
    #     else:
    #         print('Resume Model: {}'.format(args.Resume_Model))
    #         checkpoint = torch.load(args.Resume_Model, map_location='cpu')
    #         model.load_state_dict(checkpoint, strict=False)
    #     #model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    #     # Save GPU Memory
    #         del checkpoint
    #         torch.cuda.empty_cache()
    # else:
    #     print('No Resume Model')

    # if args.DataParallel:
    #     model = nn.DataParallel(model)

    # if torch.cuda.is_available():
    #     model = model.cuda()

    return model


def lr_scheduler(optimizer, iter_num, gamma, power, lr=0.001, weight_decay=0.0005):
    """Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs."""

    lr = lr * (1 + gamma * iter_num) ** (-power)

    for param_group in optimizer.param_groups:
        
        if 'lr_mult' in param_group:
            param_group['lr'] = lr * param_group['lr_mult']
        else:
            param_group['lr'] = lr

        if 'decay_mult' in param_group:    
            param_group['weight_decay'] = weight_decay * param_group['decay_mult']
        else:
            param_group['weight_decay'] = weight_decay

    return optimizer, lr

def lr_scheduler_withoutDecay(optimizer, lr=0.001, weight_decay=0.0005):
    """Learning rate without Decay."""

    for param_group in optimizer.param_groups:
        
        if 'lr_mult' in param_group:
            param_group['lr'] = lr * param_group['lr_mult']
        else:
            param_group['lr'] = lr

        if 'decay_mult' in param_group:    
            param_group['weight_decay'] = weight_decay * param_group['decay_mult']
        else:
            param_group['weight_decay'] = weight_decay

    return optimizer, lr




def Set_Param_Optim(model):
    '''Set parameters optimizer'''

    # Expression Recognition Experiment
    #if args.Experiment == 'EM': 
    for param in model.parameters():
        param.requires_grad = True

    param_optim = filter(lambda p:p.requires_grad, model.parameters())

    # AU Recognition Experimen
    
    return param_optim


def Set_Criterion_Optimizer(args, param_optim):
    '''Set Criterion and Optimizer'''

    #optimizer = optim.SGD(param_optim, lr=args.LearnRate, momentum=0.9)
    optimizer = optim.SGD(param_optim, lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)

    return optimizer


def Adjust_Learning_Rate(optimizer, epoch, LR):
    '''Adjust Learning Rate'''
    # lr = 0.001

    # for i in range(epoch):

    if epoch<=30: #15
        lr = LR
    elif epoch<=40: #30
        lr = 0.1 * LR
    else:
         lr = 0.01 * LR
    # elif epoch<=45: 
    #     lr = 0.01 * LR
    # # else:
    #     lr = 0.001 * LR
    # elif epoch<=60: 
    #      lr = 0.001 * LR
    # elif epoch<=75: 
    #      lr = 0.0001 * LR
    # else:
    #      lr = 0.00001 * LR
    # else:
    #     lr = 0.001 * LR

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        #param_group['weight_decay'] = weight_decay

    return optimizer, lr


def Compute_Accuracy(args, pred, target, acc, prec, recall):
    '''Compute the accuracy of all samples, the accuracy of positive samples, the recall of positive samples.'''

    pred = pred.cpu().data.numpy()
    pred = np.argmax(pred,axis=1)
    target = target.cpu().data.numpy()

    pred = pred.astype(np.int32).reshape(pred.shape[0],)
    target = target.astype(np.int32).reshape(target.shape[0],)

    for i in range(7):
        TP = np.sum((pred==i)*(target==i))
        TN = np.sum((pred!=i)*(target!=i))
        
        # Compute Accuracy of All --> TP+TN / All
        acc[i].update(np.sum(pred==target),pred.shape[0])
        
        # Compute Precision of Positive --> TP/(TP+FP)
        prec[i].update(TP,np.sum(pred==i))

        # Compute Recall of Positive --> TP/(TP+FN)
        recall[i].update(TP,np.sum(target==i))


def Compute_Accuracy_Expression(args, pred, target, loss_, acc_1, acc_2, prec, recall, loss):
    '''Compute the accuracy of all samples, the accuracy of positive samples, the recall of positive samples and the loss'''

    pred = pred.cpu().data.numpy()
    pred = np.argmax(pred, axis=1)
    target = target.cpu().data.numpy()

    pred = pred.astype(np.int32).reshape(pred.shape[0],)
    target = target.astype(np.int32).reshape(target.shape[0],)

    if args.Distribute == 'Basic':
        numOfLabel = 7
    elif args.Distribute == 'Compound':
        numOfLabel = 11
     
    for index in range(numOfLabel):
        TP = np.sum((pred == index) * (target == index))
        TN = np.sum((pred != index) * (target != index))

        # Compute Accuracy of All --> TP+TN / All
        acc_1[index].update(np.sum(pred == target), pred.shape[0])
        acc_2[index].update(TP, np.sum(target == index))

        # Compute Precision of Positive --> TP/(TP+FP)
        prec[index].update(TP, np.sum(pred == index))

        # Compute Recall of Positive --> TP/(TP+FN)
        recall[index].update(TP, np.sum(target == index))

    # Compute Loss
    loss.update(float(loss_.cpu().data.numpy()))

def Show_Accuracy(acc, prec, recall, class_num=7):
    """Compute average of accuaracy/precision/recall/f1"""

    # Compute F1 value    
    f1 = [AverageMeter() for i in range(class_num)]
    for i in range(class_num):
        if prec[i].avg==0 or recall[i].avg==0:
            f1[i].avg = 0
            continue
        f1[i].avg = 2*prec[i].avg*recall[i].avg/(prec[i].avg+recall[i].avg)
    
    # Compute average of accuaracy/precision/recall/f1
    acc_avg, prec_avg, recall_avg, f1_avg = 0, 0, 0, 0

    for i in range(class_num):
        acc_avg+=acc[i].avg
        prec_avg+=prec[i].avg
        recall_avg+=recall[i].avg
        f1_avg+=f1[i].avg

    acc_avg, prec_avg, recall_avg, f1_avg = acc_avg/class_num,prec_avg/class_num, recall_avg/class_num, f1_avg/class_num

    # Log Accuracy Infomation
    Accuracy_Info = ''
    
    Accuracy_Info+='    Accuracy'
    for i in range(class_num):
        Accuracy_Info+=' {:.4f}'.format(acc[i].avg)
    Accuracy_Info+='\n'

    Accuracy_Info+='    Precision'
    for i in range(class_num):
        Accuracy_Info+=' {:.4f}'.format(prec[i].avg)
    Accuracy_Info+='\n'

    Accuracy_Info+='    Recall'
    for i in range(class_num):
        Accuracy_Info+=' {:.4f}'.format(recall[i].avg)
    Accuracy_Info+='\n'

    Accuracy_Info+='    F1'
    for i in range(class_num):
        Accuracy_Info+=' {:.4f}'.format(f1[i].avg)
    Accuracy_Info+='\n'

    return Accuracy_Info, acc_avg, prec_avg, recall_avg, f1_avg

    
def GetIndexFromDataset(model, target_dataset):

    #target_index = [[] for i in range(7)]

    model.eval()  
    for step, (input,  label) in enumerate(target_dataset):
        input,index,ID=input
        #index=np.array
        #a=index[0]
        input,  label = input.cuda(),  label.cuda()
        Trag= False
        with torch.no_grad():
            #pred,_,_ = model(input,input)
            pred,_,_ = model(input,input,Trag)
         
        pred = torch.argmax(pred,dim=1).cpu()
        #print(target_dataset.dataset.labels[index.cpu().numpy()])
        #print('================================================')

        target_dataset.dataset.labels[index.cpu().numpy()]= pred.numpy()
        #print(target_dataset.dataset.labels[index.cpu().numpy()])
        #h=0
        # pes_label[0]=index
        # pes_label[1]=index
        #print(pes_label)
        #target_index[pred].append(index)

    # Target Index
    model.train()


def ZL_GetIndexFromDataset(args, name, model, target_dataset):
    
    #target_index = [[] for i in range(7)]
    #target_dataset.drop_last=False
    #data_loader = data.DataLoader(dataset=target_dataset.dataset, batch_size=args.train_batch_size, shuffle=True, num_workers=args.Num_Workers, drop_last=False)
    

    model.eval()  
    file_list = []
    #for step, (input,  label) in enumerate(data_loader):
    for idx in range(len(target_dataset.dataset.imgs)):
        input,label=target_dataset.dataset[idx]
        input,index,ID=input
        input = input.cuda()
        Trag= False
        with torch.no_grad():
            #pred,_,_ = model(input,input)
            pred,_,_ = model(input.unsqueeze(0),input.unsqueeze(0),Trag)
        
        pred=torch.softmax(pred,1)
        pred1 = torch.argmax(pred,dim=1).cpu()
        for i in range(len(pred1)):
            if pred[i][pred1[i]]>=0.95:
                dir_image2_temp = ID+' ' + str(np.array(pred1[i]))
                file_list.append(dir_image2_temp)
    na = args.Log_Name + str(name)
    with open('pre_label/'+na+'.txt', 'w') as f:
        for list in file_list:
            #print(list)
            f.write(str(list) + '\n')
        
    f.close()
    # Target Index
    model.train()
    #na=name
    #target_dataset.drop_last=True
    return na



def ZL_GetIndexFromDataset_1(args, name, model, target_dataset):
    
    #target_index = [[] for i in range(7)]

    model.eval()  
    file_list = []
    for step, (input,  label) in enumerate(target_dataset):
        input,index,ID=input
        #index=np.array
        #a=index[0]
        input,  label = input.cuda(),  label.cuda()
        Trag= False
        with torch.no_grad():
            #pred,_,_ = model(input,input)
            pred,_,_ = model(input,input,Trag)
        
        pred=torch.softmax(pred,1)
        pred1 = torch.argmax(pred,dim=1).cpu()
        for i in range(len(pred1)):
            # print(i)
            # print(pred1[i])
            # print(pred[i][pred1[i]])
            if pred[i][pred1[i]]>=0.98:
                dir_image2_temp = ID[i]+' ' + str(np.array(pred1[i]))
                file_list.append(dir_image2_temp)
        #print(target_dataset.dataset.labels[index.cpu().numpy()])
        #print('================================================')

            #target_dataset.dataset.labels[index.cpu().numpy()]= pred1.numpy()
        #print(target_dataset.dataset.labels[index.cpu().numpy()])
        #h=0
        # pes_label[0]=index
        # pes_label[1]=index
        #print(pes_label)
        #target_index[pred].append(index)
    #file=open('pre_label/'+'psolabel'+name+'.txt','w')    
    na = args.Log_Name + str(name)
    with open('pre_label/'+na+'.txt', 'w') as f:
        for list in file_list:
            #print(list)
            f.write(str(list) + '\n')
        
    f.close()
    # Target Index
    model.train()
    #na=name
    return na

def Get_Pselabel_dataloader(args,name):
    
    trans = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]),
            ])
    target_trans = None
    
    list_patition_label = pd.read_csv('pre_label/'+name+'.txt', header=None, delim_whitespace=True)
    list_patition_label = np.array(list_patition_label)
    
    data_imgs=[]
    data_labels=[]
    for index in range(list_patition_label.shape[0]):
      
        data_imgs.append(list_patition_label[index, 0].split('.')[0]+'.'+list_patition_label[index, 0].split('.')[1][0:3])#'.png')
                        #data_imgs.append(dataPath_prefix+'/RAF/basic/Image/original/'+list_patition_label[index,0])
                        ##append()接受一个对象参数，把对象添加到列表的尾部
        data_labels.append(list_patition_label[index,1])
            
    data_set = MyDataset_EM_nobox(data_imgs, data_labels,  flag='train', transform=trans, target_transform=target_trans)
    #data_set = MyDataset_EM_SFEW(data_imgs, data_labels,  flag1)
    # DataLoader
   
    data_loader = data.DataLoader(dataset=data_set, batch_size=args.train_batch_size, shuffle=True, num_workers=args.Num_Workers, drop_last=True)
    return data_loader

def Visualization(figName, model, dataloader, useClassify=True, domain='Source'):
    '''Feature Visualization in Source/Target Domain.'''
    
    assert useClassify in [True, False], 'useClassify should be bool.'
    assert domain in ['Source', 'Target'], 'domain should be source or target.'

    model.eval()

    Feature, Label = [], []

    # Get Cluster
    # for i in range(7):
    #     if domain=='Source':
    #         Feature.append(model.SourceMean.running_mean[i].cpu().data.numpy())
    #     elif domain=='Target':
    #         Feature.append(model.TargetMean.running_mean[i].cpu().data.numpy())
    # Label.append(np.array([7 for i in range(7)]))

    # Get Feature and Label
    for step, (input, label) in enumerate(dataloader):
        input, index=input
        input, label = input.cuda(),  label.cuda()
        Trag=False
        with torch.no_grad():
            feature, output, loc_output = model(input, input,Trag)
        Feature.append(feature.cpu().data.numpy())
        Label.append(label.cpu().data.numpy())

    Feature = np.vstack(Feature)
    Label = np.concatenate(Label)

    # Using T-SNE
    tsne = TSNE(n_components=2, init='pca', random_state=0, perplexity=50, early_exaggeration=3)
    embedding = tsne.fit_transform(Feature)

    # Draw Visualization of Feature
    colors = {0:'red', 1:'blue', 2:'black',  3:'green',  4:'orange',  5:'purple',  6:'darkslategray'}#, 7:'black'}
    labels = {0:'Surprised', 1:'Fear', 2:'Disgust',  3:'Happy',  4:'Sad',  5:'Angry',  6:'Neutral'}
    #labels = {0:'惊讶', 1:'恐惧', 2:'厌恶', 3:'开心', 4:'悲伤', 5:'愤怒', 6:'平静', 7:'聚类中心'}

    data_min, data_max = np.min(embedding, 0), np.max(embedding, 0)
    data_norm = (embedding - data_min) / (data_max - data_min)

    fig = plt.figure()
    ax = plt.subplot(111)

    for i in range(7):
        data_x, data_y = data_norm[Label==i][:,0], data_norm[Label==i][:,1]
        scatter = plt.scatter(data_x, data_y, c='', edgecolors=colors[i], s=5, label=labels[i], marker='^', alpha=0.6)
    #scatter = plt.scatter(data_norm[Label==6][:,0], data_norm[Label==6][:,1], c=colors[6], s=20, label=labels[6], marker='^', alpha=1)

    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height * 0.8])
    
    plt.legend(handles=[mpatches.Patch(color=colors[i], label="{:s}".format(labels[i]) ) for i in range(7)],
               loc='upper left',
               prop = {'size':8},
               #prop = matplotlib.font_manager.FontProperties(fname='simhei.ttf'), 
               bbox_to_anchor=(1.05,0.85),
               borderaxespad=0)
    plt.savefig(fname='{}'.format(figName), format="pdf", bbox_inches = 'tight')

def VisualizationForTwoDomain(figName, model, source_dataloader, target_dataloader, useClassify=True, showClusterCenter=False):
    '''Feature Visualization in Source and Target Domain.'''
    
    model.eval()

    Feature_Source, Label_Source, Feature_Target, Label_Target = [], [], [], []

    # Get Feature and Label in Source Domain
    # if showClusterCenter:
    #     for i in range(7):
    #         Feature_Source.append(model.SourceMean.running_mean[i].cpu().data.numpy())
    #     Label_Source.append(np.array([7 for i in range(7)]))   

    # for step, (input, landmark, label) in enumerate(source_dataloader):
    #     input, landmark, label = input.cuda(), landmark.cuda(), label.cuda()
    #     with torch.no_grad():
    #         feature, output, loc_output = model(input, landmark, useClassify, domain='Source')
    for step, (input, label) in enumerate(source_dataloader):
        input, index,ID=input
        input, label = input.cuda(),  label.cuda()
        Trag=False
        with torch.no_grad():
            feature, output, loc_output = model(input, input,Trag)
    
        #Feature_Source.append(input.cpu().data.numpy())
        Feature_Source.append(feature.cpu().data.numpy())
        Label_Source.append(label.cpu().data.numpy())

    Feature_Source = np.vstack(Feature_Source)
    Label_Source = np.concatenate(Label_Source)

    # Get Feature and Label in Target Domain
    # if showClusterCenter:
    #     for i in range(7):
    #         Feature_Target.append(model.TargetMean.running_mean[i].cpu().data.numpy())
    #     Label_Target.append(np.array([7 for i in range(7)]))
    for step, (input, label) in enumerate(target_dataloader):
        input, index,ID=input
        input, label = input.cuda(),  label.cuda()
        Trag=False
        with torch.no_grad():
            feature, output, loc_output = model(input, input,Trag)
            
            
        #Feature_Target.append(input.cpu().data.numpy())
        Feature_Target.append(feature.cpu().data.numpy())
        Label_Target.append(label.cpu().data.numpy())

    # for step, (input, landmark, label) in enumerate(target_dataloader):
    #     input, landmark, label = input.cuda(), landmark.cuda(), label.cuda()
    #     with torch.no_grad():
    #         feature, output, loc_output = model(input, landmark, useClassify, domain='Target')

    #     Feature_Target.append(feature.cpu().data.numpy())
    #     Label_Target.append(label.cpu().data.numpy())

    Feature_Target = np.vstack(Feature_Target)
    Label_Target = np.concatenate(Label_Target)

    # Sampling from Source Domain
    Feature_Temple, Label_Temple = [], []
    for i in range(8):
        num_source = np.sum(Label_Source==i)
        num_target = np.sum(Label_Target==i)

        num = num_source if num_source <= num_target else num_target 

        Feature_Temple.append(Feature_Source[Label_Source==i][:num])
        Label_Temple.append(Label_Source[Label_Source==i][:num]) 
 
    Feature_Source = np.vstack(Feature_Temple) 
    Label_Source = np.concatenate(Label_Temple)

    Label_Target+=8

    Feature = np.vstack((Feature_Source, Feature_Target))
    Label = np.concatenate((Label_Source, Label_Target))

    # Using T-SNE
    tsne = TSNE(n_components=2, init='pca', random_state=0, perplexity=50, early_exaggeration=3)
    embedding = tsne.fit_transform(Feature)

    # Draw Visualization of Feature
    #colors = {0:'firebrick', 1:'aquamarine', 2:'goldenrod',  3:'cadetblue',  4:'saddlebrown',  5:'yellowgreen',  6:'navy'}
    colors = {0:'red', 1:'blue', 2:'black',  3:'green',  4:'orange',  5:'purple',  6:'darkslategray'}#, 7:'black'}
    labels = {0:'Surprised', 1:'Fear', 2:'Disgust',  3:'Happy',  4:'Sad',  5:'Angry',  6:'Neutral'}

    data_min, data_max = np.min(embedding, 0), np.max(embedding, 0)
    data_norm = (embedding - data_min) / (data_max - data_min)

    fig = plt.figure()
    ax = plt.subplot(111)

    for i in range(7):

        data_source_x, data_source_y = data_norm[Label==i][:,0], data_norm[Label==i][:,1]
        source_scatter = plt.scatter(data_source_x, data_source_y, color="none", edgecolor=colors[i], s=20, label='S-'+labels[i][0]+labels[i][1], marker="*", alpha=0.4, linewidth=0.5)
        
        data_target_x, data_target_y = data_norm[Label==(i+8)][:,0], data_norm[Label==(i+8)][:,1]
        target_scatter = plt.scatter(data_target_x, data_target_y, color=colors[i], edgecolor="none", s=30, label='T-'+labels[i][0]+labels[i][1], marker="v", alpha=0.6, linewidth=0.2)

        if i==0:
            source_legend = source_scatter
            target_legend = target_scatter

    # if showClusterCenter:
    #     source_cluster = plt.scatter(data_norm[Label==7][:,0], data_norm[Label==7][:,1], c='black', s=20, label='Source Cluster Center', marker='^', alpha=1)
    #     target_cluster = plt.scatter(data_norm[Label==15][:,0], data_norm[Label==15][:,1], c='black', s=20, label='Target Cluster Center', marker='s', alpha=1)
    plt.legend(fontsize=5)
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height * 0.8])
    '''
    l1 = plt.legend(handles=[mpatches.Patch(color=colors[i], 
                    label="{:s}".format(labels[i]) ) for i in range(7)], 
                    loc='upper left', 
                    prop = {'size':8})
                    #bbox_to_anchor=(1.05,0.85), 
                    #borderaxespad=0)
    
    if showClusterCenter:
        plt.legend([source_legend, target_legend, source_cluster, target_cluster],
                   ['Source Domain', 'Target Domain', 'Source Cluster Center', 'Target Cluster Center'],
                   loc='lower left',
                   prop = {'size':7})
    else:
        plt.legend([source_legend, target_legend], ['Source Domain', 'Target Domain'], loc='lower left', prop = {'size':7})
    plt.gca().add_artist(l1)
    '''
    plt.savefig(fname='{}.pdf'.format(figName), format="pdf", bbox_inches='tight')

