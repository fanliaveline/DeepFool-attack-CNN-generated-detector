import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
import sys
sys.path.append("..")
import os
from utils.utils import MyDataset, validate, show_confMat
import torch.optim as optim
import torch.utils.data as data_utils
from torch.autograd import Variable
import math
from networks.resnet import resnet50
from PIL import Image
from deepfool import deepfool
from eval_config import *
from options.test_options import TestOptions
from PIL import ImageChops
import os
net = resnet50(num_classes=1)
#net = resnet50()
state_dict = torch.load(model_path, map_location='cpu')
net.load_state_dict(state_dict['model'])
#for para in net.parameters():
    #para.require_grad=False


net.fc = nn.Linear(2048, 2) 

net.cuda()
# Switch to evaluation mode
net.eval()

im_orig = Image.open('02938.png')

mean = [ 0.485, 0.456, 0.406 ]
std = [ 0.229, 0.224, 0.225 ]




def show_image(image, title=""):

    plt.figure()

    if title != "":

        plt.title(title)

    plt.imshow(image)

    plt.show()
# Remove the mean
im = transforms.Compose([
    transforms.Scale(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean = mean,
                         std = std)])(im_orig)

r, loop_i, label_orig, label_pert, pert_image = deepfool(im, net)

labels = open(os.path.join('synset_words.txt'), 'r').read().split('\n')

str_label_orig = labels[np.int(label_orig)].split(',')[0]
str_label_pert = labels[np.int(label_pert)].split(',')[0]

print("Original label = ", str_label_orig)
print("Perturbed label = ", str_label_pert)

def clip_tensor(A, minv, maxv):
    A = torch.max(A, minv*torch.ones(A.shape))
    A = torch.min(A, maxv*torch.ones(A.shape))
    return A

clip = lambda x: clip_tensor(x, 0, 255)

trans_im_2 = transforms.Compose([

        transforms.Resize(256),

        transforms.CenterCrop(224),

        transforms.ToTensor()])



    # Transform to PIL image format

trans_pil = transforms.ToPILImage()
tf = transforms.Compose([transforms.Normalize(mean=[0, 0, 0], std=list(map(lambda x: 1 / x, std))),

                             transforms.Normalize(mean=list(map(lambda x: -x, mean)), std=[1, 1, 1]),

                             transforms.Lambda(clip),

                             transforms.ToPILImage(),

                             transforms.CenterCrop(224)])



    # Classic image format with cropping

tf_2 = transforms.Compose([transforms.ToPILImage(),

                               transforms.CenterCrop(224)])

pert_only_image = ImageChops.subtract(tf_2(trans_im_2(im_orig)), tf(pert_image.cpu()[0]), scale=1.0 / 255)
show_image(pert_only_image, "pert only")
