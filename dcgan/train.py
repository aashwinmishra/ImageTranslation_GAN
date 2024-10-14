import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms

import numpy as np
import matplotlib.pyplot as plt
from zipfile import ZipFile
import os

from data_setup import unzip_data, get_dataloaders
from utils import get_devices, set_seeds, show_batch
from models import AnimeDiscriminator, AnimeGenerator
from engine import train 


data_path = "./drive/MyDrive/Practice_Data/AnimeFacesDatasetKaggle.zip"
data_dir = "./Data/"
unzip_data(data_path, data_dir)
image_size = 64
stats = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
transform = transforms.Compose([transforms.Resize(image_size),
                                transforms.CenterCrop(image_size),
                                transforms.ToTensor(),
                                transforms.Normalize(*stats)])
dl = get_dataloaders(data_dir, transform, 64)
device = get_devices()
disc = AnimeDiscriminator().to(device)
gen = AnimeGenerator().to(device)
train(disc, gen, dl, 64, 128, device, 25, "./generated")

