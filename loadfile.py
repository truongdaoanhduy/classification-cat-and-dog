import os
import torch
import torch.nn as nn
import torchvision
import glob
from torch.utils.data import DataLoader,random_split,Dataset
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
import torchvision.transforms as Transforms

path = 'Data/train/'

path_data_train = glob.glob(path+'*/**.jpg')

