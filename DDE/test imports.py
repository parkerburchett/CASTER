print('sdfsdf')
import sys
import os
os.chdir('./DDE')

print(sys.executable)
import torch
from torch.autograd import Variable
import torch.nn.functional as F
from torch.utils import data
from torch import nn 
import copy

from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from time import timepip
from sklearn.metrics import roc_auc_score, precision_recall_curve
from sklearn.model_selection import KFold
torch.manual_seed(2)    # reproducible torch:2 np:3
np.random.seed(3)

from dde_config import dde_NN_config
from dde_torch import dde_NN_Large_Predictor
from stream_dde import supData, unsupData

# Everything imports correctly when you are in .\CASTER\venv\Scripts\python.exe