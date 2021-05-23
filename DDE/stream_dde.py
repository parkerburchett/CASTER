import numpy as np
import pandas as pd
import torch
from torch.utils import data

from subword_nmt.apply_bpe import BPE
import codecs # in the standard python library

'''
DataFolder = './data'

unsup_train_file = 'food_smiles.csv' & 'drug_smiles.csv' & 'ddi_smiles.csv'
 
unsupervised pair dataset to pretrain the network
SMILES string as input

DDI supervised data files:

    train = 'train.csv'
    val = 'val.csv'
    test = 'test.csv' 
    

build a UnsupData which returns v_d, v_f for a batch

supTrainData which return v_d, v_f, label for DDI only 
supTrainData.num_of_iter_in_a_epoch contains iteration in an epoch

ValData which return v_d, v_f, label for DDI only 

'''



dataFolder = './data'

vocab_path = dataFolder + '/codes.txt' # this is a predefined set of common pseduo functional groups defined as common patterns
bpe_codes_fin = codecs.open(vocab_path)
bpe = BPE(bpe_codes_fin, merges=-1, separator='') # this is the module used from first feature extraction

vocab_map = pd.read_csv(dataFolder + '/subword_units_map.csv')
idx2word = vocab_map['index'].values
words2idx = dict(zip(idx2word, range(0, len(idx2word)))) 
# Once You have features, this creates the 1 hot vector
# pseudofunctional groups with index
max_set = 30

def smiles2index(s1, s2):
    """
        s1 and s2 are smile strings. 
        bpe.process_line(SMILE STRING) -> returns a list of functional groups based on the vocab in codes.txt
        I don't quite know how that was generated. Need to find out and add documentation
    """
    t1 = bpe.process_line(s1).split() #Break s1 in to a list of pseudofunctional groups based on codes.txt
    t2 = bpe.process_line(s2).split() #
    i1 = [words2idx[i] for i in t1] # get the index value based on all the keys in the index. EG if the SMILE code has C#N functional group and that has the value 12 in the dict it will return 12
    i2 = [words2idx[i] for i in t2] # index
    return i1, i2

def index2multi_hot(i1, i2):
    """
        return a 1 hot vector of all the common index values of the pseudofunctional groups
    """
    v1 = np.zeros(len(idx2word),) # default is a 1 hot vector of all zeros.
    v2 = np.zeros(len(idx2word),)
    v1[i1] = 1 # set all the indexes to be 1 where there is a functional group
    v2[i2] = 1
    v_d = np.maximum(v1, v2) # This is similar to just the sql Union. zeros where both not there and 1s where either pseudo functional group is present in s1 or s2
    return v_d

def index2single_hot(i1, i2): # Not used
    comb_index  = set(i1 + i2)
    v_f = np.zeros((max_set*2, len(idx2word)))
    for i, j in enumerate(comb_index):
        if i < max_set*2:
            v_f[i][j] = 1
        else:
            break
    return v_f

def smiles2vector(s1, s2):
    """
        Pass this function two SMILE strings, s1 and s2. 
        Returns a 1 hot vector of v_d of length 1722
        Each index in words2idx( a dictionary ) corrosponds to a pseudo Functional group.
        The vector has a 1 if either is present in the drugs. 

        For my purposes, You will only want to convert a single SMILE to a vector

    """
    i1, i2 = smiles2index(s1, s2)
    v_d = index2multi_hot(i1, i2)
    #v_f = index2single_hot(i1, i2)
    return v_d


def convert_single_SMILE_to_vector(smile: str) -> np.array:
    """
        Use 1 hot encoding to convert a smile to a 1 hot vector based on the indexes in words2idx
        # untested
        @parkerburchett 

        I don't understand why it uses both subword_units_map
    """
    pesudo_functional_groups = bpe.process_line(smile).split() 
    indexs_of_pesudo_functional_groups = [words2idx[group] for group in pesudo_functional_groups]

    vector_of_smile =v1 = np.zeros(len(idx2word),) # initalize as zeros.
    vector_of_smile[indexs_of_pesudo_functional_groups] = 1 # assign all the index of pseudo functional groups present in smile to be 1.

    return vector_of_smile


class supData(data.Dataset):

    def __init__(self, list_IDs, labels, df_ddi):
        'Initialization'
        self.labels = labels
        self.list_IDs = list_IDs
        self.df = df_ddi
        
    def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        # Load data and get label
        index = self.list_IDs[index]
        s1 = self.df.iloc[index].Drug1_SMILES
        s2 = self.df.iloc[index].Drug2_SMILES
        v_d = smiles2vector(s1, s2)
        y = self.labels[index]
        return v_d, y
    
class supData_index(data.Dataset):

    def __init__(self, list_IDs, labels, df_ddi):
        'Initialization'
        self.labels = labels
        self.list_IDs = list_IDs
        self.df = df_ddi
        
    def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        # Load data and get label
        index = self.list_IDs[index]
        s1 = self.df.iloc[index].Drug1_SMILES
        s2 = self.df.iloc[index].Drug2_SMILES
        v_d = smiles2vector(s1, s2)
        y = self.labels[index]
        return v_d, y, index

class unsupData(data.Dataset):

    def __init__(self, list_IDs, df):
        'Initialization'
        self.list_IDs = list_IDs
        self.df = df
    def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)

    def __getitem__(self, index):
        'Generates one sample of data'

        # Load data and get label
        index = self.list_IDs[index]
        s1 = self.df.iloc[index].input1_SMILES
        s2 = self.df.iloc[index].input2_SMILES
        v_d = smiles2vector(s1, s2)
        return v_d