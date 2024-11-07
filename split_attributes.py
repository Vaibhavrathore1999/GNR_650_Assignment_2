import numpy as np; np.random.seed(1)
import torch; torch.manual_seed(1)
import torch.nn as nn
import torch.nn.functional as F
from time import time
from tqdm import tqdm
from scipy import io
from torch.utils.data import DataLoader
import random,pickle
from scipy.io import savemat

DATA_DIR = f'IAB-GZSL/data/AWA2'
data = io.loadmat(f'{DATA_DIR}/res101.mat')
attrs_mat = io.loadmat(f'{DATA_DIR}/att_splits_original.mat')


a = ["/".join(data["image_files"][:,0][i][0].split("/")[-2:]) for i in range(data["image_files"].shape[0])]
b = sorted(list(set([data["image_files"][:,0][i][0].split("/")[-2] for i in range(data["image_files"].shape[0])])))

class_counts = {}
for c in b:
    class_counts[c] = sum([1 for i in a if c == i.split("/")[0]])

train_sum = 0
for c in b[0:25]:
    train_sum += class_counts[c]
test_sum = 0
for c in b[25:50]:
    test_sum += class_counts[c]

trainval_loc = np.array([list(range(1, train_sum+1))]).T
test_loc = np.array([list(range(train_sum+1, train_sum+test_sum+1))]).T
np.random.shuffle(trainval_loc)
train_loc = trainval_loc[:int(train_sum*0.8)]
val_loc = trainval_loc[int(train_sum*0.8):]

attr_dict = {k:v for k,v in attrs_mat.items()}
attr_dict["test_seen_loc"] = trainval_loc.astype(np.uint16)
attr_dict["test_unseen_loc"] = test_loc.astype(np.uint16)
attr_dict["train_loc"] = train_loc.astype(np.uint16)
attr_dict["val_loc"] = val_loc.astype(np.uint16)
attr_dict["trainval_loc"] = trainval_loc.astype(np.uint16)

savemat("IAB-GZSL/data/AWA2/att_splits.mat", attr_dict)

with open('/home/ninad/vaibhav_r/GNR_650/Assignment_2/class_embeddings.pkl', 'rb') as f:
    embeddings = pickle.load(f)
# print(embeddings.keys())
# attr_dict_new = {k:v for k,v in attrs_mat.items()}
embd = np.array(embeddings[attrs_mat["allclasses_names"][0][0][0]]).T
arr = np.zeros((300, 50))
for i in range(50):
  arr[:, i] = np.array(embeddings[attrs_mat["allclasses_names"][i][0][0]]).T
attr_dict['att'] = arr
attr_dict['original_att'] = arr
savemat("IAB-GZSL/data/AWA2/att_splits.mat", attr_dict)
