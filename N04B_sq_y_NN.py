#!/usr/bin/env python
# coding: utf-8
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
# Microsoft VS header
#--------------------------------------------------#
import os 
import sys
import os.path
from sys import platform
from pathlib import Path
#--------------------------------------------------#
if os.name == 'nt' or platform == 'win32':
    print("Running on Windows")
    if 'ptvsd' in sys.modules:
        print("Running in Visual Studio")
        try:
            os.chdir(os.path.dirname(__file__))
            print('CurrentDir: ', os.getcwd())
        except:
            pass
#--------------------------------------------------#
    else:
        print("Running outside Visual Studio")
        try:
            if not 'workbookDir' in globals():
                workbookDir = os.getcwd()
                print('workbookDir: ' + workbookDir)
                os.chdir(workbookDir)
        except:
            pass
#--------------------------------------------------#
#########################################################################################################
#########################################################################################################
import sys
import time
import torch
import numpy as np
import pandas as pd
import pickle
import argparse
import scipy
import random
import subprocess
#--------------------------------------------------#
from torch import nn
from torch.utils import data
from torch.nn.utils.weight_norm import weight_norm
#from torchvision import models
#from torchsummary import summary
#--------------------------------------------------#
from tape import datasets
from tape import TAPETokenizer
#--------------------------------------------------#
from sklearn import datasets
from sklearn import preprocessing
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
#--------------------------------------------------#
from sklearn import svm
from sklearn import metrics
from sklearn import preprocessing
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve
from sklearn.preprocessing import StandardScaler
from typing import Union, List, Tuple, Sequence, Dict, Any, Optional, Collection
#--------------------------------------------------#
#from sklearn import svm
#from sklearn.model_selection import GridSearchCV
#from sklearn.tree import DecisionTreeRegressor
#--------------------------------------------------#
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
#--------------------------------------------------#
from tpot import TPOTRegressor
from ipywidgets import IntProgress
from pathlib import Path
from copy import deepcopy
#--------------------------------------------------#
from datetime import datetime
#--------------------------------------------------#
seed=42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
## Args
Step_code="S05A_"
data_folder = Path("S_DataProcessing/")
embedding_file_list = ["S03_embedding_ESM_1B.p", 
                       "S03_embedding_BERT.p", 
                       "S03_embedding_TAPE.p", 
                       "S03_embedding_ALBERT.p", 
                       "S03_embedding_T5.p", 
                       "S03_embedding_TAPE_FT.p", 
                       "S03_embedding_Xlnet.p"] # 0, 1, 2
embedding_file = embedding_file_list[0]
properties_file= "S00_seq_prpty.p"
# embedding_file is a dict, {"seq_embeddings":seq_embeddings, "seq_ids":seq_ids, "seq_all_hiddens":seq_all_hiddens}
# properties_file is a dict, with keys as follows, 
#====================================================================================================#
# Select properties (Y) of the model 
prpty_list = ["kcat_cMUP",          # 0
              "KM_cMUP",            # 1
              "kcatOverKM_cMUP",    # 2
              "kcatOverKM_MeP",     # 3
              "kcatOverKM_MecMUP",  # 4
              "Ki_Pi",              # 5
              "fa",                 # 6
              "kcatOverKM_MePkchem",# 7
              "FC1",                # 8
              "FC2_3",              # 9
              "FC4",                # 10
              ]
prpty_select = prpty_list[0]
#====================================================================================================#
# Prediction NN settings
epoch_num=100
batch_size=128
learning_rate=0.001
NN_type_list=["Reg", "Clf"]
NN_type=NN_type_list[0]
#====================================================================================================#
hid_dim = 256   # 256
kernal_1 = 3    # 5
out_dim = 1     # 2
kernal_2 = 3    # 3
last_hid = 512  # 1024
dropout = 0.    # 0

hid_1=512
hid_2=512
#--------------------------------------------------#
'''
model = CNN(
            in_dim = NN_input_dim,
            hid_dim = 1024,
            kernal_1 = 5,
            out_dim = 2, #2
            kernal_2 = 3,
            max_len = seqs_max_len,
            last_hid = 2048, #256
            dropout = 0.
            )
            '''
#====================================================================================================#
# Results Output
results_folder = Path("S_DataProcessing/" + Step_code +"intermediate_results/")
output_file_3 = Step_code + "_all_X_y.p"
output_file_header = Step_code + "_result_"
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
# Create Temp Folder for Saving Results
print(">>>>> Creating temporary subfolder and clear past empty folders! <<<<<")
now = datetime.now()
#d_t_string = now.strftime("%Y%m%d_%H%M%S")
d_t_string = now.strftime("%m%d-%H%M%S")
#====================================================================================================#
results_folder_contents = os.listdir(results_folder)
for item in results_folder_contents:
    if os.path.isdir(results_folder / item):
        try:
            os.rmdir(results_folder / item)
            print("Remove empty folder " + item + "!")
        except:
            print("Found Non-empty folder " + item + "!")
embedding_code=embedding_file.replace(Step_code +"embedding_", "")
embedding_code=embedding_code.replace(".p", "")
temp_folder_name = Step_code + d_t_string + "_" + prpty_select + "_" + embedding_code.replace("_","") + "_" + NN_type
results_sub_folder=Path("S_DataProcessing/" + Step_code + "intermediate_results/" + temp_folder_name +"/")
if not os.path.exists(results_sub_folder):
    os.makedirs(results_sub_folder)
print(">>>>> Temporary subfolder created! <<<<<")
#########################################################################################################
#########################################################################################################
class Tee(object):
    def __init__(self, *files):
        self.files = files
    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush() # If you want the output to be visible immediately
    def flush(self) :
        for f in self.files:
            f.flush()
#--------------------------------------------------#
orig_stdout = sys.stdout
f = open(results_sub_folder / 'print_out.txt', 'w')
sys.stdout = Tee(sys.stdout, f)
print("="*50)
#--------------------------------------------------#
print("embedding_file: ", embedding_file)
print("prpty_select: ", prpty_select)
#--------------------------------------------------#
print("epoch_num: ", epoch_num)
print("batch_size: ", batch_size)
print("learning_rate: ", learning_rate)
print("NN_type: ", NN_type)
#--------------------------------------------------#
print("-"*50)
for i in ['hid_dim', 'kernal_1', 'out_dim', 'kernal_2', 'last_hid', 'dropout']:
    print(i, ": ", locals()[i])
print("-"*50)
#########################################################################################################
#########################################################################################################
# Get Input files
# Get Sequence Embeddings from S03 pickles.
with open( data_folder / embedding_file, 'rb') as seqs_embeddings:
    seqs_embeddings_pkl = pickle.load(seqs_embeddings)
X_seqs_all_hiddens_list = seqs_embeddings_pkl['seq_embeddings'] # new: X_seqs_all_hiddens_list, old: seq_all_hiddens_list
#====================================================================================================#
# Get properties_list.
with open( data_folder / properties_file, 'rb') as seqs_properties:
    properties_dict = pickle.load(seqs_properties) # [[one_compound, y_prpty_reg, y_prpty_cls, y_prpty_cls2, SMILES], [], [], ...[] ]
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
def Get_X_y_data(X_seqs_all_hiddens_list, properties_dict, prpty_select):
    # new: X_seqs_all_hiddens_list
    y_data = properties_dict[prpty_select]
    X_seqs_all_hiddens = []
    y_seqs_prpty = []
    print("len(X_seqs_all_hiddens_list): ", len(X_seqs_all_hiddens_list))
    print("len(y_data): ", len(y_data))

    for j in range(len(X_seqs_all_hiddens_list)):
        if not (np.isnan(y_data[j])):
            X_seqs_all_hiddens.append(X_seqs_all_hiddens_list[j])
            y_seqs_prpty.append(y_data[j])
    return X_seqs_all_hiddens, y_seqs_prpty

#########################################################################################################
#########################################################################################################
if NN_type=="Reg":
    X_seqs_all_hiddens, y_seqs_prpty = Get_X_y_data(X_seqs_all_hiddens_list, properties_dict, prpty_select)
if NN_type=="Clf":
    X_seqs_all_hiddens, y_seqs_prpty = Get_X_y_data_clf(X_seqs_all_hiddens_list, properties_dict, prpty_select)
#====================================================================================================#
#save_dict=dict([])
#save_dict["X_seqs_all_hiddens"] = X_seqs_all_hiddens
#save_dict["y_seqs_prpty"] = y_seqs_prpty
#pickle.dump( save_dict , open( results_folder / output_file_3, "wb" ) )
#print("Done getting X_seqs_all_hiddens and y_seqs_prpty!")
print("len(X_seqs_all_hiddens): ", len(X_seqs_all_hiddens), ", len(y_seqs_prpty): ", len(y_seqs_prpty) )
#====================================================================================================#
# StandardScalar

scaler = StandardScaler()
X_seqs_all_hiddens = scaler.fit_transform(X_seqs_all_hiddens)

# Get size of some interests
X_seqs_all_hiddens_dim = X_seqs_all_hiddens[0].shape[0]
X_seqs_num = len(X_seqs_all_hiddens)

print("seqs dimensions: ", X_seqs_all_hiddens_dim)
print("seqs counts: ", X_seqs_num)

seqs_max_len = 546
print("seqs_max_len: ", seqs_max_len)
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#

def split_data(X_seqs_all_hiddens, y_seqs_prpty, train_split, test_split, random_state=seed):
    X_tr, X_ts, y_tr, y_ts = train_test_split(X_seqs_all_hiddens, y_seqs_prpty, test_size=(1-train_split), random_state=random_state)
    X_va, X_ts, y_va, y_ts = train_test_split(X_ts, y_ts, test_size = (test_split/(1.0-train_split)) , random_state=random_state)
    return X_tr, y_tr, X_ts, y_ts, X_va, y_va

X_tr, y_tr, X_ts, y_ts, X_va, y_va = split_data(X_seqs_all_hiddens, y_seqs_prpty, 0.8, 0.1, random_state=seed)
#====================================================================================================#
print("len(X_tr): ", len(X_tr))
print("len(X_ts): ", len(X_ts))
print("len(X_va): ", len(X_va))

#########################################################################################################
#########################################################################################################
class CNN_dataset(data.Dataset):
    def __init__(self, embedding, target, max_len):
        super().__init__()
        self.embedding = embedding
        self.target = target
        self.max_len = max_len
    def __len__(self):
        return len(self.embedding)
    def __getitem__(self, idx):
        return self.embedding[idx], self.target[idx]
    def collate_fn(self, batch:List[Tuple[Any, ...]]) -> Dict[str, torch.Tensor]:
        embedding, target = zip(*batch)
        batch_size = len(embedding)
        emb_dim = embedding[0].shape[1]
        arra = np.full([batch_size,self.max_len,emb_dim], 0.0)
        for arr, seq in zip(arra, embedding):
            arrslice = tuple(slice(dim) for dim in seq.shape)
            arr[arrslice] = seq        
        return {'seqs_embeddings': torch.from_numpy(arra),  'y_property': torch.tensor(list(target))}
#########################################################################################################
#########################################################################################################
def generate_CNN_loader(X_tr_seqs, y_tr,
                        X_va_seqs, y_va,
                        X_ts_seqs, y_ts,
                        seqs_max_len, batch_size):
    X_y_tr = CNN_dataset(list(X_tr_seqs), y_tr, seqs_max_len)
    X_y_va = CNN_dataset(list(X_va_seqs), y_va, seqs_max_len)
    X_y_ts = CNN_dataset(list(X_ts_seqs), y_ts, seqs_max_len)
    train_loader = data.DataLoader(X_y_tr, batch_size, True,  collate_fn=X_y_tr.collate_fn)
    valid_loader = data.DataLoader(X_y_va, batch_size, False, collate_fn=X_y_va.collate_fn)
    test_loader  = data.DataLoader(X_y_ts, batch_size, False, collate_fn=X_y_ts.collate_fn)
    return train_loader, valid_loader, test_loader

train_loader, valid_loader, test_loader = generate_CNN_loader(X_tr, y_tr, X_va, y_va, X_ts, y_ts, seqs_max_len, batch_size)
train_loader_list = [train_loader, ]
valid_loader_list = [valid_loader, ]
test_loader_list  = [test_loader, ]
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
NN_input_dim=X_seqs_all_hiddens_dim
print("NN_input_dim: ", NN_input_dim)

print(X_tr, y_tr)


#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
class LoaderClass(data.Dataset):
    def __init__(self, embedding, label):
        super(LoaderClass, self).__init__()
        self.embedding = embedding
        self.label = label
    def __len__(self):
        return len(self.embedding)
    def __getitem__(self, idx):
        return self.embedding[idx], self.label[idx]
#====================================================================================================#
train_loader = data.DataLoader(LoaderClass(X_tr,y_tr),batch_size,True)
valid_loader = data.DataLoader(LoaderClass(X_va,y_va),batch_size,False)
test_loader  = data.DataLoader(LoaderClass(X_ts,y_ts),batch_size,False)
#########################################################################################################
#########################################################################################################
class MLP2(nn.Module):
    def __init__(self,
                 in_dim: int,
                 hid_1: int,
                 hid_2: int                 
                 ):
        super(MLP2,self).__init__()
        self.fc1 = weight_norm(nn.Linear(in_dim,hid_1),dim=None) 
        self.dropout1 = nn.Dropout(p=0.) 
        self.fc2 = weight_norm(nn.Linear(hid_1,hid_2),dim=None)
        self.fc3 = weight_norm(nn.Linear(hid_2,1),dim=None)

    def forward(self,input):
        output = nn.functional.leaky_relu(self.fc1(input))
        output = self.dropout1(output)
        output = nn.functional.leaky_relu(self.fc2(output))
        output = self.fc3(output)
        return output
#########################################################################################################
#########################################################################################################
class MLP2_clf(nn.Module):
    def __init__(self,
                 in_dim: int,
                 hid_2: int                 
                 ):
        super(MLP2_clf,self).__init__()
        self.fc1 = weight_norm(nn.Linear(in_dim,hid_2),dim=None)
        self.dropout1 = nn.Dropout(p=0.0)
        self.fc2 = weight_norm(nn.Linear(hid_2,1),dim=None)
    #--------------------------------------------------#
    def forward(self, input):
        output = nn.functional.leaky_relu(self.fc1(input))
        output = self.dropout1(output)
        output = torch.sigmoid(self.fc2(output))
        return output
#########################################################################################################
#########################################################################################################
if NN_type == "Reg":
    #====================================================================================================#
    model = MLP2(in_dim=NN_input_dim, hid_1=hid_1, hid_2=hid_2)
    model.double()
    model.cuda()
    print("#"*50)
    print(model.eval())
    print("#"*50)
    optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)
    #optimizer = torch.optim.SGD(model.parameters(),lr=learning_rate)
    criterion = nn.MSELoss()
    #====================================================================================================#
    for epoch in range(epoch_num): 
        model.train()
        for one_seqs_cmpd_ppt_pair in train_loader:
            #print(one_seqs_cmpd_ppt_pair)
            input, target = one_seqs_cmpd_ppt_pair
            input, target = input.double().cuda(), target.cuda()
            output = model(input)
            loss = criterion(output,target.view(-1,1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        #--------------------------------------------------#
        model.eval()
        y_pred_valid = []
        y_real_valid = []
        for one_seqs_cmpd_ppt_pair in valid_loader:
            input,target = one_seqs_cmpd_ppt_pair
            input = input.double().cuda()
            output = model(input)
            output = output.cpu().detach().numpy().reshape(-1)
            target = target.numpy()
            y_pred_valid.append(output)
            y_real_valid.append(target)
        y_pred_valid = np.concatenate(y_pred_valid)
        y_real_valid = np.concatenate(y_real_valid)
        slope, intercept, r_value_va, p_value, std_err = scipy.stats.linregress(y_pred_valid, y_real_valid)
        #--------------------------------------------------#
        y_pred = []
        y_real = []
        for one_seqs_cmpd_ppt_pair in test_loader:
            input,target = one_seqs_cmpd_ppt_pair
            input = input.double().cuda()
            output = model(input)
            output = output.cpu().detach().numpy().reshape(-1)
            target = target.numpy()
            y_pred.append(output)
            y_real.append(target)
        y_pred = np.concatenate(y_pred)
        y_real = np.concatenate(y_real)

        #--------------------------------------------------#
        slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(y_pred, y_real)
        print("epoch: {} | vali_r_value: {} | loss: {} | test_r_value: {} ".format( str((epoch+1)+1000).replace("1","",1) , np.round(r_value_va,4), loss, np.round(r_value,4)))
        #====================================================================================================#
        if ((epoch+1) % 1) == 0:
            _, _, r_value, _ , _ = scipy.stats.linregress(y_pred, y_real)
            pred_vs_actual_df = pd.DataFrame(np.ones(len(y_pred)))
            pred_vs_actual_df["actual"] = y_real
            pred_vs_actual_df["predicted"] = y_pred
            pred_vs_actual_df.drop(columns=0, inplace=True)
            pred_vs_actual_df.head()
            #--------------------------------------------------#
            sns.set_theme(style="darkgrid")
            y_interval=max(np.concatenate((y_pred, y_real),axis=0))-min(np.concatenate((y_pred, y_real),axis=0))
            x_y_range=(min(np.concatenate((y_pred, y_real),axis=0))-0.1*y_interval, max(np.concatenate((y_pred, y_real),axis=0))+0.1*y_interval)
            g = sns.jointplot(x="actual", y="predicted", data=pred_vs_actual_df,
                                kind="reg", truncate=False,
                                xlim=x_y_range, ylim=x_y_range,
                                color="blue",height=7)

            g.fig.suptitle("Predictions vs. Actual Values, R = " + str(round(r_value,3)) + ", Epoch: " + str(epoch+1) , fontsize=18, fontweight='bold')
            g.fig.tight_layout()
            g.fig.subplots_adjust(top=0.95)
            g.ax_joint.text(0.4,0.6,"", fontsize=12)
            g.ax_marg_x.set_axis_off()
            g.ax_marg_y.set_axis_off()
            g.ax_joint.set_xlabel('Actual Values',fontsize=18 ,fontweight='bold')
            g.ax_joint.set_ylabel('Predictions',fontsize=18 ,fontweight='bold')
            g.savefig(results_sub_folder / (output_file_header + "epoch_" + str(epoch+1)) )






