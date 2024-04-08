# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 14:19:00 2024

@author: anton
"""
import numpy as np
import yaml
import csv
import pandas as pd

#%% importing data time stamps for p1, and assigning to variables
with open('Participant1.yaml', 'r') as yaml_file:
    p1_ts = yaml.safe_load(yaml_file)

for key, value in p1_ts.items():
    variable_name = f"p1_{key}"  # Prepend "p1_" to the variable name
    globals()[variable_name] = value


#%% extracting emg data for p1, and assigning to variables according to muscles
p1_data=np.array(pd.read_csv('participant1.csv', header=None))
QL=p1_data[:,0]
HamL=p1_data[:,1]
HamR=p1_data[:,2]
QR=p1_data[:,3]
TibL=p1_data[:,4]
TibR=p1_data[:,5]
GasL=p1_data[:,6]
GasR=p1_data[:,7]
#%% extracting activities from p1
p1_cartoon=np.zeros([p1_Cartoon_T2-p1_Cartoon_T1,8])
p1_floor_fast=np.zeros([p1_Floor_fast_T2-p1_Floor_fast_T1,8])
p1_floor_run=np.zeros([p1_Floor_run_T2-p1_Floor_run_T1,8])
p1_floor_self=np.zeros([p1_Floor_self_T2-p1_Floor_self_T1,8])

for i in range(8):
    p1_cartoon[:,i]=p1_data[p1_Cartoon_T1:p1_Cartoon_T2,i]
    p1_floor_fast[:,i]=p1_data[p1_Floor_fast_T1:p1_Floor_fast_T2,i]   
    p1_floor_run[:,i]=p1_data[p1_Floor_run_T1:p1_Floor_run_T2,i]
    p1_floor_self[:,i]=p1_data[p1_Floor_self_T1:p1_Floor_self_T2,i]
