#%%
import os 
from os.path import join
import sys
import numpy as np
import torch
import json
import random
import math
import csv
import pandas as pd
from tqdm import tqdm

sys.path.append('../')

from utils.data import *
from config_scigen import *

#%%

csv_file = os.path.join(home_dir, 'data/mp_20/test.csv')
# load df from csv
df = pd.read_csv(csv_file)
df_keys = df.keys()
# for k in df_keys:
#     print(k, type(df.iloc[0][k]))
print(df.info())
print(df.head())

#%%
# check if df is loaded correctly
idx = 0
row = df.iloc[idx]
cif = row['cif']
print(cif)



#%%
