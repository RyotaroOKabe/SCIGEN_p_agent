#%%
import os 
from os.path import join
import sys
import subprocess
import numpy as np
import torch
import json
import random
import math
import csv
import pandas as pd
from tqdm import tqdm
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split

sys.path.append('../')

from utils.data import *
from utils.common import *
from utils.materials import vis_structure, chemical_symbols
from config_data import alex_config
from config_scigen import *

random.seed(seedn)

#%%
alex_type = '2d'
alex_name = f"alex_{alex_type}"
alex_pbe = alex_config['pbe'][alex_type]
alex_url_tag = alex_config['url_tag'][alex_type]
data_dir = os.path.join(home_dir, f'data/{alex_name}')
save_csv = True
keys_dict = {'material_id': 'mat_id', 'formation_energy_per_atom': 'e_form', 'band_gap_ind': 'band_gap_ind', 'band_gap_dir': 'band_gap_dir', 'e_above_hull':'e_above_hull', 'pretty_formula': 'formula', 'spacegroup.number':'spg', 'spacegroup.number.conv':'spg', 'natm': 'nsites'}
threshold_lower_dict = {'formation_energy_per_atom': 2.0, 'e_above_hull': 0.3, 'natm': 41}  #TODO: adjust this
fig_dir = data_dir 
header = f"{alex_name}_{alex_pbe}"
idxs = range(alex_config['num_idx'][alex_type])
dfs = []
for idx in idxs:
    try: 
        file_name = f"alexandria{alex_url_tag}_{format(idx, '03d')}"
        csv_path = os.path.join(data_dir, f'{file_name}.csv')
        # load df_ from csv_path
        df_ = pd.read_csv(csv_path)
        dfs.append(df_)
        print(f"Loaded {file_name}")

    except Exception as e:
        print(f"Error in loading {file_name}: {e}")

df = pd.concat(dfs, ignore_index=True)


# %%
# apply filter to the dataframe
print(f"threshold_lower_dict: {threshold_lower_dict}")

df_filtered = df.copy()
for k, v in threshold_lower_dict.items():
    df_filtered = df_filtered[df_filtered[k] < v]

df_filtered = df_filtered.reset_index(drop=True)

print(f"Before filtering: {len(df)}")
print(f"After filtering: {len(df_filtered)}")

#%%
# split the data into train, val, test sets
num_total = None#   1000
if num_total is not None:
    n = num_total if num_total < len(df_filtered) else len(df_filtered)
else:
    n = len(df_filtered)
n_train = int(n * 0.8)
n_val = int(n * 0.1)
n_test = n - n_train - n_val

# sample n rows from the dataframe at random
df_sample = df_filtered.sample(n=n, random_state=seedn).reset_index(drop=True)
df_train, df_test = train_test_split(df_sample, test_size=n_test, random_state=seedn)
df_train, df_val = train_test_split(df_train, test_size=n_val, random_state=seedn)
print(f"Train: {len(df_train)}, Val: {len(df_val)}, Test: {len(df_test)}")

if save_csv:
    # save the dataframes
    df_train.to_csv(os.path.join(data_dir, 'train.csv'), index=False)
    df_val.to_csv(os.path.join(data_dir, 'val.csv'), index=False)
    df_test.to_csv(os.path.join(data_dir, 'test.csv'), index=False)

    # save the data configuaration as a json file
    save_config = {}
    save_config['alex_type'] = alex_type
    save_config['alex_pbe'] = alex_pbe
    save_config['alex_url_tag'] = alex_url_tag
    save_config['alex_name'] = alex_name
    save_config['num_total'] = num_total
    save_config['seedn'] = seedn
    save_config['train'] = len(df_train)
    save_config['val'] = len(df_val)
    save_config['test'] = len(df_test)
    save_config['total'] = n
    save_config['total_original'] = len(df)
    save_config['total_filtered'] = len(df_filtered)
    save_config['threshold_lower_dict'] = threshold_lower_dict
    # save the configuration as a json file
    config_path = os.path.join(data_dir, 'data_config.json')
    with open(config_path, 'w') as f:
        json.dump(save_config, f)

#%%
# visualize the distribution of the filtered data
# plot other properties 
keys_data = ['e_form', 'band_gap_ind', 'band_gap_dir', 'e_above_hull', 'spg']
keys_dict_analyze = {'formation_energy_per_atom': 'e_form', 'band_gap_ind': 'band_gap_ind', 'band_gap_dir': 'band_gap_dir', 'e_above_hull':'e_above_hull', 'spacegroup.number.conv':'spg', 'natm': 'nsites'}
prop_dict = {key: [] for key in keys_dict_analyze.values()}
prop_dict['elements'] = {el:[] for el in chemical_symbols}
for i, row in enumerate(df_sample.iterrows()):
    for k, v in keys_dict_analyze.items():
        prop_dict[v].append(row[1][k])
    elements = row[1]['elements']
    # now the elements is a string, convert it to a list
    elements = elements.replace("'", "").replace("[", "").replace("]", "").replace(" ", "").split(',')
    mat_id = row[1]['material_id']
    
    for el in elements:
        prop_dict['elements'][el].append(mat_id)
#%%
for k in keys_data:
    fig, ax = plt.subplots(1,1,figsize=(8,6))
    ax.hist(prop_dict[k], bins=50, color='b', alpha=0.7)
    ax.set_xlabel(f'{k}')
    ax.set_ylabel(f'count')
    ax.set_title(f"[{k}] total {len(prop_dict[k])} materials")
    fig.savefig(join(fig_dir, f'{header}_{k}_hist.png'))
    
#%%
fig, axs = plt.subplots(2,1,figsize=(14,7))
elem_dict = prop_dict['elements']
elem_exist = {}
elem_no_exist = []
max_num = 0
for el in chemical_symbols:
    el_num = len(elem_dict[el])
    if el_num > 0:
        elem_exist[el] = el_num
    else: 
        elem_no_exist.append(el)
    if el_num > max_num:
        max_num = el_num
# plot ax.bar(elem_exist.keys(), elem_exist.values()), but could you plot them in 2 rows?
for i, ax in enumerate(axs):
    if i == 0:
        keys = sorted(list(elem_exist.keys())[:len(elem_exist)//2], key=lambda x: chemical_symbols.index(x))
        values = [elem_exist[k] for k in keys]
    else:
        keys = sorted(list(elem_exist.keys())[len(elem_exist)//2:], key=lambda x: chemical_symbols.index(x))
        values = [elem_exist[k] for k in keys]
    ax.bar(keys, values)
    ax.set_ylim(0, max_num*1.1)
    ax.set_xlabel('element')
    ax.set_ylabel('count')
fig.suptitle(f"total {len(elem_exist)} elements out of {len(elem_dict)}")
fig.savefig(join(fig_dir, f'{header}_elements_hist.png'))
print(f"{len(elem_no_exist)} elements do not exist in the dataset: {sorted(elem_no_exist)}")
 





# %%
