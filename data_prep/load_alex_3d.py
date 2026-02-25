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

sys.path.append('../')

from utils.data import *
from utils.common import *
from utils.materials import vis_structure
from config_scigen import *

random.seed(seedn)

#%%
data_dir = os.path.join(home_dir, 'data/alex_3d')
keys_dict = {'material_id': 'mat_id', 'formation_energy_per_atom': 'e_form', 'band_gap_ind': 'band_gap_ind', 'band_gap_dir': 'band_gap_dir', 'e_above_hull':'e_above_hull', 'pretty_formula': 'formula', 'spacegroup.number':'spg', 'spacegroup.number.conv':'spg', 'natm': 'nsites'}
threshold_lower_dict = {'formation_energy_per_atom': 2.0, 'e_above_hull': 0.08, 'natm': 21}  #TODO: adjust this

idxs = range(10)
data0 = {'entries': []}
for idx in idxs:
    try: 
        json_file = f"alexandria_{format(idx, '03d')}.json"
        json_path = os.path.join(data_dir, json_file)
        if not os.path.exists(json_path):
            print(f"{json_path} does not exist, so download from the source.")
            url = f'https://alexandria.icams.rub.de/data/pbe/{json_file}.bz2'
            # Download the file and suppress output
            subprocess.run(["wget", url, "-P", data_dir], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            # Unzip the file and suppress output
            subprocess.run(["bzip2", "-d", f"{json_path}.bz2"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        data_ = json.load(open(json_path, 'r'))
        # save the data only if it satisfies the threshold
        # load_entry = True
        # for k, v in threshold_lower_dict.items():
        #     if data_['entries'][0]['data'][k] >= v:
        #         load_entry = False
        #         print(f"Skip {json_file} because {k} is {data_['entries'][0]['data'][k]}")
        #         break
        data0['entries'].extend(data_['entries'])
        print(f"Loaded {len(data0['entries'])} entries from {json_file}, and now the total number of entries is {len(data_['entries'])}")
    except Exception as e:
        print(f"Error in loading {json_file}: {e}")

#%%
# filter the data by the threshold
data = {'entries': []}
for i, entry in tqdm(enumerate(data0['entries']), total=len(data0['entries'])):
    load_entry = True
    for j, (k, v) in enumerate(threshold_lower_dict.items()):
        key = keys_dict[k]
        # print(f"({i}) entry['data'][{key}]: {entry['data'][key]}")
        if entry['data'][key] >= v:
            load_entry = False
            break
    if load_entry:
        data['entries'].append(entry)
print(f"Filter the data by the threshold: {threshold_lower_dict}")
print(f"Before filtering: {len(data0['entries'])}, After filtering: {len(data['entries'])}")

# %%
# check individual material data
idx = 11
struct_dict = data['entries'][idx]['structure']
pstruct = Structure.from_dict(struct_dict)
fig, ax = plt.subplots(1,1,figsize=(6,5))
ax = vis_structure(pstruct, ax=ax, supercell=np.diag([1,1,1]), title=None, rot='5x,5y,90z', savedir=None)

# visualize dictionary
visualize_dict_structure(data['entries'][idx])


#%%
# select materials to load as f 
# threshold_lower_dict = {'eform': 0.1, 'ehull': 0.1, 'natm': 21}
df = pd.DataFrame()
num = len(data['entries'])
for i, entry in tqdm(enumerate(data['entries']), total=len(data['entries'])):
    data_ = entry['data']
    for k, v in keys_dict.items():
        df.loc[i, k] = data_[v]

    for sg in ['spacegroup.number.conv', 'spacegroup.number']:
        df.loc[i, sg]  = int(df.loc[i, sg])

    df.loc[i, 'elements'] = elem_string(entry['data']['elements'])
    
    pstruct = Structure.from_dict(entry['structure'])
    cif_str = pstruct.to(fmt='cif')
    df.loc[i, 'cif'] = cif_str

    # symmetrized (with spacegroup) structure
    try: 
        sga = SpacegroupAnalyzer(pstruct, symprec=0.1)
        pstruct_sga = sga.get_symmetrized_structure()
    except Exception as e:
        print(f"Error in getting symmetrized structure for {i}-th row ({df.loc[i, 'material_id']}) : {e}")
        sga = SpacegroupAnalyzer(pstruct, symprec=0.05)
        pstruct_sga = sga.get_symmetrized_structure()

    df.loc[i, 'cif.conv'] = pstruct_sga.to(fmt='cif')
    df.loc[i, 'natm'] = len(pstruct)

# %%
# apply filter to the dataframe
print(f"threshold_lower_dict: {threshold_lower_dict}")

df_filtered = df
for k, v in threshold_lower_dict.items():
    df_filtered = df_filtered[df_filtered[k] < v]

df_filtered = df_filtered.reset_index(drop=True)

print(f"Before filtering: {len(df)}")
print(f"After filtering: {len(df_filtered)}")

#%%
# split the data into train, val, test sets
n = len(df_filtered)
n_train = int(n * 0.8)
n_val = int(n * 0.1)
n_test = n - n_train - n_val

from sklearn.model_selection import train_test_split
df_train, df_test = train_test_split(df_filtered, test_size=n_test, random_state=seedn)
df_train, df_val = train_test_split(df_train, test_size=n_val, random_state=seedn)
print(f"Train: {len(df_train)}, Val: {len(df_val)}, Test: {len(df_test)}")

# save the dataframes
df_train.to_csv(os.path.join(data_dir, 'train.csv'), index=False)
df_val.to_csv(os.path.join(data_dir, 'val.csv'), index=False)
df_test.to_csv(os.path.join(data_dir, 'test.csv'), index=False)

# %%
