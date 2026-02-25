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
from config import *

random.seed(seedn)

#%%
data_dir = os.path.join(home_dir, 'data/alex_2d_01')
idxs = [0,1]
data = {'entries': []}
for idx in idxs:
    json_file = f"alexandria_2d_{format(idx, '03d')}.json"
    json_path = os.path.join(data_dir, json_file)
    if not os.path.exists(json_path):
        print(f"{json_path} does not exist, so download from the source.")
        url = f'https://alexandria.icams.rub.de/data/pbe_2d/{json_file}.bz2'
        # Download the file and suppress output
        subprocess.run(["wget", url, "-P", data_dir], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        # Unzip the file and suppress output
        subprocess.run(["bzip2", "-d", f"{json_path}.bz2"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    data_ = json.load(open(json_path, 'r'))
    data['entries'].extend(data_['entries'])
    print(f"Loaded {len(data['entries'])} entries from {json_file}, and now the total number of entries is {len(data_['entries'])}")

# %%
idx = 11
struct_dict = data['entries'][idx]['structure']
pstruct = Structure.from_dict(struct_dict)
fig, ax = plt.subplots(1,1,figsize=(6,5))
ax = vis_structure(pstruct, ax=ax, supercell=np.diag([3,3,1]), title=None, rot='5x,5y,90z', savedir=None)

#%%
# List to store all the structures
pstruct_list = []

# Iterate over each entry and load the structure
for entry in data['entries']:
    structure_dict = entry['structure']  # Extract the structure sub-dictionary
    pstruct = Structure.from_dict(structure_dict)  # Load the structure
    pstruct_list.append(pstruct)  # Store the structure in the list

#%%
idx = 4 
pstruct = pstruct_list[idx]
cart = pstruct.cart_coords
frac = pstruct.frac_coords  
lattice = pstruct.lattice.matrix
species = pstruct.species

# height of the lattice (divide the volume of the unit cell by ab-plane area)
volume = np.linalg.det(lattice)
ab_area = np.linalg.norm(np.cross(lattice[0], lattice[1]))
lat_height = volume / ab_area

cart_z_range = np.ptp(cart[:,2])
z_ratio = lat_height/cart_z_range

fig, ax = plt.subplots(1,1,figsize=(6,5))
ax = vis_structure(pstruct, ax=ax, supercell=np.diag([3,3,1]), title=None, rot='5x,5y,90z', savedir=None)


#%%
# plot the histogram of the number of atoms per unit cell 
natoms = []
for pstruct in pstruct_list:
    natoms.append(len(pstruct))
fig, ax = plt.subplots(1,1,figsize=(8,6))
ax.hist(natoms, bins=max(natoms), color='b', alpha=0.7)
ax.set_xlabel(f'Number of atoms in the structure')
ax.set_title(f"total {len(natoms)} materials")


#%%
# plot the histogram of the distribution of z_ratios
z_ratios = []
lat_heights = []
cart_z_ranges = []
alphas, betas, gammas = [], [], []
bottoms, tops, b_t_ratios = [], [], []
for pstruct in pstruct_list[:500]:
    cart = pstruct.cart_coords
    frac = pstruct.frac_coords  
    lattice = pstruct.lattice.matrix
    lattice_angles = pstruct.lattice.angles
    alphas.append(lattice_angles[0])
    betas.append(lattice_angles[1])
    gammas.append(lattice_angles[2])
    species = pstruct.species

    # height of the lattice (divide the volume of the unit cell by ab-plane area)
    volume = np.linalg.det(lattice)
    ab_area = np.linalg.norm(np.cross(lattice[0], lattice[1]))
    lat_height = volume / ab_area

    cart_z_range = np.ptp(cart[:,2])
    cart_z_min, cart_z_max = np.min(cart[:,2]), np.max(cart[:,2])
    bottom, top = cart_z_min, lat_height - cart_z_max
    b_t_ratio = bottom / top
    
    z_ratio = cart_z_range / lat_height
    z_ratios.append(z_ratio)
    lat_heights.append(lat_height)
    cart_z_ranges.append(cart_z_range)
    bottoms.append(bottom)
    tops.append(top)
    b_t_ratios.append(b_t_ratio)

fig, axs = plt.subplots(1,2,figsize=(16,6))
ax0  = axs[0]
ax0.hist(z_ratios, bins=50, color='b', alpha=0.7)
ax0.set_xlabel('z_ratio')
ax0.set_title(f"z_ratio distribution")

ax1 = axs[1]
# plot the histogram of alpbas, betas, gammas, by stacking them
ax1.hist([alphas, betas, gammas], bins=50, color=['r', 'g', 'b'], alpha=0.7, label=['alpha', 'beta', 'gamma'])
ax1.set_xlabel('angle (degree)')
ax1.set_title(f"lattice angles distribution")
ax1.legend()


fig, ax = plt.subplots(1,1,figsize=(6,6))
ax.hist(b_t_ratios, bins=50, color='b', alpha=0.7)
ax.set_xlabel('bottom/top ratio')
ax.set_title(f"bottom/top ratio distribution")


fig, axs = plt.subplots(1,3,figsize=(15,6))
value_list = [lat_heights, cart_z_ranges, z_ratios]
label_list = ['lat_heights', 'cart_z_ranges', 'z_ratios']
combination_list = [[0,1], [0,2], [1,2]]
for i, (label, combination) in enumerate(zip(label_list, combination_list)):
    ax = axs[i]
    v1, v2 = value_list[combination[0]], value_list[combination[1]]
    ax.scatter(v1, v2, s=5)
    ax.set_xlabel(label_list[combination[0]])
    ax.set_ylabel(label_list[combination[1]])

#%%
# find the argmin of the z_ratios
min_idx = np.argmin(z_ratios)
pstruct = pstruct_list[min_idx]
fig, ax = plt.subplots(1,1,figsize=(6,5))
ax = vis_structure(pstruct, ax=ax, supercell=np.diag([1,1,1]), title=None, rot='5x,5y,90z', savedir=None)


#%%
# plot other properties 
keys_data = ['e_form', 'band_gap_ind', 'band_gap_dir', 'e_above_hull', 'spg']
prop_dict = {key: [] for key in keys_data}
prop_dict['elements'] = {el:[] for el in chemical_symbols}
for i, entry in enumerate(data['entries']):
    data_ = entry['data']
    for key in keys_data:
        prop_dict[key].append(data_[key])
    elements = entry['data']['elements']
    mat_id = entry['data']['mat_id']
    
    for el in elements:
        prop_dict['elements'][el].append(mat_id)
#%%
for k in keys_data:
    fig, ax = plt.subplots(1,1,figsize=(8,6))
    ax.hist(prop_dict[k], bins=50, color='b', alpha=0.7)
    ax.set_xlabel(f'{k}')
    ax.set_ylabel(f'count')
    ax.set_title(f"[{k}] total {len(prop_dict[k])} materials")

#%%
# plot bar chart for the elements
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
fig.suptitle(f"total {len(elem_exist)} elements from {len(elem_dict)} materials")
print(f"{len(elem_no_exist)} elements do not exist in the dataset: {sorted(elem_no_exist)}")
    
# %%
