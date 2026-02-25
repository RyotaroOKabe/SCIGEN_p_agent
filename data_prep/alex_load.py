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
from config_data import alex_config
from config_scigen import *

random.seed(seedn)

#%%
alex_type = '2d'
skip_csv_exist = True
alex_name = f"alex_{alex_type}"
alex_pbe = alex_config['pbe'][alex_type]
alex_url_tag = alex_config['url_tag'][alex_type]
data_dir = os.path.join(home_dir, f'data/{alex_name}')
keys_dict = {'material_id': 'mat_id', 'formation_energy_per_atom': 'e_form', 'band_gap_ind': 'band_gap_ind', 'band_gap_dir': 'band_gap_dir', 'e_above_hull':'e_above_hull', 'pretty_formula': 'formula', 'spacegroup.number':'spg', 'spacegroup.number.conv':'spg', 'natm': 'nsites'}
idxs = range(alex_config['num_idx'][alex_type])
data0 = {'entries': []}
for idx in idxs:
    try: 
        file_name = f"alexandria{alex_url_tag}_{format(idx, '03d')}"
        print(f"Loading {file_name}")
        json_file = f"{file_name}.json"
        json_path = os.path.join(data_dir, json_file)
        if not os.path.exists(json_path):
            print(f"{json_path} does not exist, so download from the source.")
            url = f'https://alexandria.icams.rub.de/data/{alex_pbe}/{json_file}.bz2'
            # Download the file and suppress output
            subprocess.run(["wget", url, "-P", data_dir], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            # Unzip the file and suppress output
            subprocess.run(["bzip2", "-d", f"{json_path}.bz2"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        data_ = json.load(open(json_path, 'r'))

        csv_path = os.path.join(data_dir, f'{file_name}.csv')
        
        if os.path.exists(csv_path) and skip_csv_exist:
            print(f"{csv_path} already exists, so skip loading.")
            continue
        df_ = pd.DataFrame()
        num = len(data_['entries'])
        for i, entry in tqdm(enumerate(data_['entries']), total=len(data_['entries'])):
            data_ = entry['data']
            for k, v in keys_dict.items():
                df_.loc[i, k] = data_[v]

            for sg in ['spacegroup.number.conv', 'spacegroup.number']:
                df_.loc[i, sg]  = int(df_.loc[i, sg])

            df_.loc[i, 'elements'] = elem_string(entry['data']['elements'])
            
            pstruct = Structure.from_dict(entry['structure'])
            cif_str = pstruct.to(fmt='cif')
            df_.loc[i, 'cif'] = cif_str

            # symmetrized (with spacegroup) structure
            try: 
                sga = SpacegroupAnalyzer(pstruct, symprec=0.1)
                pstruct_sga = sga.get_symmetrized_structure()
            except Exception as e:
                print(f"Error in getting symmetrized structure for {i}-th row ({df_.loc[i, 'material_id']}) : {e}")
                sga = SpacegroupAnalyzer(pstruct, symprec=0.05)
                pstruct_sga = sga.get_symmetrized_structure()

            df_.loc[i, 'cif.conv'] = pstruct_sga.to(fmt='cif')
            df_.loc[i, 'natm'] = len(pstruct)
    

        df_.to_csv(csv_path, index=False)
        print(f"Saved {len(data_['entries'])} entries from {json_file} to {file_name}.csv")
        del df_
        del data_
        # data0['entries'].extend(data_['entries'])
        # print(f"Loaded {len(data0['entries'])} entries from {json_file}, and now the total number of entries is {len(data_['entries'])}")
    except Exception as e:
        print(f"Error in loading {json_file}: {e}")



# %%
