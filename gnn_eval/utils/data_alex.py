
import os
import pandas as pd
import pickle as pkl
from pymatgen.core.structure import Structure
from pymatgen.io.ase import AseAtomsAdaptor
from utils.data import str2pmg

def load_alex(data_dir, csv_header, idxs_data, save_path=None):
    dfs = []
    for idx in idxs_data:
        try: 
            file_name = f"{csv_header}_{format(idx, '03d')}"
            csv_path = os.path.join(data_dir, f'{file_name}.csv')
            # load df_ from csv_path
            df_ = pd.read_csv(csv_path)
            dfs.append(df_)

        except Exception as e:
            print(f"Error in loading {file_name}: {e}")

    df = pd.concat(dfs, ignore_index=True)

    new_rows = []  # To collect new rows
    for index, row in df.iterrows():
        row1 = {}  # Initialize the dictionary here
        row1['mpid'] = row['material_id']
        pstruct = str2pmg(row['cif'])
        astruct = AseAtomsAdaptor().get_atoms(pstruct)
        row1['structure'] = astruct
        row1['f_energy'] = row['formation_energy_per_atom']
        row1['ehull'] = row['e_above_hull']
        for k in ['band_gap_ind','band_gap_dir']:
            row1[k] = row[k]
        new_rows.append(row1)

    data = pd.DataFrame(new_rows).reset_index(drop=True)
    if save_path is not None:
        with open(save_path, 'wb') as f:
            pkl.dump(data, f)
    
    return data