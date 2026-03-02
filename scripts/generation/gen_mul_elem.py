import os, sys
from os.path import join
# sys.path.append('../../')
from config_scigen import hydra_dir, job_dir

############
model_path = join(hydra_dir, job_dir)
dataset = 'mp_20'
batch_size = 50 # Number of materials to generate in one batch
num_batches_to_samples = 10 # Number of batches to sample
num_materials = batch_size * num_batches_to_samples
save_traj_idx = []  # List of indices to save trajectory
num_run = 1 # Number of runs
idx_start = 0   # Starting index
header = 'sc'   # Header for the label
sc_list = ['kag']   # List of SCs to generate
# atom_list = ['Mn', 'Fe', 'Co', 'Ni', 'Ru', 'Nd', 'Gd', 'Tb', 'Dy', 'Yb', 'Cu']
atom_list = ['Mn', 'Fe', 'Co', 'Ni', 'Cu']
t_mask = True   # Use mask for atom type
frac_z = 0.5   # Fraction of z-axis for mask. If None, frac_z is radomly selected in [0, 1).
save_cif = True # Whether to save CIF files
###################

sc_natm_range = {   # max number of atoms in the unit cell
    'tri': [1, 4],    # 1 atom for triangle
    'hon': [1, 8],      # 2 atom for honeycomb
    'kag': [1, 12],    # 3 atom for kagome
    'sqr': [1, 4],    # 1 atom for square
    'elt': [1, 8],    # 2 atom for elongated triangle
    'sns': [1, 16],    # 4 atom for snub square
    'tsq': [1, 16],    # 4 atom for truncated square
    'srt': [1, 20],    # 6 atom for small rhombitrihexagonal
    'snh': [1, 20],    # 6 atom for snub hexagonal
    'trh': [1, 20],    # 6 atom for 
    'grt': [1, 30],    # 12 atom
    'lieb': [1, 12],    # 3 atom
    # 'pyc': [24, 24],    # 16 atom for Pyrochlore (cubic)
    # 'pyc': [23, 40],    # 16 atom for Pyrochlore (cub
    'pyc': [32, 32],    # 16 atom for Pyrochlore (cubic)
    'hkg': [18, 30],    # 6 atom for Hyper-kagome
    'kah': [1, 12],    # 5 atom for kagomePyrochlore
    'van': [1, 20],   # Vanilla DiffCSP (no constraint)
}

# Handle c_scale argument conditionally
t_mask_arg = f'--t_mask {t_mask}'

for i, sc in enumerate(sc_list):
    for atype in atom_list:
        atom_list_ = [atype]
        for j in range(idx_start, idx_start + num_run):
            tag = format(j, '03d')
            
            # Handle save_traj argument: add `--save_traj` if the current index is in save_traj_idx
            save_traj_arg = '--save_traj True' if j in save_traj_idx else ''

            label = f"{header+'_' if len(header) > 0 else ''}{sc}{atype}{num_materials}_{tag}"
            natm_range = [str(i) for i in sc_natm_range[sc]]

            # Construct the command string
            job_command = f'python script/generation.py --model_path {model_path} \
                        --dataset {dataset} --label {label} --sc {sc} \
                        --batch_size {batch_size} --num_batches_to_samples {num_batches_to_samples}   \
                        --natm_range {" ".join(natm_range)} {save_traj_arg}   \
                        --known_species {" ".join(atom_list_)}   \
                        --frac_z {frac_z}   \
                        {t_mask_arg}'

            print([i, j], job_command)
            os.system(job_command)
            print([i, j], label, 'done')
            if save_cif:
                save_cif_command = f'python script/save_cif.py --job_dir {job_dir} --label {label}'
                os.system(save_cif_command)
            print()