import os
from config_scigen import *

############
# Must match the parameters used in gen_mul_elem.py so that labels resolve correctly.
batch_size = 50
num_batches_to_samples = 10
num_materials = batch_size * num_batches_to_samples   # 500
num_run = 1
idx_start = 0
header = 'sc'
sc_list = ['lieb']
# atom_list = ['Mn', 'Fe', 'Co', 'Ni', 'Ru', 'Nd', 'Gd', 'Tb', 'Dy', 'Yb', 'Cu']
atom_list = ['Mn', 'Fe', 'Co', 'Ni', 'Cu']
gen_cif = True
screen_mag = False
###################

for i, sc in enumerate(sc_list):
    for atype in atom_list:
        for j in range(idx_start, idx_start + num_run):
            tag = format(j, '03d')
            label = f"{header+'_' if len(header) > 0 else ''}{sc}{atype}{num_materials}_{tag}"
            job_command = f'python script/eval_screen.py  --label {label} --gen_cif {gen_cif} --screen_mag {screen_mag}'
            print([i, j, atype], job_command)
            os.system(job_command)
            print([i, j, atype], label, 'done')
            print()
