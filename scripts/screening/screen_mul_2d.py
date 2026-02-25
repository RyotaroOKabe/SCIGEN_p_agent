import os
from config_scigen_2d import *

num_materials = 10000
num_run =3
idx_start = 5
header = 'sc'  
arch_list = ['sns', 'tsq', 'srt', 'snh', 'trh', 'van'] # 'sns',   # ['van', 'tri', 'hon', 'kag', 'sqr'] 
gen_cif = True
screen_mag = False

for j in range(idx_start, idx_start+num_run):
    for i, arch in enumerate(arch_list):
        tag = format(j, '03d')
        label = f'{header}_{arch}{num_materials}_{tag}'
        job_command = f'python script/eval_screen_2d.py  --label {label} --gen_cif {gen_cif} --screen_mag {screen_mag}'
        print([i, j], job_command)
        os.system(job_command)
        print([i, j], label, 'done')
        print()

