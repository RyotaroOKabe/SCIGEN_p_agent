import os
from os.path import join
import csv
import sys
import json
import torch
from pymatgen.core import Structure, Lattice
from torch import nn
# Local imports
from config_scigen import gnn_eval_path, home_dir, hydra_dir, job_dir, out_name, stab_pred_name_A, stab_pred_name_B, mag_pred_name
sys.path.append(gnn_eval_path)
from gnn_eval.utils.record import log_buffer, logger
from script.mat_utils import *
from script.eval_funcs import load_model, load_model_mag, parse_arguments, process_data, load_df, classify_stability, generate_cif_files
from script.periodic_trends import plotter

torch.set_default_dtype(torch.float64)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
save_dir = os.path.join(home_dir, 'figures')

def give_use_name(header, elem, tail):
    return f"{header}{elem}{tail}"

def save_dict_as_csv(dictionary, filename):
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        # Write the header row, if needed
        # writer.writerow(['Element', 'Value'])
        for key, value in dictionary.items():
            writer.writerow([key, value])

def main():
    args = parse_arguments(job_dir, out_name)
    # model_names = [stab_pred_name_A, stab_pred_name_B]  #TODO: Adjust the model names if needed
    # if args.screen_mag:
    #     model_names.append(mag_pred_name)
    jobdir = join(hydra_dir, args.job_dir)
    # use_name_ = f"gen_{args.label}" if args.label else "gen"
    header = 'gen_sc_pyc'
    all_cifs = False
    gen_cif = True
    tail = '2000_002'
    eval_code = 'eval_screen_pt.py'
    target_elems = ['Cu', 'Co', 'Ni', 'Sc', 'Ti', 'V', 'Cr', 'Ce',' Nd', 'Sm', 'Yb', 'Tm']

    for elem in target_elems:
        elem_cif_dict = {el: [] for el in chemical_symbols[1:]}
        try: 
            use_name = give_use_name(header, elem, tail)
            
            # use_path = join(jobdir, f"eval_{use_name}.pt")
            # folder_tag = '_filtered' if not args.screen_mag else '_mag'
            folder_tag = '_smact' if 'pt.' in eval_code else '_filtered'
            cif_folder = use_name + folder_tag
            label = use_name.replace('gen_', '')
            if all_cifs:
                # folder_tag = ''
                # label = use_name.replace('gen_', '')
                if gen_cif:
                    os.system(f"python script/save_cif.py --label {label} --job_dir {args.job_dir}")
                cif_folder = 'eval_gen_' + label
            else: 
                if gen_cif:
                    os.system(f"python script/{eval_code} --label {label} --job_dir {args.job_dir}")
            
            # cif_dir= join(jobdir, use_name + folder_tag)
            cif_dir= join(jobdir, cif_folder)

            # list of cif files in the cif_dir
            cif_files = [f for f in os.listdir(cif_dir) if f.endswith('.cif')]
            cif_files.sort()
            logger.info(f"Found {len(cif_files)} CIF files in {cif_dir}.")
            # load the cif files as a list of pymatgen structures
            for i, cif_file in enumerate(cif_files):
                pstruct = Structure.from_file(join(cif_dir, cif_file))
                # get chemical symbols (without duplicates) from the structure
                elems = list(set([str(el) for el in pstruct.species]))
                # print(f"elems: {elems}")
                # drop 'elem' from the list
                elems.remove(elem)
                
                for elem_ in elems:
                    elem_cif_dict[elem_].append(cif_file.replace('.cif', ''))

            # get the dictio0nary of the count of each element i chemical_symbols
            elem_count_dict = {elem: len(files) for elem, files in elem_cif_dict.items()}
            print(f"elem_count_dict: {elem_count_dict}")
            # save the dictionary as a csv file
            csv_count_file = join(cif_dir, f"{use_name}_elem_count.csv")
            save_dict_as_csv(elem_count_dict, csv_count_file)
            print(f"Saved the count of each element in {csv_count_file}")
            # plot the distribution of the elements on the periodic table
            p = plotter(csv_count_file, output_filename=csv_count_file.replace('.csv', '.html'), under_value=0, over_value=max(elem_count_dict.values()))
        except Exception as e:
            logger.error(f"Error in processing {elem}: {e}")
            continue
if __name__ == "__main__":
    main()
