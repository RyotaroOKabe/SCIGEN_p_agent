import os
import pandas as pd
import pickle as pkl
from pymatgen.core.structure import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

def load_cif_files_to_dataframe(folder_path):
    data = {
        'structure': [],
        'lattice_lens': [],
        'lattice_angs': [],
        'sg': []
    }

    for filename in os.listdir(folder_path):
        if filename.endswith('.cif'):
            try: 
                cif_path = os.path.join(folder_path, filename)
                structure = Structure.from_file(cif_path)
                analyzer = SpacegroupAnalyzer(structure)
                space_group = analyzer.get_space_group_number()

                lattice = structure.lattice
                lattice_lengths = lattice.lengths
                lattice_angles = lattice.angles

                data['structure'].append(structure)
                data['lattice_lens'].append(lattice_lengths)
                data['lattice_angs'].append(lattice_angles)
                data['sg'].append(space_group)
            except Exception as e:
                print(filename, e)
    df = pd.DataFrame(data)
    return df


def elem_string(elements):
    el_str = "["
    for el in elements:
        el_str += f"'{el}', "
    el_str = el_str[:-2] + "]"
    return el_str

