#%%
import numpy as np
import matplotlib.pyplot as plt
import os
from os import listdir
from os.path import isfile, join
from os.path import join as opj
import pandas as pd
from tqdm import tqdm
import pickle as pkl
from copy import copy

import matplotlib as mpl
import matplotlib.pyplot as plt
from copy import copy
import imageio
from tqdm import tqdm

from pymatgen.core.structure import Structure
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.io.cif import CifWriter

from ase import Atoms
from ase.visualize.plot import plot_atoms
from ase.build import make_supercell
from ase.io import read

from utils.common import palette, sub   

bar_format = '{l_bar}{bar:10}{r_bar}{bar:-10b}'

chemical_symbols = [
    # 0
    'X',
    # 1
    'H', 'He',
    # 2
    'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne',
    # 3
    'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar',
    # 4
    'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
    'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr',
    # 5
    'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd',
    'In', 'Sn', 'Sb', 'Te', 'I', 'Xe',
    # 6
    'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy',
    'Ho', 'Er', 'Tm', 'Yb', 'Lu',
    'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi',
    'Po', 'At', 'Rn',
    # 7
    'Fr', 'Ra', 'Ac', 'Th', 'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk',
    'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr',
    'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds', 'Rg', 'Cn', 'Nh', 'Fl', 'Mc',
    'Lv', 'Ts', 'Og']

def vis_structure(struct_in, ax=None, supercell=np.diag([1,1,1]), title=None, rot='5x,5y,90z', savedir=None, palette=palette):
    if type(struct_in)==Structure:
        struct = Atoms(list(map(lambda x: x.symbol, struct_in.species)) , # list of symbols got from pymatgen
                positions=struct_in.cart_coords.copy(),
                cell=struct_in.lattice.matrix.copy(), pbc=True) 
    elif type(struct_in)==Atoms:
        struct=struct_in
    struct = make_supercell(struct, supercell)
    symbols = np.unique(list(struct.symbols))
    len_symbs = len(list(struct.symbols))
    z = dict(zip(symbols, range(len(symbols))))

    if ax is None:
        fig, ax = plt.subplots(figsize=(6,5))
        fig.patch.set_facecolor('white')
    norm = plt.Normalize(vmin=0, vmax=len(symbols)-1)
    cmap = mpl.colors.LinearSegmentedColormap.from_list('cmap', palette)
    color = [mpl.colors.to_hex(k) for k in cmap(norm([z[j] for j in list(struct.symbols)]))]
    plot_atoms(struct, ax, radii=0.25, colors=color, rotation=(rot))

    ax.set_xlabel(r'$x_1\ (\AA)$')
    ax.set_ylabel(r'$x_2\ (\AA)$')
    if title is None:
        ftitle = f"{struct.get_chemical_formula().translate(sub)}"
        fname =  struct.get_chemical_formula()
    else: 
        ftitle = f"{title} / {struct.get_chemical_formula().translate(sub)}"
        fname = f"{title}_{struct.get_chemical_formula()}"
    ax.set_title(ftitle, fontsize=15)
    if savedir is not None:
        path = savedir
        if not os.path.isdir(f'{path}'):
            os.mkdir(path)
        fig.savefig(f'{path}/{fname}.png')
    if ax is not None:
        return ax



def movie_structs(astruct_list, name, savedir=None, supercell=np.diag([1,1,1])):
    if not os.path.exists(savedir):
        os.makedirs(savedir)
    for i, struct_in in enumerate(astruct_list):
        vis_structure(struct_in, supercell, title=f"{{0:04d}}".format(i), savedir=savedir)
    
    with imageio.get_writer(os.path.join(savedir, f'{name}.gif'), mode='I') as writer:
        for figurename in sorted(os.listdir(savedir)):
            if figurename.endswith('png'):
                image = imageio.imread(os.path.join(savedir, figurename))
                writer.append_data(image)


# material data conversion
def str2pymatgen(cif):
    pstruct=Structure.from_str(cif, "CIF")
    return pstruct

# pymatgen > ase.Atom
def pymatgen2ase(pstruct):
    return Atoms(list(map(lambda x: x.symbol, pstruct.species)),
                    positions = pstruct.cart_coords.copy(),
                    cell = pstruct.lattice.matrix.copy(), 
                    pbc=True)

def ase2pymatgen(astruct):
    lattice = Lattice(astruct.cell)  # Convert the cell to a Pymatgen Lattice
    species = astruct.get_chemical_symbols()  # Get the list of element symbols
    positions = astruct.get_positions()  # Get the atomic positions
    pstruct = Structure(lattice, species, positions)
    return pstruct



def xyz_to_cif(xyz_file, cif_file, pbc=(True, True, True), pmg=True):
    """
    Convert an XYZ file to a CIF file, incorporating periodic boundary conditions and magnetic moments.
    
    Args:
    - xyz_file: Path to the input .xyz file.
    - cif_file: Path to the output .cif file.
    - pbc: Tuple of three boolean values indicating periodicity along x, y, z axes (default is (True, True, True)).
    """
    # Step 1: Read the XYZ file using ASE
    atoms = read(xyz_file)
    
    # Step 2: Set periodic boundary conditions (pbc)
    atoms.set_pbc(pbc)

    # Step 4: Convert ASE Atoms object to Pymatgen Structure
    structure = AseAtomsAdaptor.get_structure(atoms)
    # set pbc condition using pymatgen.core.structure.Structure
    structure.add_site_property("selective_dynamics", [[True, True, True]]*len(structure))

    # Step 5: Write the structure to a CIF file with Pymatgen
    writer = CifWriter(structure)
    writer.write_file(cif_file)
    
    print(f"Successfully converted {xyz_file} to {cif_file} with pbc={pbc}.")

    if pmg:
        return structure
    else: 
        return atoms



