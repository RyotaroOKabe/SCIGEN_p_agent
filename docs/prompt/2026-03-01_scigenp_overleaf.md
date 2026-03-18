/pscratch/sd/r/ryotaro/data/generative/overleaf/scigenp_overview
Here's the overleaf folder which I am going to write a short doc to explain what we can do with SCIGEN+ (/pscratch/sd/r/ryotaro/data/generative/SCIGEN_p_agent) to the collaborator. could you help me write a doc? 

when you write the doc, you can reference our previous work (SCIGEN) in /pscratch/sd/r/ryotaro/data/generative/overleaf/SCIGEN_archive (do not edit this folder)
for example, you can use figures or notations used in this folder for our new doc. 

You showld present some materials structures. In that case, please generate new crystal structures using:
/pscratch/sd/r/ryotaro/data/generative/SCIGEN_p_agent/scripts/generation/gen_mul_natm.py
/pscratch/sd/r/ryotaro/data/generative/SCIGEN_p_agent/config_scigen.py

Plot the materials structure with:
/pscratch/sd/r/ryotaro/data/generative/SCIGEN_p_agent/utils/visualize/crystal_viz.py

The The initial part of the documentation should be the basic overview about what this code (SCIGEN+) can do.
No mathematical explanation is necessary. 

[1] 
The first chapter is about what is the input variables which we can specify as the input: 
type of structural motifs (e.g., Kagome, Lieb, ...) C, the number of atoms per unit cell N, the atom types A (but can specify partially, such as only specifying the atom types of Kagome lattice. 

Also, as you can find in tab_arch_latt of /pscratch/sd/r/ryotaro/data/generative/overleaf/SCIGEN_archive/2_supplement_clean.tex, each structural motif need to dedicate $N^{c}$ in N. You can summarize the table to show the relationship of structural motifs (here just a few patters are enough: triangular, honeycomb, kagome, square, lieb), the lattice shape, and N^c. 

[2]
The following chapters are about the case studies how we can use SCIGEN+.

[2-1]
specify: structural motif, N, A for structural motif sites (A^c). 
Free: A for the rest of the atoms
show in what situation we can use this kind of case studies (e.g., have idea about what kind of motifs you want to study, but have no idea about the rest of the atoms)
Present the three images of examples (subfigure abc):
fig1a: kagome, A^c = Fe, N=6
fig1b:  kagome, A^c = Fe, N=8
fig1c:  kagome, A^c = Fe, N=10


[2-2]
specify: structural motif, N,  
Free: A for all atoms
show in what situation we can use this kind of case studies (e.g., have idea about what kind of motifs you want to study, but have no idea about the type of atoms)
Present the three images of examples (subfigure abc):
fig2a: kagome, N=6
fig2b: kagome, N=8
fig2c:  kagome, N=10

[2-3]
specify: structural motif, N,  specify the all atom types (e.g., chemical formula)
show in what situation we can use this kind of case studies (e.g., have idea about what kind of motifs you want to study, and have some idea about the chemical formula)
Present the three images of examples (subfigure abc):
fig3a: kagome, N=6 (please consder chemical formula!!)
fig3b: kagome, N=8 (please consder chemical formula!!)
fig3c:  kagome, N=10 (please consder chemical formula!!)


[2-4]
explore structural motifs
show in what situation we can use this kind of case studies (e.g., want to explore which structural motifs can induce interesting phenomena)


[2-5]
Create new structural motif 
Starting from the motif template SC_Template in /pscratch/sd/r/ryotaro/data/generative/SCIGEN_p_agent/script/sc_utils.py, develop a new structural motif. 
Here' you do not need to work on demo, but write a message about what we can do, how we can explore. 


