# Seminar Report: "Quantum Complexity in Simple Materials" — Xavier Roy (Columbia)

**Seminar Series:** Chez Pierre (Condensed Matter Seminar), MIT Physics
**Date:** March 9, 2026
**Speaker:** Prof. Xavier Roy, Department of Chemistry, Columbia University
**Host:** Prof. Riccardo Comin (MIT)
**Poster:** [Announcement](https://physics.mit.edu/wp-content/uploads/2026/01/Xavier-Roy.-Chez-Pierre_announcement-poster.pdf)
**Recording:** [260309-Chez-Pierr-Xavier-Roy-Columbia.m4a](https://www.dropbox.com/scl/fi/202pjgj4obuzzc8e8wmjz/260309-Chez-Pierr-Xavier-Roy-Columbia.m4a?rlkey=3p7vbl6dcaibik0scz7eve8mw&st=bv5tpbgr&dl=0)
**Attended by:** Ryotaro Okabe (RO)

---

## 1. Speaker Background

Xavier Roy is a solid-state chemist at Columbia University (full professor since 2024). He did his undergrad at the University of Montreal / École Polytechnique, PhD at the University of British Columbia (graduated 2011), and postdoc at Princeton. He has received the NSF CAREER Award and became a Brown Investigator in 2025. His group specializes in the design, crystal growth, and characterization of new van der Waals materials, working at the intersection of solid-state chemistry and condensed matter physics.

---

## 2. Talk Overview

The talk covered two main projects, both centered on discovering **quantum complexity hidden in structurally simple materials** — compounds with few elements and familiar crystal structure types that nonetheless host exotic quantum phenomena.

---

## 3. Part I: Frustrated 2D Metals and Compact Localized States in PdAlI

### 3.1 Lattice Symmetry and Electronic Structure

Roy opened by reviewing how lattice geometry determines band structure, using three canonical examples:
- **Honeycomb lattice** → Dirac cones (graphene, predicted 1947, realized 2004)
- **Lieb lattice** (checkerboard + edge decoration) → Dirac crossings intersected by a flat band
- **Kagome lattice** → flat bands + Dirac cones

He emphasized that combining **simple lattice symmetry theory with new materials discovery** has historically been the recipe for breakthrough physics (graphene being the prime example).

### 3.2 PdAlI: A "Simple" Compound with Hidden Frustration

The featured material is **palladium aluminum iodide (PdAlI)**, a derivative of the FCC lattice. Despite its structural simplicity (only 3 elements, a common crystal structure type), it hosts a **decorated checkerboard lattice** in its mid-plane:

- The Pd-Al checkerboard plane maps onto the Lieb lattice model
- Decoration with Pd $d_{yz}$/$d_{xy}$ and Al $p_z$ orbitals creates orbital-driven frustration
- This generates **flat bands** from destructive interference of electron wavefunctions

Roy grew large single crystals ("looks like expensive aluminum foil") and confirmed the electronic structure using **ARPES** (angle-resolved photoemission spectroscopy).

### 3.3 Compact Localized States (CLS) — "Ghost Atoms"

The most striking result: **first experimental observation of electronic compact localized states** in a solid-state material.

**Background:**
- Flat bands are collections of degenerate, localized states (compact localized states, CLS)
- CLS were predicted theoretically and observed in photonic systems but never in electronic systems
- Engineering destructive interference for electrons is much harder than for photons (shorter wavelength)

**Key experiment:**
- Using **STM** (scanning tunneling microscopy, collaboration with Abhay Pasupathy at Columbia), they imaged the surface of PdAlI
- At **Pd vacancy sites** within the checkerboard, they observed highly localized electronic states embedded in the metallic background
- These states appear as sharp peaks in STS (scanning tunneling spectroscopy) — completely unexpected for a metal
- Spatial mapping reveals states that look like **atomic orbitals** (S-like, D-like) but are **multi-site quantum objects** spanning 1-2 nm

**"Molecular orbital theory" of CLS:**
- When two Pd vacancies are one unit cell apart, the CLS peaks split into bonding and anti-bonding pairs — exactly as predicted by molecular orbital theory
- The spatial maps show bonding (constructive) and anti-bonding (destructive) combinations
- Remarkably, the energy ordering is **inverted** compared to standard MO theory (anti-bonding lower than bonding) due to the relative phase of the orbitals involved

**Topological protection:** These CLS are topologically protected — they appear whenever the specific Pd vacancy is present, without fine-tuning. Other defect types (Pd outside the checkerboard, iodine vacancies) do not produce CLS.

### 3.4 Chemical Substitution and Kondo Effects

- Substituting Fe or other 3d metals for Pd at the vacancy site **lowers** the CLS energy into the metallic background
- The filled CLS behaves like a magnetic impurity → **Kondo effect**
- The Kondo resonance comes not from the substituent atom's magnetic moment but from the **filled compact localized state** between the atoms
- Fe substitution creates chains of linked CLS that form "Kondo molecules" with split Kondo peaks

---

## 4. Part II: Multiferroic Quantum Magnetism in NbOX₂ / VOX₂

### 4.1 Niobium Oxyhalides (NbOI₂)

Roy's group works on a family of van der Waals ferroelectric materials: niobium oxyhalides (NbOX₂, X = Cl, Br, I).

- NbOI₂ is a layered ferroelectric with Nb in a 4+ oxidation state ($d^1$)
- The $d^1$ electron pairs in Nb-Nb dimers along the non-polar axis → material is diamagnetic
- They observed **coherent propagating polarization waves** (phasons / amplitude modes of the ferroelectric order parameter) using pump-probe techniques and super-resolution scattering

### 4.2 Vanadium Oxyhalides (VOX₂) — Ferroelectricity Meets Quantum Magnetism

A postdoc (Willa) identified **VOI₂** as a candidate where magnetism and ferroelectricity might coexist:
- V replaces Nb → lighter atom, no dimerization at any temperature
- The V magnetic moment ($d^1$, spin-1/2) is completely unquenched
- Crystal structure confirmed as non-centrosymmetric polar (space group $Imm2$)

**Key findings:**
- **No magnetic ordering** down to 2 K (confirmed by neutron diffraction and EPR)
- Instead: broad maximum in magnetic susceptibility around 120 K → **spin-singlet ground state**
- The system behaves as a **1D spin-1/2 chain** with strong coupling through halogens, weak coupling through oxide
- Magnetic data fits a **tetramer model** (dimer of dimers) rather than simple dimerization
- Extracted spin gap: ~20 meV (large, explaining the high-temperature onset)
- Electron diffraction shows **4× superlattice peaks** along the spin chain direction, consistent with tetramerization

**Multiferroic coupling (indirect evidence):**
- Tetramerization and ferroelectricity are along orthogonal directions
- SHG intensity and magnetic susceptibility show correlated temperature dependence
- Raman spectroscopy shows discrete transition at ~125 K with new zone-folded peaks and suppression of quasi-elastic scattering → consistent with spin gap opening
- Some phonon modes show oscillations along the magnetic direction → spin-phonon coupling

**Ongoing/planned experiments:**
- Inelastic neutron scattering (beam time coming) to observe singlet-triplet excitations of the tetramer
- Investigation of **polarons** — spins that fail to form tetramers and remain as entangled impurities (the low-temperature susceptibility tail)
- Chemical tuning via halide substitution: demonstrated tunability of symmetry (centrosymmetric ↔ non-centrosymmetric) and band gap in the 1-2-2 family

---

## 5. Key Themes and Takeaways

### 5.1 "Quantum Complexity in Simple Materials"

The unifying message: structurally simple compounds (few elements, common crystal structure types) can host rich quantum phenomena when the right combination of **lattice geometry + orbital character + chemical substitution** is achieved. The complexity is hidden in the interplay of these factors, not in structural complexity.

### 5.2 Flat Bands as a Design Principle

Roy's work on PdAlI demonstrates that flat bands are not just theoretical curiosities — they give rise to observable, manipulable quantum states (CLS). The decorated checkerboard / Lieb lattice model is the theoretical backbone, and the CLS phenomenon should generalize to **other flat-band lattices including kagome** (Roy explicitly mentioned this).

### 5.3 Experimental Accessibility

- Van der Waals materials can be exfoliated, contacted, and gated using standard lithography
- Large single crystals (centimeter-scale) enable bulk characterization (neutron diffraction, Raman, ARPES)
- Chemical substitution provides systematic tunability

---

## 6. Connection to SCIGEN+ Project

### 6.1 Direct Relevance: Flat Band Materials Discovery

Roy's PdAlI work is a perfect case study for what SCIGEN+ aims to discover computationally:
- **Decorated checkerboard / Lieb lattice** is one of SCIGEN's supported structural motifs ($N^c = 3$)
- The key insight — that orbital decoration of a geometric lattice creates flat bands — is exactly the design principle SCIGEN encodes through structural constraints
- The CLS phenomenon Roy observed is a direct consequence of flat band physics that SCIGEN+ could help discover in new materials

### 6.2 Potential SCIGEN+ Contributions

1. **Candidate generation for Lieb/checkerboard flat bands**: SCIGEN can generate novel compounds with the decorated checkerboard motif. DPO fine-tuning could steer generation toward compounds where the orbital character (d-orbital decoration) favors flat band formation — exactly the PdAlI mechanism.

2. **Kagome CLS candidates**: Roy mentioned that CLS should generalize to kagome lattices with different orbital symmetries. SCIGEN with kagome constraints + DPO preference for flat band isolation could systematically search for kagome materials hosting CLS.

3. **Chemical substitution screening**: Roy's group manually explores substituents (Fe, Ni, etc. for Pd). SCIGEN+ could generate substitution variants computationally, with DPO preferences trained on band structure quality near the flat band energy.

4. **Multiferroic van der Waals materials**: The VOX₂ family suggests that combining ferroelectricity with quantum magnetism in simple layered structures is fertile ground. SCIGEN could explore the broader chemical space of transition-metal oxyhalides with specific structural motifs.

### 6.3 Expert Knowledge as Preference Signal

At the end of the seminar, RO briefly spoke with Prof. Roy about the SCIGEN+ approach of integrating human expert intuition into generative models. Roy was receptive to the idea and open to future discussion. His group's deep chemical intuition about which structural motifs and orbital decorations lead to interesting physics is exactly the kind of expert knowledge that could serve as a high-quality preference signal for DPO fine-tuning.

**Key quote from the brief exchange**: Roy acknowledged the value of encoding chemist's intuition into AI models, and expressed willingness to discuss further when needed ("Yeah, no problem").

### 6.4 Implications for Preference Design

Roy's work highlights that **flat band quality depends critically on orbital character, not just geometry**. This suggests that for SCIGEN+ preference data:
- Simple geometric flatness scores may miss the key physics (orbital-driven frustration)
- Expert preferences that consider orbital character and band isolation near specific energies are essential
- The checkerboard/Lieb case shows that even "obvious" flat band geometries can fail if the orbital decoration is wrong — human judgment is needed to distinguish good from bad flat band candidates

---

## 7. Questions and Discussion During Q&A

Several notable questions from the audience:

1. **On chemical substitution at vacancy sites**: Asked whether substituting the "wrong" atom (not the Pd in the checkerboard) produces CLS. Answer: No — only the specific Pd vacancy within the checkerboard generates CLS. Other defects show no localized states.

2. **On topological protection**: CLS are topologically protected — no fine-tuning needed. Removing the specific Pd atom always produces CLS.

3. **On multiferroic definition**: A questioner noted confusion about the multiferroic claim — the theoretical prediction assumed coexistence of magnetic order + ferroelectric order, but Roy's data shows quantum magnetism (no long-range order) + ferroelectricity. Roy clarified that this is more exotic than predicted: quantum magnetism replaces classical magnetic order.

4. **On polarization waves vs spin waves**: Question about whether the observed excitation could be a spin wave. Roy confirmed it is a polarization wave (the ferroelectric analog of a magnon), not a spin wave, based on their mapping data.

5. **On stability**: The materials are stable in oxygen but sensitive to humidity. Workable in gloveboxes and dry environments.

---

## 8. References and Materials Mentioned

- **PdAlI**: Decorated checkerboard lattice, compact localized states (unpublished at time of talk)
- **NbOI₂**: Van der Waals ferroelectric, coherent polarization waves
- **VOI₂**: Ferroelectric + quantum magnetism, tetramerization, spin-1/2 chain
- **Collaboration**: Abhay Pasupathy (Columbia, STM), John Harter (Columbia, super-resolution scattering), Raquel (theory, decorated checkerboard model)
- **Funding**: NSF CAREER, Brown Investigator
