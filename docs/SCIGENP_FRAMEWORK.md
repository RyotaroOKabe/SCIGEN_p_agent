# SCIGEN+: Structural-Constraint-Guided Crystal Generation

> **For NotebookLM slide deck** — each `##` section maps to one slide.
> All diagrams are self-contained; no prior knowledge of diffusion models required.

---

## 1. What Problem Does SCIGEN+ Solve?

**Goal:** Generate novel crystal structures that contain a specific geometric arrangement of atoms — without retraining any model.

### The Challenge

Designing magnetic or quantum materials often requires atoms to sit on a specific **geometric motif** inside the crystal:

| Motif | Why it matters |
|-------|---------------|
| **Kagome** | Flat electronic bands → frustrated magnetism |
| **Honeycomb** | Kitaev physics, topological insulation |
| **Lieb** | Localized flat-band states |
| **Pyrochlore** | Spin-ice, quantum spin liquid candidates |

Current ML crystal generators produce *random* structures — they cannot guarantee that a target motif appears in every generated sample.

### SCIGEN+ Answer

> Enforce the motif as an **inference-time constraint** on a frozen pretrained diffusion model.
> No retraining. No weight updates. Plug-and-play.

---

## 2. Three Inputs → One Crystal

SCIGEN+ requires exactly three inputs per generation run:

```
┌─────────────────────────────────────────────────────────────┐
│                     SCIGEN+ INPUTS                          │
│                                                             │
│  C  ──── Structural Motif   (e.g., kagome lattice)          │
│          Specifies WHICH sites are fixed and WHERE          │
│                                                             │
│  N  ──── Total Atom Count   (e.g., N = 8)                   │
│          N = N_constrained + N_free                         │
│          (e.g., kagome: N_c = 3, so N_free = 5)            │
│                                                             │
│  Aᶜ ──── Motif Atom Type    (e.g., Fe, Mn, Co, Ni, Cu)     │
│          Element placed on the constrained sites            │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### What each input controls

| Input | Type | Example | Controls |
|-------|------|---------|----------|
| **C** | Structural motif class | Kagome (SG 191) | Fixed fractional coordinates + spacegroup |
| **N** | Integer | 8 | Total atoms per unit cell |
| **Aᶜ** | Element symbol(s) | Fe, Mn | Species assigned to motif sites |

> **The remaining N − Nᶜ atoms** (species + positions) are **freely predicted** by the diffusion model.

---

## 3. What Is a Structural Motif (C)?

A structural motif defines a **template of atomic positions** inside the unit cell.

### Kagome Example

```
        Top view (ab-plane)
        ─────────────────────
        ●───●
       / \ / \
      ●   ●   ●      ← 3 Fe atoms per unit cell (N_c = 3)
       \ / \ /
        ●───●

  Fractional coordinates:
    Site 1:  (0,   0,   z)
    Site 2:  (1/2, 0,   z)
    Site 3:  (0,   1/2, z)

  z is a free parameter (out-of-plane position)
  Spacegroup: 191 (hexagonal)
```

### Motif Library

| Code | Motif | Nᶜ | Spacegroup | Crystal family |
|------|-------|-----|-----------|----------------|
| `tri` | Triangular | 1 | 191 | Hexagonal |
| `hon` | Honeycomb | 2 | 191 | Hexagonal |
| `kag` | Kagome | 3 | 191 | Hexagonal |
| `sqr` | Square | 1 | 141 | Tetragonal |
| `lieb` | Lieb | 3 | 141 | Tetragonal |
| `pyc` | Pyrochlore | 16 | 227 | Cubic |
| `hkg` | Hyper-Kagome | 12 | 213 | Cubic |

> **Adding a new motif** requires only two lines of Python: a spacegroup integer and a tensor of fractional coordinates. No retraining needed.

---

## 4. The Base Model: DiffCSP+

SCIGEN+ sits on top of **DiffCSP+**, a pretrained diffusion model for crystal generation.

### DiffCSP vs DiffCSP+

```
DiffCSP  ──────────────────────────────────────────────────────
  Lattice:    6 raw parameters (a, b, c, α, β, γ) — FIXED before sampling
  Atoms:      Fractional coordinates + atom types (joint diffusion)
  Limitation: Unit cell size must be manually specified → over-constrained

DiffCSP+  ─────────────────────────────────────────────────────
  Lattice:    6D vector via POLAR DECOMPOSITION — learned jointly
  Atoms:      Fractional coordinates + atom types (joint diffusion)
  Advantage:  Model freely adjusts unit cell size → more stable structures
  (Note: DiffCSP++ also adds Wyckoff constraints; SCIGEN+ uses lattice only)
```

### Polar Decomposition Lattice Representation

```
  Lattice matrix L (3×3)
        │
        ▼  polar decomposition
  L_sym = √(L · Lᵀ)   ← symmetric positive-definite part
        │
        ▼  logarithmic map
  v ∈ ℝ⁶              ← 6D vector (smooth, unconstrained space)
        │
        ▼  spacegroup mask
  v_proj               ← only symmetry-allowed components kept

  Example — Hexagonal (kagome):
    Full 6D:  [v1, v2, v3, v4, v5, v6]
    Mask:     [ 0,  0,  0,  0,  1,  1]  ← only v5, v6 active
    Result:   enforces a = b, γ = 120°; c is free
```

---

## 5. The Generation Pipeline

```
  INPUTS
  ┌───────────────────┐
  │  C  N  Aᶜ         │
  └────────┬──────────┘
           │
           ▼
  ┌───────────────────────────────────────────────────────────┐
  │  INITIALIZATION  (t = T)                                  │
  │                                                           │
  │  Constrained sites:  x_known = kagome positions (clean)  │
  │  Free sites:         x_free  = random noise              │
  │  Lattice:            random 6D vector                    │
  │                                                           │
  │  Merge:  x_T = mask · x_known  +  (1−mask) · x_free     │
  └───────────────────────────────────────────────────────────┘
           │
           ▼  ──── repeat T times ────────────────────────────
  ┌───────────────────────────────────────────────────────────┐
  │  DENOISING STEP  (t → t−1)        Predictor–Corrector     │
  │                                                           │
  │  ① CORRECTOR (free sites only)                            │
  │     DiffCSP+ predicts denoised positions for free atoms  │
  │     Constrained sites: x_known(t) = x₀ + σ(t) · ε       │
  │     (noisy oracle — keeps motif geometry at correct       │
  │      noise level for this timestep)                       │
  │     Merge → x_{t−½}                                      │
  │                                                           │
  │  ② PREDICTOR (all sites)                                  │
  │     DiffCSP+ refines full structure                       │
  │     Constrained sites: x_known(t−1) = x₀ + σ(t−1) · ε  │
  │     Free sites: diffusion model update                   │
  │     Merge → x_{t−1}                                      │
  │     Project lattice → spacegroup-valid representation     │
  └───────────────────────────────────────────────────────────┘
           │
           ▼  (after T steps)
  ┌───────────────────────────────────────────────────────────┐
  │  OUTPUT  (t = 0)                                          │
  │  • Constrained atoms:  exactly on motif sites (σ(0) = 0) │
  │  • Free atoms:         settled by diffusion model         │
  │  • Lattice:            self-consistent with structure     │
  └───────────────────────────────────────────────────────────┘
```

---

## 6. Step-by-Step Denoising Visualization

How the crystal evolves from noise to structure (kagome example, N=8, Aᶜ=Fe):

```
t = T  (pure noise)
  ┌──────────┐
  │ ·  ○  ·  │   ○ = constrained (Fe, kagome)  — scattered noisily
  │  ·   ·   │   · = free atoms                — random positions
  │ ·  ○  ·  │   Lattice: distorted box
  └──────────┘

t = T/2  (partial denoising)
  ┌──────────┐
  │  ·  ○ ·  │   ○ sites: still slightly noisy, but approaching kagome pattern
  │   ·   ·  │   · sites: beginning to cluster into plausible positions
  │  ·  ○    │   Lattice: becoming hexagonal
  └──────────┘

t = 0  (final crystal)
  ┌──────────┐
  │  · ○─○ · │   ○ sites: exactly on kagome positions
  │   ·   ·  │   · sites: stable chemical environment found by model
  │  · ○─○ · │   Lattice: proper hexagonal unit cell
  └──────────┘

  Key invariant at every step:
  x_total(t)  =  mask · [x₀ + σ(t)·ε]  +  (1−mask) · x_model(t)
                 ─────────────────────     ──────────────────────
                   constrained (motif)          free (model)
```

---

## 7. The Masking Equation

The core equation that enforces the constraint at every denoising step:

```
  x(t)  =  mask · x_known(t)  +  (1 − mask) · x_free(t)

  where:
    mask        ∈ {0, 1}^N      1 = constrained site, 0 = free site
    x_known(t)  = x₀ + σ(t)·ε  noisy oracle (clean motif + noise level t)
    x_free(t)   = model output  DiffCSP+ prediction for free sites

  Same equation applies separately to:
    • Fractional coordinates  (3D per atom)
    • Atom types              (one-hot, 100D per atom)

  Lattice is NEVER masked — always free (DiffCSP+ adjusts it freely)
```

> **Analogy:** This is identical to **inpainting** in image diffusion —
> the motif sites are "known pixels"; the rest of the crystal is "inpainted" by the model.

---

## 8. Three Generation Modes

SCIGEN+ supports three degrees of freedom in specifying the output:

```
MODE 1 — Specify motif + N + Aᶜ (most constrained)
  ┌───────────────────────────────────────────┐
  │  C = kagome,  N = 8,  Aᶜ = Fe            │
  │  • 3 Fe atoms locked on kagome sites      │
  │  • 5 atoms: types freely chosen by model  │
  │  • Result: e.g. Fe₃Mn₂Al₃ kagome crystal │
  └───────────────────────────────────────────┘

MODE 2 — Specify motif + N only (free atom types)
  ┌───────────────────────────────────────────┐
  │  C = kagome,  N = 8,  Aᶜ = (any)         │
  │  • 3 motif atoms: type chosen by model    │
  │  • 5 atoms: types freely chosen by model  │
  │  • Result: e.g. Co₆Ru₂ kagome crystal    │
  └───────────────────────────────────────────┘

MODE 3 — Specify motif + N + full formula (most controlled)
  ┌───────────────────────────────────────────┐
  │  C = kagome,  N = 8,  formula = Fe₆Si₂   │
  │  • 3 Fe on kagome sites (t_mask = True)   │
  │  • Post-filter by composition             │
  │  • Result: clean binary/ternary formulas  │
  └───────────────────────────────────────────┘
```

---

## 9. Why SCIGEN+ Is Better Than SCIGEN

```
                    SCIGEN              SCIGEN+
                    ──────              ───────
Base model          DiffCSP             DiffCSP+ (polar decomp.)
Unit cell size      Manually fixed      Freely predicted by model
                    (draw from MP20     (no manual specification)
                     NN-distance dist.)
Over-constrained?   Yes                 No
Stable yield        Low (~few %)        Higher
Motif enforcement   Same (masking)      Same (masking)
Wyckoff constraint  No                  No (intentionally omitted —
                                        hard to combine with
                                        arbitrary motif geometry)
```

**Key improvement:** In SCIGEN, the lattice parameter `a` had to be set as `a = 2 × dᶜ`, where `dᶜ` was sampled from the training-set nearest-neighbor distribution. This over-constrains the cell and forces the model to work around a fixed geometry. SCIGEN+ removes this bottleneck.

---

## 10. Defining a New Structural Motif

Adding a new motif to SCIGEN+ requires **two pieces of information only**:

```python
class SC_MyNewMotif(SC_Base):
    def __init__(self, ...):
        super().__init__(...)
        self.spacegroup = 191          # ← (1) spacegroup integer
        self.frac_known = torch.tensor([   # ← (2) fractional coords
            [0.0,  0.0,  0.5],         #   site 1
            [0.5,  0.0,  0.5],         #   site 2
            [0.25, 0.25, 0.5],         #   site 3
        ])
```

Then register in `sc_dict`:
```python
sc_dict = {
    ...
    'new': SC_MyNewMotif,    # ← one line
}
```

Now `--sc new` works immediately in generation. No retraining.

---

## 11. Agentic Loop: AI-Driven Motif Discovery

Because motif definition requires only a spacegroup and fractional coordinates,
an **LLM agent** can autonomously propose, implement, and evaluate new motifs:

```
  ┌─────────────────────────────────────────────────────────┐
  │  AGENTIC LOOP                                           │
  │                                                         │
  │  1. Agent observes:  kagome results show flat bands     │
  │     near Fermi level → want denser connectivity        │
  │                                                         │
  │  2. Agent proposes:  "decorated kagome" motif           │
  │     (add 3 extra atoms at triangle centers)             │
  │                                                         │
  │  3. Agent implements:                                   │
  │     SC_DecoratedKagome(spacegroup=191,                  │
  │       frac_known=[[0,0,z],[½,0,z],[0,½,z],             │
  │                   [¼,¼,z],[¾,¼,z],[¼,¾,z]])            │
  │                                                         │
  │  4. Agent runs:  python script/generation.py --sc dkag  │
  │                                                         │
  │  5. Agent screens:  pass rate 8% → 14% improvement     │
  │                                                         │
  │  6. Agent reports and iterates ──────────────────────┐  │
  │  └──────────────────────────────────────────────────→┘  │
  └─────────────────────────────────────────────────────────┘
```

SCIGEN+ is a **natural environment for autonomous AI materials discovery** —
the minimal interface (spacegroup + fractional coords) is exactly what an LLM can reliably generate and reason about.

---

## 12. End-to-End Pipeline Summary

```
  USER                SCIGEN+                   OUTPUT
  ─────               ───────────────────────   ──────────────────────

  Specify             Load pretrained            CIF files of generated
  C, N, Aᶜ  ──────►  DiffCSP+ (frozen)  ──────► crystal structures
                              │
                              ▼
                      T denoising steps
                      ┌─────────────────┐
                      │ At each step t: │
                      │                 │
                      │ Known sites:    │
                      │ x = x₀+σ(t)·ε  │
                      │                 │
                      │ Free sites:     │
                      │ x = model pred  │
                      │                 │
                      │ Merge by mask   │
                      │ Project lattice │
                      └─────────────────┘
                              │
                              ▼
                      Stability screening
                      (CHGNet + SMACT)
                              │
                              ▼
                      Candidate structures
                      for DFT validation
```

### Key Properties

| Property | Value |
|----------|-------|
| **Constraint type** | Inference-time (no retraining) |
| **Base model** | DiffCSP+ (pretrained, frozen) |
| **Motif guarantee** | 100% — constrained sites always on motif |
| **Free atom types** | Predicted by diffusion model |
| **Lattice** | Freely optimized (polar decomposition) |
| **Adding new motif** | 2 lines of Python, 0 retraining steps |
| **Compatible with fine-tuning** | Yes — fine-tune DiffCSP+ first, apply SCIGEN+ at inference |

---

## 13. Connection to Post-Training / RLHF

Because SCIGEN+ is **fully decoupled** from training:

```
  WORKFLOW WITH PREFERENCE ALIGNMENT

  Step 1: Pretrain DiffCSP+ on MP-20 dataset
          (open-source, no confidentiality issues)
                    │
                    ▼
  Step 2: Post-train with Holder-DPO
          (align to human/expert preference)
          Weights updated here — no motif constraint yet
                    │
                    ▼
  Step 3: Deploy fine-tuned DiffCSP+ + SCIGEN+ at inference
          Motif constraint layered on at generation time
                    │
                    ▼
  Result: Structures that satisfy BOTH
          • Expert preference (from DPO)
          • Geometric motif constraint (from SCIGEN+)
```

> The two ideas — **SCIGEN+** and **Holder-DPO** — compose naturally because
> one operates at training time and the other at inference time.

---

*Generated from source code at `/pscratch/sd/r/ryotaro/data/generative/SCIGEN_p_agent`*
*Last updated: 2026-03-02*
