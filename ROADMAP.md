# SCIGEN Roadmap

## Phase 1: Foundation (Current Status)
**Objective**: Establish robust generative capabilities with structural constraints.

- [x] **Framework Development**: Core Diffusion + CSPNet architecture implementation.
- [x] **Constraint Integration**: "Structural Constraint Manifold" for enforcing space group symmetry (Kagome, etc.).
- [x] **Training Pipeline**: Multi-GPU support for efficient training on MP-20 and Alexandria datasets.
- [x] **Initial Validation**: Demonstrated ability to generate stable crystals with desired symmetries.
- [ ] **Agentic Prototype**: Initial implementation of "Generate $\to$ Evaluate" loop with VASP.

## Phase 2: Scientific Validation (Q3 2026)
**Objective**: Prove that generated materials are not just valid, but *valuable*.

- [ ] **Lab Synthesis**: Partner with experimental groups to synthesize top candidates (e.g., predicted Kagome magnets).
- [ ] **Property Characterization**: Verify flat bands and magnetic properties in synthesized samples.
- [ ] **Social Welfare Function (SWF) Learning**: Train the evaluator using active learning from expert feedback (Human-in-the-Loop).
- [ ] **Benchmarking**: Compare "Agentic Exploration" vs. "Random Search" yield metrics.

## Phase 3: Autonomous Discovery (2027+)
**Objective**: Full-scale autonomous agent for solving open problems in physics.

- [ ] **Closed-Loop Autonomy**: Fully automated cycle of Generation $\to$ DFT $\to$ Evaluation $\to$ Refinement.
- [ ] **"Hilbert's Problems" Campaign**: Deploy agents to solve specific theoretical challenges (e.g., Quantum Spin Liquid realization).
- [ ] **Platform Scaling**: Cloud-native deployment for massive parallel exploration.
- [ ] **Community Ecosystem**: Open-source release of pre-trained "Motif Models" for the materials science community.
