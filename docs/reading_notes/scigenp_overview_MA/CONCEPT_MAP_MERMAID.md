# SCIGEN+ DPO Concept Map - Mermaid Diagrams

> **Purpose:** Visual overview of all key concepts and their relationships
> **Format:** Multiple Mermaid diagrams for different aspects
> **Audience:** For understanding the big picture

---

## 🗺️ MAIN CONCEPT MAP: Complete Overview

```mermaid
graph TB
    %% Main goal
    SCIGENP[SCIGEN+ Flat Band Materials Discovery]

    %% Three main components
    DIFFUSION[Crystal Diffusion DiffCSP++]
    DPO[Preference Learning Direct Preference Optimization]
    SCIGEN[Structural Constraints SCIGEN Projection]

    SCIGENP --> DIFFUSION
    SCIGENP --> DPO
    SCIGENP --> SCIGEN

    %% Diffusion details
    DIFFUSION --> CHANNELS[Multi-Channel Diffusion]
    CHANNELS --> L_CHANNEL[Lattice L DDPM]
    CHANNELS --> F_CHANNEL[Fractional Coords F Wrapped Gaussian]
    CHANNELS --> A_CHANNEL[Atom Types A DDPM]

    %% DPO details
    DPO --> HOLDER[Hölder-DPO Robust Training]
    DPO --> MARGIN[DPO Margin g_theta t]
    MARGIN --> IMPROVEMENT[Improvement Score I_theta x t]
    IMPROVEMENT --> DENOISING[Denoising Errors d_theta d_ref]

    HOLDER --> GAMMA[Robustness Param gamma = 2.0]
    HOLDER --> REDESCEND[Redescending Outlier Down-weighting]

    %% SCIGEN details
    SCIGEN --> PROJECTION[Projection Pi_C]
    PROJECTION --> FREE[Free DOF Model Controls]
    PROJECTION --> FIXED[Fixed DOF Motif Values]

    %% Three phases
    SCIGENP --> PHASES[Three-Phase Training]
    PHASES --> PHASEA[Phase A Offline DPO]
    PHASES --> PHASEB[Phase B Motif-Focused]
    PHASES --> PHASEC[Phase C Active Learning]

    %% Phase A
    PHASEA --> MP20[MP-20 Dataset Clean Crystals]
    PHASEA --> FORWARD[Forward Corruption x_t from q]
    PHASEA --> UNCONSTRAINED[No SCIGEN during training]

    %% Phase B
    PHASEB --> BRIDGE[Bridge Formulation Pseudo-Bridge]
    PHASEB --> EXECUTED[Executed Policy p_tilde_theta_C]
    PHASEB --> CONSTRAINED[SCIGEN Active during training]

    BRIDGE --> RECONSTRUCTION[Bridge Reconstruction x0 to xb to x0hat]
    BRIDGE --> BRIDGELEVEL[Bridge Level b Partial Rollout]

    EXECUTED --> CANCELLATION[Constraint Cancellation Lemma 3.1]
    CANCELLATION --> FREELOSS[Loss on Free DOF Only]

    %% Phase C
    PHASEC --> UNCERTAINTY[Uncertainty Sampling]
    PHASEC --> NOVELTY[Novelty Detection]

    %% Key concepts
    IMPROVEMENT --> TRACTABLE[Tractable Proxy Single-Timestep]
    F_CHANNEL --> TORUS[Torus Topology 0-1 periodic]
    F_CHANNEL --> WRAPPED[Wrapped Distance Delta F G]

    DENOISING --> COUPLING[Simple Coupling Same epsilon_F]

    %% Data flow
    PHASEA -.->|fine-tune| PHASEB
    PHASEB -.->|fine-tune| PHASEC

    %% Styling
    classDef mainGoal fill:#ff6b6b,stroke:#c92a2a,stroke-width:3px,color:#fff
    classDef phase fill:#4ecdc4,stroke:#087f5b,stroke-width:2px,color:#fff
    classDef method fill:#95e1d3,stroke:#0ca678,stroke-width:2px
    classDef math fill:#ffd93d,stroke:#f59f00,stroke-width:2px
    classDef data fill:#b8e6f7,stroke:#1971c2,stroke-width:2px

    class SCIGENP mainGoal
    class PHASEA,PHASEB,PHASEC phase
    class DIFFUSION,DPO,SCIGEN method
    class MARGIN,IMPROVEMENT,DENOISING,HOLDER math
    class MP20,BRIDGE,EXECUTED data
```

---

## 🔄 TRAINING PIPELINE: Three-Phase Flow

```mermaid
flowchart LR
    START([Start Pretrained DiffCSP++])

    START --> A_DATA[(Phase A Data MP-20 DFT)]
    A_DATA --> A_TRAIN[Phase A Training Unconstrained H-DPO]
    A_TRAIN --> A_MODEL{{Model theta_A}}

    A_MODEL --> B_GEN[Generate with theta_A + SCIGEN]
    B_GEN --> B_SCREEN[Screen CHGNet + DFT]
    B_SCREEN --> B_ANNOTATE[Human Annotation with kappa]
    B_ANNOTATE --> B_DATA[(Phase B Data Per-motif)]
    B_DATA --> B_TRAIN[Phase B Training Bridge H-DPO]
    B_TRAIN --> B_MODEL{{Model theta_B}}

    B_MODEL --> C_GEN[Generate Batch]
    C_GEN --> C_SELECT[Active Selection Uncertainty]
    C_SELECT --> C_QUERY[Expert Query]
    C_QUERY --> C_DATA[(Phase C Data 50-100 pairs)]
    C_DATA --> C_TRAIN[Incremental Training]
    C_TRAIN --> C_MODEL{{Model theta_C}}

    C_MODEL --> FINAL([Final Model Generate Flat Band Materials])

    C_MODEL -.->|iterate| C_GEN

    style START fill:#e9ecef,stroke:#495057,stroke-width:2px
    style FINAL fill:#51cf66,stroke:#2f9e44,stroke-width:3px,color:#fff
    style A_MODEL fill:#ffd43b,stroke:#fab005,stroke-width:2px
    style B_MODEL fill:#ff922b,stroke:#fd7e14,stroke-width:2px
    style C_MODEL fill:#ff6b6b,stroke:#fa5252,stroke-width:2px
    style A_DATA fill:#a5d8ff,stroke:#339af0,stroke-width:2px
    style B_DATA fill:#a5d8ff,stroke:#339af0,stroke-width:2px
    style C_DATA fill:#a5d8ff,stroke:#339af0,stroke-width:2px
```

---

## 🧮 DPO MATHEMATICS: From RL to Loss

```mermaid
graph TB
    START[RL Objective max E r - beta KL]

    START --> OPTIMAL[Optimal Policy pi_star]
    OPTIMAL --> REWARD[Reward from Policy r = beta log pi_star / pi_ref]
    REWARD --> BT[Bradley-Terry Model P y_w succ y_l]
    BT --> RATIO[Policy Ratio log pi_theta / pi_ref]
    RATIO --> MARGIN[Margin g_theta = I_theta w - I_theta l]
    MARGIN --> HOLDER[Hölder Loss l_gamma beta T g_theta]

    HOLDER --> GRADIENT[Gradient redescending]
    GRADIENT --> INFLUENCE[Influence iota_gamma u_theta]
    INFLUENCE --> ROBUST[Robust to Outliers]

    %% Side branches
    BT --> PREF[Preference Data x_w x_l kappa]
    RATIO --> IMPROVEMENT[Improvement Score I_theta tractable]
    IMPROVEMENT --> DENOISE[Denoising Error d_theta - d_ref]

    HOLDER --> GAMMA_PARAM[Robustness gamma = 2.0]
    HOLDER --> BETA_PARAM[Sharpness beta = 0.1]

    style START fill:#4ecdc4,stroke:#087f5b,stroke-width:2px,color:#fff
    style HOLDER fill:#ff6b6b,stroke:#c92a2a,stroke-width:2px,color:#fff
    style ROBUST fill:#51cf66,stroke:#2f9e44,stroke-width:2px,color:#fff
```

---

## 🌉 PHASE B BRIDGE FORMULATION: Why Needed?

```mermaid
graph TB
    DATA[Phase B Data x_0_w x_0_l from SCIGEN]

    DATA --> NEED[Need Trajectory tau for DPO]
    NEED --> PROBLEM[Problem Original tau not stored]

    PROBLEM --> WHY1[Memory Cost T states per structure]
    PROBLEM --> WHY2[Generation Used Model A + SCIGEN]
    PROBLEM --> WHY3[Need Executed Policy p_tilde_theta_C]

    WHY3 --> SOLUTION[Solution Pseudo-Bridge]

    SOLUTION --> FORWARD[Step 1 Forward Corrupt x_0 to x_b]
    FORWARD --> REVERSE[Step 2 Reverse with ref + SCIGEN x_b to x0hat]

    REVERSE --> REINTRODUCE[Reintroduces SCIGEN Dynamics]
    REINTRODUCE --> APPROXIMATE[Approximate x0hat ≈ x_0]

    APPROXIMATE --> TRACTABLE[Tractable Computable]
    APPROXIMATE --> PRACTICAL[Practical Good Enough]

    FORWARD --> BRIDGELEVEL[Bridge Level b tradeoff]
    BRIDGELEVEL --> SMALL_B[Small b preserves info]
    BRIDGELEVEL --> LARGE_B[Large b more coverage]

    style DATA fill:#a5d8ff,stroke:#339af0,stroke-width:2px
    style PROBLEM fill:#ff6b6b,stroke:#fa5252,stroke-width:2px,color:#fff
    style SOLUTION fill:#51cf66,stroke:#2f9e44,stroke-width:2px,color:#fff
    style TRACTABLE fill:#ffd43b,stroke:#fab005,stroke-width:2px
```

---

## 🏗️ MULTI-CHANNEL DIFFUSION: Crystal Structure

```mermaid
graph TB
    CRYSTAL[Crystal Structure x]

    CRYSTAL --> L[Lattice L 6 params]
    CRYSTAL --> F[Fractional Coords F periodic 0-1]
    CRYSTAL --> A[Atom Types A categorical]

    L --> L_FORWARD[Forward DDPM q L_t L_0]
    L --> L_REVERSE[Reverse p_theta L_t-1 L_t]
    L_REVERSE --> L_GAUSSIAN[Gaussian kernel]

    F --> F_FORWARD[Forward DDPM q F_t F_0]
    F --> F_REVERSE[Reverse p_theta F_t-1 F_t]
    F_REVERSE --> F_WRAPPED[Wrapped Gaussian kernel]
    F_WRAPPED --> F_TORUS[Torus Topology 0-1 periodic]
    F_WRAPPED --> F_COUPLING[Simple Coupling same epsilon_F]

    A --> A_FORWARD[Forward DDPM q A_t A_0]
    A --> A_REVERSE[Reverse p_theta A_t-1 A_t]
    A_REVERSE --> A_GAUSSIAN[Gaussian kernel]

    L_REVERSE --> JOINT[Joint Training]
    F_REVERSE --> JOINT
    A_REVERSE --> JOINT

    JOINT --> IMPROVEMENT[Improvement Score I_theta sum over L F A]
    IMPROVEMENT --> WEIGHTED[Weighted Sum omega_t]

    style CRYSTAL fill:#4ecdc4,stroke:#087f5b,stroke-width:2px,color:#fff
    style L fill:#ffd43b,stroke:#fab005,stroke-width:2px
    style F fill:#ff922b,stroke:#fd7e14,stroke-width:2px
    style A fill:#ff6b6b,stroke:#fa5252,stroke-width:2px
    style JOINT fill:#51cf66,stroke:#2f9e44,stroke-width:2px,color:#fff
```

---

## 🎯 HÖLDER ROBUSTNESS: Down-weighting Outliers

```mermaid
graph TB
    INPUT[Preference Pair x_w x_l kappa]

    INPUT --> MARGIN[Compute Margin u_theta = beta T g_theta]
    MARGIN --> CASES{Model Confidence?}

    CASES -->|u much less 0| OUTLIER[Outlier iota_gamma ≈ 0 Nearly ignored]
    CASES -->|u ≈ 0| UNCERTAIN[Uncertain iota_gamma ≈ 1.5 Learn most here]
    CASES -->|u much greater 0| CONFIDENT[Already learned iota_gamma ≈ 0]

    OUTLIER --> GRADIENT[Gradient Weighting]
    UNCERTAIN --> GRADIENT
    CONFIDENT --> GRADIENT

    GRADIENT --> ADAPTIVE[Adaptive Data-driven]

    INPUT --> KAPPA[Annotator Confidence kappa]
    KAPPA --> DIAGNOSTIC[Diagnostic Only not for training]
    DIAGNOSTIC --> VALIDATE[Validate Check correlation]

    ADAPTIVE --> ROBUST[Robust to Label Noise]

    style INPUT fill:#a5d8ff,stroke:#339af0,stroke-width:2px
    style OUTLIER fill:#ff6b6b,stroke:#fa5252,stroke-width:2px,color:#fff
    style UNCERTAIN fill:#ffd43b,stroke:#fab005,stroke-width:2px
    style CONFIDENT fill:#51cf66,stroke:#2f9e44,stroke-width:2px,color:#fff
    style ROBUST fill:#4ecdc4,stroke:#087f5b,stroke-width:2px,color:#fff
```

---

## 📊 CONSTRAINT CANCELLATION: Free vs Fixed DOF

```mermaid
graph LR
    INPUT[Model Proposal u_t-1 from p_theta]

    INPUT --> SCIGEN[SCIGEN Projection Pi_C]
    SCIGEN --> FREE[Free DOF u_free kept]
    SCIGEN --> FIXED[Fixed DOF x_C_star_fix overwritten]

    FREE --> EXECUTED_FREE[Executed p_tilde free components]
    FIXED --> EXECUTED_FIXED[Executed indicator fixed = motif]

    EXECUTED_FREE --> RATIO[Log-Ratio for DPO]
    EXECUTED_FIXED --> RATIO

    RATIO --> CANCEL[Indicators Cancel in ratio]
    CANCEL --> RESULT[Only Free DOF in loss]

    RESULT --> MASK[Masked Residual C_bar_eff odot r_theta]
    MASK --> NORM[Normalized Error divide by num free]

    NORM --> FAIR[Fair Comparison across constraints]

    style INPUT fill:#a5d8ff,stroke:#339af0,stroke-width:2px
    style SCIGEN fill:#4ecdc4,stroke:#087f5b,stroke-width:2px,color:#fff
    style FREE fill:#51cf66,stroke:#2f9e44,stroke-width:2px,color:#fff
    style FIXED fill:#868e96,stroke:#495057,stroke-width:2px,color:#fff
    style CANCEL fill:#ffd43b,stroke:#fab005,stroke-width:2px
    style FAIR fill:#ff922b,stroke:#fd7e14,stroke-width:2px
```

---

## 🔑 KEY TERMINOLOGY: Quick Reference

```mermaid
mindmap
  root((SCIGEN+ DPO))
    Diffusion
      Multi-Channel
        Lattice L
        Fractional F
        Atom Types A
      Forward q
      Reverse p_theta
      DDPM
      Wrapped Gaussian
    DPO
      Preference Learning
      Bradley-Terry
      Improvement Score
      Margin g_theta
      Hölder Loss
        Robustness gamma
        Redescending
        Influence
    SCIGEN
      Projection Pi_C
      Free DOF
      Fixed DOF
      Constraint C
      Executed Policy
      Cancellation
    Phase A
      MP-20 Data
      Offline
      Unconstrained
      Forward Corruption
    Phase B
      Pseudo-Bridge
      Motif-Focused
      Constrained
      Round-Trip
    Phase C
      Active Learning
      Uncertainty
      Incremental
    Robustness
      Outliers
      Contamination
      Influence Function
      Hölder Divergence
```

---

## 📖 How to Use These Diagrams

### 1. **Main Concept Map**
- Start here for complete overview
- See how all pieces fit together
- Identify which topics to study deeper

### 2. **Training Pipeline**
- Understand data flow through phases
- See how models improve iteratively
- Follow generation → annotation → training loop

### 3. **DPO Mathematics**
- Trace derivation from RL to loss
- Understand why improvement scores work
- See Hölder robustness integration

### 4. **Phase B Bridge**
- Understand why pseudo-bridge needed
- See reconstruction process
- Connect to constraint cancellation

### 5. **Multi-Channel Diffusion**
- Understand per-channel formulations
- See why different channels need different treatment
- Connect to joint training

### 6. **Hölder Robustness**
- Visualize adaptive weighting
- Understand outlier down-weighting
- Compare to confidence κ

### 7. **Constraint Cancellation**
- See free vs fixed decomposition
- Understand why only train on free DOF
- Connect to masking in loss

---

## 🎨 Rendering Instructions

To render these Mermaid diagrams:

### Option 1: GitHub/GitLab
- Push this file to GitHub/GitLab
- Diagrams render automatically in preview

### Option 2: Mermaid Live Editor
- Go to https://mermaid.live/
- Copy-paste each diagram
- Export as PNG/SVG

### Option 3: VS Code
- Install "Markdown Preview Mermaid Support" extension
- Open this file
- Preview renders diagrams

### Option 4: Command Line
```bash
# Install mermaid-cli
npm install -g @mermaid-js/mermaid-cli

# Render to PNG
mmdc -i CONCEPT_MAP_MERMAID_FIXED.md -o concept_map.png

# Render to SVG
mmdc -i CONCEPT_MAP_MERMAID_FIXED.md -o concept_map.svg -t neutral
```

---

## 🔗 Cross-References

- **Detailed explanations:** See [reading_session_2026-03-16_DETAILED.md](reading_session_2026-03-16_DETAILED.md)
- **Mathematical derivations:** See [DERIVATIONS_ANNOTATED.md](DERIVATIONS_ANNOTATED.md)
- **Notation clarifications:** See [VOICE_TRANSCRIPT_CLARIFICATIONS.md](VOICE_TRANSCRIPT_CLARIFICATIONS.md)
- **Quick Q&A:** See [reading_session_2026-03-16.md](reading_session_2026-03-16.md)

---

## 💡 Tips for Understanding

1. **Start with Main Concept Map** → Get the big picture
2. **Follow Training Pipeline** → Understand data flow
3. **Trace DPO Mathematics** → See derivation logic
4. **Study Phase B Bridge** → Understand why it's complex
5. **Check Terminology Mindmap** → Quick reference

**Color coding:**
- 🔴 **Red** = Problems/Challenges
- 🟢 **Green** = Solutions/Methods
- 🟡 **Yellow** = Intermediate steps
- 🔵 **Blue** = Data/Input
- 🟣 **Purple** = Final results

---

**Note:** This is a fixed version with ASCII-safe characters instead of Unicode symbols (θ, ε, etc.) to ensure proper Mermaid rendering.
