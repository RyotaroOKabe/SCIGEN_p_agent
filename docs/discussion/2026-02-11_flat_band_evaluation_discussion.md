# Discussion Report: Flat Band Evaluation Methodology & Strategic Direction
**Date**: 2026-02-11
**Participants**: Ryotaro, Masaki Adachi
**Topic**: Refinement of Pairwise Evaluation Metrics, Discrepancy Analysis, and Roadmap Definition

---

## 1. Executive Summary
The meeting focused on reviewing the results of the newly implemented **Pairwise Evaluation Framework** (both Physics-based and Vision-based). While the framework successfully ranks materials, a discrepancy between the two methods (`mp-29950` vs `mp-19961`) sparked a deeper discussion on the **definition of "flatness"** and the **subjectivity** of expert evaluation.

**Key Decision**: instead of manually tuning weights or assuming a linear objective function, we will shift to an **Inverse Estimation** approach. We will create a small "Golden Dataset" of expert-labeled pairs to *learn* the optimal weights for the scoring function.

**References**:
*   [Vision Evaluation Report](../progress/2026-02-11_progress_vision.md)
*   [Pairwise Evaluation Report](../progress/2026-02-11_progress_pairwise.md)
*   [Vision Documentation](../docs/2026-02-11_vision_eval_doc.md)

---

## 2. Review of Current Status
Ryotaro presented the progress on two parallel evaluation tracks:

### A. Physics-Based Evaluation (Action 1)
*   **Method**: Weighted sum of 4 metrics (Flatness, Extent, Near-Fermi, Isolation) with Monte Carlo weight perturbation.
*   **Result**: **`mp-29950`** was the undisputed winner (100% Win Rate).
*   **Observation**: This material is numerically very flat within the defined energy window but may have dispersive features outside it.

### B. Vision-Based Evaluation (Action 2)
*   **Method**: GPT-4o Vision "Zero-Shot" pairwise comparison.
*   **Result**: **`mp-19961`** was the winner (100% Win Rate). `mp-29950` dropped to 4th place.
*   **Observation**: The LLM penalized `mp-29950` for "messiness" or "dispersion," favoring the cleaner, straighter lines of `mp-19961`.

### The Discrepancy
The divergence between the "Numerical Winner" and the "Visual Winner" highlights that our current physics formula might over-prioritize local numerical bandwidth ($W$) while the vision model (and likely human experts) implicitly value global "cleanliness" and linearity.

---

## 3. Technical Discussion & Critique

### 3.1. Physical Definition of Flatness
Adachi-san emphasized that "Flatness" ($W \to 0$) is a proxy, not the ultimate physical goal.
*   **U/W Ratio**: The true driver of interesting physics (correlation effects, superconductivity) is the ratio of Coulomb repulsion ($U$) to Bandwidth ($W$).
*   **Partial Flatness**: A band does not need to be flat across the entire Brillouin Zone. A **Van Hove Singularity** (inflection point where $\nabla E = 0$) leads to a DOS peak and is physically sufficient.
*   **Constraint**: The current strict "global flatness" requirement might be excluding interesting candidates that are only flat in specific k-path segments.

### 3.2. Evaluation Philosophy: "Inverse Estimation"
A core realization was that **we do not know the "Ground Truth" function**. Manually debating whether "Isolation" should be weighted 0.1 or 0.2 is inefficient.
*   **Proposal**: Treat the expert's subjective ranking (e.g., Prof. Okabe's intuition) as the target $y$.
*   **Action**: Instead of "Forward Optimization" (picking weights $\to$ finding materials), we will use **"Inverse Optimization"**:
    1.  Select a small, diverse set of materials ($N \approx 20$).
    2.  Get expert labels (Human Preference $A \succ B$).
    3.  **Learn** the weights ($w_{flat}, w_{ext}, \dots$) that best reproduce these labels.

### 3.3. Vision Model Skepticism
Adachi-san noted that the Vision model returned "Confidence: High" for almost all decisions.
*   **Critique**: In valid scientific assessment, comparing two very similar flat bands should yield *low* confidence.
*   **Hypothesis**: The model might be "hallucinating certainty."
*   **Correction**: Future prompts should encourage "Tie" declarations or lower confidence scores for close calls.

---

## 4. Strategic Pivot: The "Human-in-the-Loop" Pipeline

The team agreed to separate the workflow into two distinct phases:

### Phase A: Calibrating the "Ground Truth" (Immediate)
1.  **Dataset Construction**: Select ~20 materials.
    *   Include: "Obvious Flat", "Obvious Non-Flat", and crucially **"Ambiguous/Intermediate"** cases.
    *   *Note*: The current dataset is too binary (Best vs Worst). We need "Middle" examples to train the discriminator.
2.  **Manual Labeling**: Ryotaro (and potentially other experts) will manually rank these pairs to establish a Golden Set.
3.  **Weight Regression**: Use this set to calculate the optimal weights for the Physics Function.

### Phase B: Generative Exploration (Medium Term)
Once the Evaluation Function is calibrated (Phase A):
1.  **Generation**: Use **Diffusion Models** (crystallographic generation) or **Graph Transformers** to generate new candidates.
2.  **Goal**: Demonstrate that the "Hit Rate" (probability of generating a flat band) increases when using the calibrated evaluator and agentic loop compared to random generation.
3.  **Pareto Frontier**: If a single "Best" weighting is impossible, plot the Pareto Frontier of the 4 metrics to show the trade-off surface.

---

## 5. New Idea: "Unsolved Problems" Benchmark
Adachi-san proposed a broader initiative inspired by the "Erdős Problems" or "LDS Problems" in mathematics.
*   **Concept**: Create a curated list of **"Unsolved Problems in Materials Science"** (e.g., specific contradictions in theory, un-synthesized but predicted structures).
*   **Usage**: Use this as a high-level benchmark for AI Agents. If an agent can propose a valid solution path to an open problem, it demonstrates true reasoning capability beyond simple "property optimization."
*   **Action**: Start compiling a list of such problems by consulting with theoretical physicists.

---

## 6. Action Items

| Owner | Task | Deadline |
| :--- | :--- | :--- |
| **Ryotaro** | **Dataset Expansion**: Select ~20 materials including "intermediate" flatness cases (random selection from MP). | Next Meeting |
| **Ryotaro** | **Manual Evaluation**: Perform pairwise comparison (Human Eye) on this set to create the "Golden Labels." | Next Meeting |
| **Ryotaro** | **Inverse Weighting**: Attempt to fit the Physics Function weights to match the Manual Labels. | Next Meeting |
| **Ryotaro** | **Pareto Plot**: Visualize the trade-off between the 4 current metrics (Flatness vs Isolation etc.). | Next Meeting |
| **Team** | **Schedule**: Next regular meeting set for **Wed 17:00 PST / Thu 10:00 JST**. | - |

---

## 7. Links & Resources
*   **Recording Transcript**: [Dropbox Link](https://www.dropbox.com/scl/fi/fl2cx4k3g6e2igcgergml/260211-lattice-lab.m4a?rlkey=o95rclsitd8xp0tvov5ten9eu&st=e2ygssec&dl=0)
*   **Zoom Summary**: [Zoom Doc](https://docs.zoom.us/doc/fEEcjhz4TVW03kv7fSq4zA)
