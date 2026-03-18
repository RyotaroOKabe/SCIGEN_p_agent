# Discussion Report: Flat Band Evaluation Progress (Feb 18, 2026)

**Participants:** Ryotaro Okabe (RO), Adachi-san (Collaborator/PI), and AI Agent  
**Location:** Lattice Lab / Virtual  
**Date:** February 18, 2026  
**Recording:** [260218-Lattice-lab.m4a](https://www.dropbox.com/scl/fi/ge0x1p3z0nzgkj46qlz3p/260218-Lattice-lab.m4a?rlkey=kupangneo2cu0eqc1tl50hbkg&st=qbtbznwh&dl=0)

---

## 1. Executive Summary
This report details the progress of the "Band Structure Evaluation" project discussed during the Feb 18 session. The meeting centered on the transition from static algorithmic scoring to human-in-the-loop preference modeling, the selection of optimal Pareto axes, and the integration of Vision-based LLMs for material discovery.

## 2. Detailed Discussion Points

### 2.1 AI Horizons: Knowing what LLMs don't know
- **UAI/ICML Context**: Adachi-san shared insights from the pure AI research side (UAI/ICML), specifically focusing on agents that can detect their own ignorance.
- **Scientific Search**: They discussed how a signal for "not knowing" (uncertainty) is crucial for scientific exploration. Instead of an LLM hallucinating an answer, it should signal uncertainty, which can then be used to guide the search for new materials.

### 2.2 Algorithmic Results & Pareto Frontier Axis Selection
- **The Axis Dilemma**: RO presented results using the `flat4metrics` framework but highlighted the difficulty in choosing the "best" two axes for visualization.
- **Selected Mapping**: We converged on **Flatness (X-axis, smaller W is better)** vs. **Isolation (Y-axis, larger Gap is better)** as the primary Pareto frontier.
- **Material Samples**: Discussed specific materials from the "Lattice Lab" (Lieb lattice-based) dataset. Notable samples like `lieb_000_01839` and `lieb_000_05561` showcase the trade-off between extreme flatness and poor isolation.

### 2.3 Human Evaluation Performance
- **High-Throughput Labeling**: RO reported answering **600 pairs** of band structures himself.
- **Workflow Speed**: Quantitative feedback shows that an expert can complete a set of 20 comparisons in approximately **2 minutes** (~6 seconds per pair), suggesting that intuitive visual judgment is extremely fast.
- **Bias Check**: Preliminary analysis of the 600 pairs showed no significant bias towards choosing the left (A) or right (B) image, validating the dataset's integrity.
- **Hosting**: Discussed moving the questionnaire from a free server (slow cold starts) to **GitHub Pages or Hugging Face** to facilitate broader expert feedback.

### 2.4 ML Strategy: Learning the "Expert Utility"
- **Inverse Preference Learning**: Use the 600 pairs to train an estimator for the human "Utility Function" ($U$).
- **In-Context Learning (ICL)**: Rather than fine-tuning, we discussed using a few hundred of these human-judged pairs as "prototypes" in the prompt for Vision-based LLMs to calibrate their judgments to match human experts.

## 3. Generative Future: RL & Diffusion
- **Generative Goal**: Adachi-san confirmed the goal of moving beyond evaluation into **active generation**.
- **Diffusion Models**: Guidance of crystal structure diffusion models using the learned human preference score as a reward signal.
- **Lattice Lab Private Repo**: RO will be invited to the private repository to work on these high-complexity generative tasks.

## 4. Closing Philosophical Note: Automation vs. Discovery
- **TRI (Toyota Research Institute) Comparison**: Discussed the difference between the "Classical" approach (solving known problems/automation) vs. "Scientific Discovery" (identifying new problems). 
- **The "Sexy" Approach**: Using LLMs to navigate the latent space of materials to find "beautiful" or "physically interesting" structures that algorithms might miss but humans identify intuitively.

---
**Next Meeting:** Feb 25, 2026 (Weekly Sync)  
**Action Items:**
- [x] Finalize `formula` column in `algo_results.csv` (Completed).
- [ ] invite RO to Lattice Lab GitHub organization.
- [ ] Prepare TOP 5 "Human-Preferred" band plots for manual review.
- [ ] Test LLM Vision calibration with the 600-pair benchmark.
