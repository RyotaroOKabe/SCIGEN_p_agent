# Discussion Report: Flat Band Material Discovery via Diffusion-DPO (Feb 25, 2026)

**Participants:** Ryotaro Okabe (RO), Masaki Adachi (MA / Lattice Lab), Fukui-san, Higuchi-san, and others
**Location:** Lattice Lab Meeting
**Date:** February 25, 2026
**Recording:** [260225-lattice-lab.m4a](https://www.dropbox.com/scl/fi/qcrfi2clqfmyg68f28rlc/260225-lattice-lab.m4a?rlkey=35g25nduls966s1oyznwnu10g&st=sfea64h8&dl=0)
**Zoom Transcript/Summary:** [Zoom AI Companion Notes](https://docs.zoom.us/doc/FJUFW9CcTEu5yZbmw2P5ww)

---

## 1. Executive Summary

The Feb 25 meeting had two main agendas. The first half was a literature review session prompted by MA's pre-meeting message: he identified a series of closely related papers from another group doing ML-based flat band genome search in 2D materials. The group studied the Overleaf draft that MA had prepared (a ChatGPT-generated LaTeX derivation of the Diffusion-DPO fine-tuning loss for SCIGEN) and discussed its validity together.

The core technical question that structured the second half of the meeting was: **what is the right paper to write given what we can actually do, and what exactly do the prior works fail at that we can do better?** This led to a three-level hierarchy of scientific contribution — extending the database (Level 1), discovering new structural clusters (Level 2), and finding human-unexpected materials (Level 3) — and converged on Level 1 as the immediate target.

The discussion also covered: the Hölder-DPO framework's mathematical guarantees and its surprising origin story; three data-collection strategies (crowdsourcing, expert-only, and async human-in-the-loop) with their trade-offs; the question of whether language-model integration is necessary for an initial paper; and an unexpected direction about **searching among Hölder-DPO's rejected outliers for novel flat band candidates**.

The meeting split next steps into two parallel tracks: MA leads the Diffusion-DPO algorithm implementation, and RO leads the discovery definition (database clustering analysis).

---

## 2. Opening Discussion: Agentic Workflows for Materials — Related Work Survey

### 2.1 *El Agente Sólido* (arXiv:2602.17886)

The meeting opened with a brief survey of prior work on agentic and automated workflows for computational materials science. One participant had spotted a LinkedIn post from **The Matter Lab at UofT** (Aspuru-Guzik group) promoting their work on autonomous solid-state simulations ([LinkedIn post](https://www.linkedin.com/posts/the-matter-lab-uoft_ai4science-chemai-autonomouschemistry-activity-7432057975998935040-ZYEI/?utm_source=share&utm_medium=member_desktop&rcm=ACoAAE3Te-gBKfjqV_MbrH-fMFxLbBPncQUznnk)). The underlying paper was identified as:

**[arXiv:2602.17886](https://arxiv.org/pdf/2602.17886)** — *"El Agente Sólido: A New Age(nt) for Solid State Simulations"* — Sai Govind Hari Kumar, Yunheng Zou, Andrew Wang, et al. (incl. Aspuru-Guzik), 2026.

The paper introduces a **hierarchical multi-agent framework** where large language models autonomously translate natural-language scientific objectives into executable solid-state computational pipelines. Key technical details noted in the discussion:

- **DFT backend**: uses **Quantum ESPRESSO** as the simulation engine. The system drives QE automatically from the LLM-generated instructions, handling input file construction, job submission, and output parsing.
- **Scope**: covers structure generation, DFT calculation execution, phonon calculations, and ML potential integration in a single agentic loop.
- **Rubric-based evaluation**: the system uses an LLM-readable rubric to judge whether each computational step produced a sensible result — e.g., whether convergence was achieved, whether the output is physically reasonable.
- **Reproducibility framing**: the paper frames the work as "lowering barriers to entry" and improving reproducibility in computational materials discovery — anyone with a natural-language request can, in principle, run a solid-state simulation without knowing the details of QE input formats.

**Code availability:** Discussed as **not open-source** (the group noted this twice for emphasis). The LinkedIn post and paper describe the method but the code is currently closed.

**Discussion comment on accessibility:** Someone remarked that the paper and LinkedIn post are written in a "magazine-like" style — accessible and enjoyable to read, like a *Newton* science magazine article, rather than a dense technical methods paper. This is an intentional presentation choice by the Aspuru-Guzik group.

**Discussion comment on reproducibility:** The group noted:
> "If the method is documented clearly enough in the paper, you could potentially reproduce it using agent-based coding — just give an LLM the method description and have it write the implementation. Whether this paper reaches that level of clarity remains to be seen."

### 2.2 ChemRefine ([github.com/sterling-group/ChemRefine](https://github.com/sterling-group/ChemRefine))

A second related automated workflow tool mentioned during the survey: **ChemRefine** from the Sterling group. Key details:

- Designed for **molecular chemistry** (conformer sampling, geometry optimization, transition state discovery) rather than crystalline solid-state materials.
- Backend: **ORCA 6.0+** quantum chemistry package for DFT.
- Pipeline: automated conformer generation (XTB via GOAT) → optional ML potential refinement (MACE or FairChem) → DFT geometry optimization → high-level single-point energy calculation. Each stage feeds into the next with energy-window filtering and Boltzmann population analysis between steps.
- HPC integration: SLURM job scheduler support with caching/error recovery.

While ChemRefine targets molecular rather than solid-state systems, it illustrates the same general philosophy: **progressive refinement with automated inter-stage filtering**, reducing the need for manual intervention between levels of theory.

### 2.3 How These Relate to the QMG Project

The group's takeaway from this quick survey:

> "There are already groups working on automated solid-state simulation pipelines. *El Agente Sólido* automates the *execution* of known calculations. Our work is different: we're not automating known calculations — we're using *human preference* to define what a good calculation target looks like. The bottleneck we address (what makes a material interesting?) is upstream of what El Agente Sólido addresses (how do you run the calculation once you know what to look for)."

This positioning — **preference-based target selection** upstream of automated execution — is distinct from all the agentic workflow papers surveyed. El Agente Sólido assumes you already know what property to look for; the QMG project is solving the problem of *how to define and learn* what to look for.

---

## 3. Related Work: The 2DMatPedia Flat Band Genome Series

### 3.1 Papers Identified (Adachi-san's Pre-Meeting Share)

MA flagged a series of papers from a group that appears to have been working intensively on ML-based flat band discovery in 2D materials. The group confirmed the series has at least **three papers**:

1. **[arXiv:2207.09444](https://arxiv.org/abs/2207.09444)** — *Machine learning approach to genome of two-dimensional materials with flat electronic bands* — the foundational paper, uses autoencoders / variational autoencoders to characterize and search for flat band materials within the **2DMatPedia** database.
2. **[Nature Communications Physics (2025)](https://www.nature.com/articles/s42005-025-01936-2)** — A follow-up from the same group.
3. **[arXiv:2506.07518](https://arxiv.org/abs/2506.07518v1)** — The latest preprint; this one predicts a flat band indicator **from crystal structure alone** using a **graph neural network (GNN) combined with a language model backbone**. Concretely: the crystal structure goes through the GNN+LM pipeline, a flat band score is output, and materials are then ranked by score.

Their code/models are **not open-source** (described in the meeting as "LI-limited" — access-restricted). There was a brief discussion about whether they might be reproducible via agent-assisted coding if the methods were described clearly enough in the paper.

### 3.2 What the Prior Work Actually Does (and How to Understand It)

The key pipeline in the latest paper (arXiv:2506.07518) is roughly:

```
Crystal structure → GNN + Language Model → flat band score → ranking
```

This means: given a crystal, the model predicts how likely it is to have a flat band, without ever running DFT or looking at band structures. The score is purely structural. The search space is **restricted to 2DMatPedia**, i.e., structures already in the existing database; no new structures are generated.

The earlier paper (2207.09444) uses autoencoders/VAEs to embed materials in a latent space, cluster them, and explore the database. The group inside the database does **clustering to find diversity**: they try to cover different "types" of flat band materials that exist in the database, then report what structural families they found.

### 3.3 Limitations of the Prior Work — What They Cannot Do

This was a central discussion point. The specific question posed: *"If their black-magic image- or structure-based approach can already find flat bands, why do we need more?"*

The group identified two critical limitations of the prior work:

1. **Scoring criteria are incomplete.** The prior work scores materials on "flatness" alone — essentially asking: "how flat does the band look (or how flat is the predicted band)?". It does **not** capture:
   - **Proximity to the Fermi level**: a flat band deep below/above E_F is much less interesting than one sitting right at E_F.
   - **Isolation from other bands**: a flat band that is energetically isolated (gap on both sides) is far more useful for studying flat-band physics than one buried in a spaghetti of crossing bands.
   These two criteria are exactly what the multi-criteria human preference approach encodes.

2. **Search is confined to the database.** Because the prior work is discriminative (predict a score for a given structure), it can only evaluate structures that already exist in 2DMatPedia. It cannot propose genuinely new structures. The QMG group's generative model (SCIGEN) can, in principle, generate structures outside this distribution.

The framing agreed upon for the paper: "The prior work is an impressive first step. However, it does not account for proximity to the Fermi level or band isolation — criteria that are central to the physical usefulness of a flat band. Our approach captures these through human preference, and our generative model enables discovery beyond the database boundary."

---

## 4. Scientific Contribution: A Three-Level Hierarchy

Fukui-san proposed thinking about the project's potential contribution along a hierarchy of three levels, in increasing order of impact and difficulty:

### Level 1 — Generate Flat Band Materials Not in Any Existing Database

**What it means:** Produce novel crystal structures (via SCIGEN fine-tuned on the preference data) that, when DFT is run, show flat bands near the Fermi level with good isolation — and that are demonstrably not present in 2DMatPedia or other standard databases.

**How to achieve it:** The concrete workflow discussed:
1. Fine-tune SCIGEN on the current preference dataset so it preferentially generates flat-band-like materials.
2. Generate a large number of structures continuously.
3. Filter for: (a) structural stability (DFT calculation or a cheap proxy), (b) genuinely novel (not in database), and (c) flat band confirmed by DFT band structure.
4. Present the surviving candidates.

This does **not require language integration or quality-diversity optimization** — simple generate-and-filter is sufficient. As Fukui-san said explicitly: "If we just want to get to Level 1, we don't need to involve language at all. Generate, screen afterwards, done."

**Why it is publishable even on its own:** Every result is a genuinely new candidate material with a flat band. The prior work cannot produce these because it only searches within databases.

**Scaling:** Fukui-san noted that if you generate enough structures, you will almost certainly stumble on new structural clusters as a byproduct, effectively providing a path toward Level 2 without additional effort.

### Level 2 — Discover a New Structural Family / Cluster of Flat Band Materials

**What it means:** Find that the generated materials fall into a cluster in the embedding space that does not correspond to any existing structural family in the database. For example, existing flat band families include kagome, Lieb, honeycomb; a new cluster would be an analogous family not previously recognized.

**How to operationalize it:** Requires:
- Choosing an **embedding space** for materials (crystal structure embedding, or band structure embedding — see Section 7).
- Mapping the **existing database** into that space to identify current clusters.
- Showing that generated materials produce a **new, populated cluster** in that space.
- Ideally, interpreting the new cluster in terms of structural motifs or symmetry groups.

The group noted that the prior work (2207.09444 series) also does clustering, but strictly within the database. Their clusters correspond to known structural motifs (kagome, honeycomb, space group families). A "new cluster" from the generative model would constitute a genuine discovery of a new flat band family.

**On the nature of flat band diversity:** One participant raised the question of what the right "axes" of flat band diversity are. One proposed axis: **momentum-space extent of the flat band** — is the band flat across the entire Brillouin zone, or only along certain high-symmetry lines? This corresponds to qualitatively different physics. Unlike discrete space group classifications, this type of variation is more **continuous**, meaning clustering in this space would produce a soft/continuous separation rather than discrete labeled families.

### Level 3 — Find a Material That Humans Would Not Have Predicted

**What it means:** The generative model proposes a structure that, by prior physics knowledge or intuition, would not have been expected to host a flat band — yet DFT confirms it does. The example used in the discussion: "Ah, *this* structure has a flat band? I never would have thought of that."

**Why it is hard:** Defining "what a human wouldn't have thought of" is rigorous only if you can show that the structure is absent from human intuition — hard to prove formally. Nonetheless, impact-wise this is the most powerful claim.

**Relation to prior work:** This is analogous to discovering a new allotrope of carbon (graphene) or a new topological phase — entirely unexpected from existing structural databases.

The group agreed: **Level 1 is the immediate target**; Level 2 is a stretch goal; Level 3 is aspirational but would be "very impactful if it happens."

---

## 5. Adachi-san's Proposed Pipeline

Before the meeting, MA sent a detailed proposal in Slack, including an Overleaf draft:
[Overleaf: Diffusion-DPO Loss Derivation](https://www.overleaf.com/4673732276ftsphvtjnbmm#f2756f)

The group looked at this Overleaf document together during the meeting. The pipeline has five steps:

### Step 1: Construct a Large Preference Dataset

Use the `band_eval_web` pairwise comparison questionnaire to collect human judgments. For each pair of band structure images (A vs. B), an annotator labels which one shows a flatter / more desirable band near E_F. The three scoring axes — **flatness**, **isolation**, **proximity to Fermi level** — serve as conventional quantitative sanity checks and for dataset analysis, but the primary input for fine-tuning is the raw pairwise preference labels.

Crowdsourcing from multiple annotators (not just experts) is the preferred strategy for scale; the Hölder-DPO framework handles noisy/wrong labels automatically (see Section 5).

### Step 2: Fine-tune SCIGEN with Diffusion-DPO (Hölder Version)

Apply the **Diffusion-Hölder-DPO loss** to fine-tune SCIGEN so it generates flat band materials preferentially. The derivation is laid out in the Overleaf draft ([PDF local copy](Diffusion_finetune.pdf); [Overleaf](https://www.overleaf.com/4673732276ftsphvtjnbmm#f2756f)), which was reviewed together during the meeting. A summary of its key technical content:

**Crystal representation.** A crystal is represented as $\mathbf{x} = (\mathbf{L}, \mathbf{X}, \mathbf{A})$, where $\mathbf{L} \in \mathbb{R}^{3\times3}$ is the lattice matrix, $\mathbf{X} \in [0,1)^{N\times3}$ are fractional coordinates, and $\mathbf{A} \in \{1,\ldots,|\mathcal{A}|\}^N$ are atom types.

**Preference dataset.** $\mathcal{D} = \{(c_i, \mathbf{x}^w_i, \mathbf{x}^l_i, \kappa_i)\}$, where $c_i$ is the generation condition, $\mathbf{x}^w_i$ / $\mathbf{x}^l_i$ are the preferred / rejected crystals, and $\kappa_i \in \{1,2,3,4,5\}$ is the annotator's confidence score collected via `band_eval_web`.

**From standard DPO to diffusion.** Standard DPO operates on discrete token log-probabilities. For a diffusion model the policy log-ratio is approximated per denoising step using the denoising error:

$$\log \frac{p_\theta(\mathbf{x}_0)}{\text{ref}} \approx \omega_t \left(\|\boldsymbol{\varepsilon} - \boldsymbol{\varepsilon}_{\mathrm{ref}}(\mathbf{x}_t,t)\|^2 - \|\boldsymbol{\varepsilon} - \boldsymbol{\varepsilon}_\theta(\mathbf{x}_t,t)\|^2\right)$$

where $\omega_t$ is a time-dependent weighting and $\boldsymbol{\varepsilon}$ is the ground-truth noise.

**SCIGEN-aware masking.** SCIGEN constrains certain structural channels (e.g., fixed lattice motif) via a binary mask $\mathbf{M}^{(z)} \in \{0,1\}^{\text{shape}(z)}$. The constrained noisy crystal at time $t$ is:

$$\mathbf{x}_t^{\mathrm{SCI},(z)} = \mathbf{M}^{(z)} \odot \mathbf{x}_t^{\mathrm{con},(z)} + \bar{\mathbf{M}}^{(z)} \odot \mathbf{x}_t^{\mathrm{unk},(z)}$$

The DPO gradient must only flow through the **unconstrained** (unknown) channels; constrained channels are excluded from the denoising error. The per-channel improvement score is:

$$I_\theta(\mathbf{x}_0, t) = \sum_{z} \omega_t^{(z)} \left( E_{\mathrm{ref}}^{(z)}(\mathbf{x}_0,t) - E_\theta^{(z)}(\mathbf{x}_0,t) \right)$$

The pairwise margin is $g_\theta(t) = I_\theta(\mathbf{x}^w,t) - I_\theta(\mathbf{x}^l,t)$, and the **SCIGEN-DPO loss** is:

$$\mathcal{L}_{\mathrm{SCI\text{-}DPO}}(\theta) = \mathbb{E}_{(c,\mathbf{x}^w,\mathbf{x}^l)\sim\mathcal{D}}\, \mathbb{E}_{t\sim\mathrm{Unif}} \left[-\log\sigma\!\left(\beta\, g_\theta(t)\right)\right]$$

**Hölder robustification.** To handle noisy labels the loss is modified to be Hölder-continuous in $p_\theta(t) = \sigma(\beta g_\theta(t))$:

$$\mathcal{L}^{(\gamma)}_{\mathrm{H\text{-}DPO}}(t) = -(1+\gamma)\,p_\theta(t)^\gamma + \gamma\,p_\theta(t)^{1+\gamma}$$

for Hölder exponent $\gamma > 0$. This zeroes out the gradient for adversarially extreme labels (relaxing property). Confidence scores $\kappa_i$ from the questionnaire are incorporated as per-sample weights, down-weighting low-confidence annotations automatically.

**Outlier detection.** The document also derives a contamination estimator that scores each data point by how anomalous its label is, enabling post-hoc ranking of the most "surprising" rejected items (see Section 6.5).

**What DPO training requires** — RO asked explicitly: "For building the DPO loss, do we need the band structure images?" Answer from discussion: **No.** What the DPO loss needs is:
- The **preference dataset**: $(c, \mathbf{x}^w, \mathbf{x}^l, \kappa)$ quadruples.
- The **model's denoising network** $\boldsymbol{\varepsilon}_\theta$ and a frozen **reference copy** $\boldsymbol{\varepsilon}_{\mathrm{ref}}$.
- The **SCIGEN mask** $\mathbf{M}^{(z)}$ specifying which channels are constrained.

Band structure images are used to generate the labels (humans judge the band structures), but they are not fed into the DPO loss itself.

**Implementation:** Use the **TRL (Transformer Reinforcement Learning) library** from HuggingFace:
> "There is already a fine-tuning library for LLMs — HuggingFace's TRL. You plug in the custom loss function, give it the data, and it handles the training loop. So if we just provide the loss, we can reproduce the training by plugging in our preference data."

The plan: write a custom loss class implementing $\mathcal{L}^{(\gamma)}_{\mathrm{SCI\text{-}H\text{-}DPO}}$, drop it into TRL, feed in the preference dataset, and run training on SCIGEN.

### Step 3: Language-Guided Generation (Optional / Future Work)

Add pretraining that couples a **language model** to SCIGEN, enabling natural-language prompts to steer generation (e.g., "generate a kagome-type material with transition metals"). This enables curiosity-driven, agentic exploration:

> "If you can control SCIGEN with language, you can think of something you want and just ask it. That makes the exploration much more interactive and agentic."

However, the group explicitly discussed whether this is **needed for an initial paper**. Consensus: **probably not**. If the goal is Level 1 (generate new flat band materials), language guidance is unnecessary — simple generation + screening is sufficient. Language guidance becomes relevant if the paper wants to push the agentic/interactive discovery angle.

### Step 4: Quality-Diversity (QD) Exploration

After fine-tuning, use **quality-diversity (QD) optimization** (e.g., MAP-Elites) or LLM-guided search over the material space to find **diverse** flat band solutions rather than converging to a single optimum. The goal is to populate the space of novel flat band types, not just to generate one good material.

### Step 5: DFT Validation and Selection

Run DFT on promising candidates, confirm flat bands, analyze the results physically, and present in the paper with clustering analysis showing what new types were found.

---

## 6. Hölder-DPO: Technical Details, Mathematical Guarantees, and Origin Story

### 5.1 The Published NeurIPS Paper

The foundational paper: **[arXiv:2505.17859](https://arxiv.org/abs/2505.17859)** — *"Scalable Valuation of Human Feedback through Provably Robust Model Alignment"*.

This introduces **Hölder-DPO** (named after the Hölder continuity property used in its derivation), a variant of Direct Preference Optimization (DPO) designed for noisy crowdsourced labels.

### 5.2 What Hölder-DPO Does, Technically

Standard DPO assumes all preference labels are correct and trains the model to maximize the probability of the preferred item. Hölder-DPO modifies this by replacing the standard loss with a **Hölder-continuous approximation** based on a **Taylor expansion**. The key properties:

**Mathematical guarantee (Relaxing Property):**
> "Even if the most adversarially bad label enters the dataset — say, one data point whose label is completely wrong and would, in standard DPO, maximally distort the result — the algorithm guarantees that this data point's contribution to the loss is **zeroed out**. Its influence is literally zero, even in the worst case."

More precisely: the loss is constructed such that for any single adversarial outlier, the gradient contribution is bounded to zero regardless of how extreme the label is.

**Taylor expansion interpretation:**
> "The algorithm is based on a first- to second-order Taylor expansion approximation. Up to second order, we can prove that this approximation correctly recovers the majority vote signal."

The mathematical guarantees are therefore: (1) **zeroed-out adversarial influence** for any single outlier (relaxing property), and (2) **second-order Taylor approximation accuracy** for the majority signal recovery.

**Per-label scoring:**
As a byproduct, the algorithm can **rank** each label by how "abnormal" it is. Concretely: each data point gets an anomaly score that reflects how much it deviates from the majority signal. This is not the primary purpose, but it falls out naturally from the loss computation.

**Ground truth assumption — Majority Vote:**
> "We assume that what the majority of annotators agree on is the correct label. Individual annotators might differ, but the majority consensus reflects the true preference. The algorithm is designed to recover this majority signal robustly."

This is the correct assumption for this project: individual physicists may differ on edge cases, but there is broad consensus on clear-cut flat bands. If everyone says "this one is flat," it's probably flat.

### 5.3 The Origin Story (Not Just Noise Elimination)

Fukui-san revealed a surprising origin story for the method:

> "When I originally designed this, my goal was NOT to eliminate noisy labels. Actually, I was MORE interested in the noise — in the anomalous cases. In science, the most interesting results are the ones that existing knowledge cannot explain. An experiment that breaks the current theory is where progress happens. I wanted to detect those automatically: 'Can we automatically flag the outliers that no current theory can explain?'"

So **the original motivation was to identify scientifically interesting anomalies**, not to clean data. The method automatically identifies what deviates from majority consensus — originally to find scientifically interesting outliers, and in this project repurposed to find (and down-weight) label noise.

The implication for this project: see Section 5.5.

### 5.4 Deriving the Diffusion Version (Unpublished)

The published Hölder-DPO is derived for **LLMs** (autoregressive models with discrete token distributions). The **diffusion model version** requires adapting the derivation because the model family is different (continuous vs. discrete, score-based vs. autoregressive).

From the discussion:
> "It's LLMs in the published version. If we want to apply it to SCIGEN, we need the diffusion version. LLMs and diffusion models are subtly different, so the derivation needs to be rewritten — the formulas and proofs. But honestly, it wasn't hard. I handed it off to a statistics collaborator and they finished it quickly. Then they got bored and moved on to another topic."

**Status:** The diffusion version **exists and is mathematically correct** but has **never been published or submitted**. It is currently buried as an unused result. This is the unpublished methodological contribution available for the current paper.

The Overleaf draft that MA shared was generated with ChatGPT assistance and contains the LaTeX derivation of this loss. The group reviewed it together to assess whether it is correct and complete.

### 5.5 The Rejected Outliers as a Discovery Strategy

An important observation raised during the discussion: because Hölder-DPO produces a ranked list of anomalous labels, we can examine what is in the "rejected" / high-anomaly set.

> "Some of the rejected items will just be mistakes — an annotator clicked the wrong one. But some might be genuinely interesting: a material that *most* people think is not a good flat band candidate, but one or two careful annotators thought was interesting. The algorithm calls this an outlier and down-weights it. But what if that outlier is actually a new type of flat band that looks unusual to most people precisely because it's different from the familiar patterns?"

Fukui-san:
> "You can use the anomaly scores to rank the rejected items. The ones at the very top of the anomaly ranking are the most worth examining. They are either genuine mistakes, or genuinely surprising materials. You can't tell which from the label alone — but if you run DFT on them and find a real flat band, that's a discovery. Those might be where Level-3 contributions are hiding."

Higuchi-san asked whether this is a different mechanism from the DPO-based approach: it is the **same** mechanism. Hölder-DPO both (a) down-weights noisy labels during training and (b) scores each label by anomaly degree — these are two sides of the same operation.

### 5.6 Academic Attribution Challenge

A brief side discussion about the NeurIPS Hölder-DPO paper: at the NeurIPS venue, people from Google and others asked whether the method could be used for ChatGPT fine-tuning, and the answer was "yes, technically." However, since ChatGPT and similar systems are **closed models**, any application of the method by those organizations would not result in public citations or attributable impact.

> "It's not satisfying to have your method used internally by a closed model. You want people to cite the paper. But if the model is closed and no paper comes out about it, the citation never happens."

This motivates applying the method to **open, publishable settings** — exactly what this materials discovery project provides.

---

## 7. Data Collection: Strategies, Trade-offs, and the Scale Question

### 6.1 The Three Strategies

#### Strategy A — Crowdsourcing (Cheap, Noisy Labels)
- Collect pairwise labels from many annotators via the `band_eval_web` interface.
- Labels are fast and cheap; individual annotators may make mistakes.
- Hölder-DPO handles the noise automatically.
- **This is the direction the project is currently heading.**

The website enables this strategy: anyone with access can log in, complete a session of 20 comparisons in ~2 minutes, and generate labeled pairs.

#### Strategy B — Expert-Only Labeling (Reliable, Expensive)
- Use only labels from high-trust experts (e.g., the PI, senior collaborators).
- Each label is highly reliable, but expert time is scarce.
- Natural companion: **active learning** — select the most informative pairs to show the expert, minimizing the number of expert queries needed to build a good model.

#### Strategy C — Async Human-in-the-Loop (A Hybrid Framework)
This was presented as a novel methodological framework that differs from standard active learning in an important way:

**Standard active learning:** when the algorithm identifies an uncertain case it needs labeled, it sends the query to the human, and the human **must respond immediately** (or at least within the current training loop iteration). This creates a real-time synchronization requirement that is **very inconvenient** for humans.

**Async human-in-the-loop (the proposed approach):**
> "Instead of requiring an immediate response, uncertain cases go into a 'hot pool.' The human is notified, but they can respond whenever they have time — today, tomorrow, next week. Meanwhile, the system keeps running. It only makes predictions on cases it is already confident about. The uncertain cases are simply skipped."

The key property:
> "While waiting, the system is not degrading in performance. It's only doing what it's confident about, so the performance on those cases is fine. When the human eventually comes back and labels the pool, the model gets a large batch update. After the update, the set of confident cases expands, and the exploration range widens."

This is framed as "a more human-considerate framework" — the human is not enslaved to real-time query answering. The system gracefully degrades to a conservative mode while waiting, then jumps forward when human input arrives.

### 6.2 The Expert vs. Cheap Label Correlation Model

A more technically nuanced discussion on how to combine cheap labels (e.g., from RO or other grad students) with expert labels (e.g., from the PI):

> "Assume cheap labels and expert labels are correlated but not perfectly correlated. In regions of the materials space where they agree well, there's no need to ask the expert — cheap labels are sufficient proxies. In regions where they disagree, that's where expert input is most valuable. Active learning should query the expert precisely in those disagreement regions."

This requires:
- An initial calibration set: **some expert labels on a subset** of pairs, to estimate where cheap and expert labels agree/disagree.
- This initial calibration set does not need to be large — enough to estimate the correlation structure.
- After calibration, targeted active learning queries the expert only on pairs in the disagreement regions.

The practical schedule suggested:
- **Stage 1:** Collect many cheap labels (crowdsourcing) + a small number of expert labels (for calibration). No need to gather all expert labels at once.
- **Stage 2 / 3:** Use the calibration to identify disagreement regions; add targeted expert queries at those points.

### 6.3 How Much Data Is Needed?

RO raised a practical concern: as of this meeting, the preference dataset covers approximately **6 materials** worth of comparisons. Is this enough for fine-tuning SCIGEN?

> "Honestly, I'm not sure. We'd have to try. If there are already 6 materials, that's actually quite a lot relative to what I was originally imagining — I was thinking we'd do maybe 10. I had been thinking in-context learning might be sufficient instead of fine-tuning."

**In-context learning** (ICL) was mentioned as an alternative: instead of gradient-based fine-tuning on the preference data, use the labeled pairs as few-shot examples in the model's context at inference time to steer generation toward flat band materials. This requires no training but may be less effective for SCIGEN (a diffusion model, not an LLM).

RO also raised a data quality concern:
> "Some of the existing labels were done quickly — clicking through rapidly. I'm a bit worried that some of those are unreliable. Maybe even with fewer pairs, it's better to have done them more carefully?"

Fukui-san's response: Hölder-DPO addresses this exactly.
> "Our method filters out the suspicious labels — the 'clicked through quickly' ones will look like noise and get down-weighted. The hypothesis is: a large noisy dataset filtered by Hölder-DPO produces a cleaner effective subset that outperforms a smaller but carefully curated dataset. That's the whole point."

---

## 8. Embedding Space for Clustering: Crystal Structure vs. Band Structure

### 7.1 The Question

For Level 2 (discovering new structural clusters), a critical choice is: **in what space do we embed materials to define clusters?** Two main options were discussed:

**Option A — Crystal structure embedding**: embed the crystal structure (atom types, positions, symmetry) using a structural GNN or symmetry-aware fingerprint. This is the approach used by the prior work (2DMatPedia series). Cluster structure corresponds to structural motif families (kagome, Lieb, honeycomb, space groups).

**Option B — Band structure embedding**: embed the computed band structure (or a predicted band structure) in some latent space. The rationale: two structurally very different materials might have similar band structures (flat band topology), while two structurally similar materials might differ in their band structure.

### 7.2 Technical Proposal for Band Structure Embedding

One participant mentioned the existence of models that **predict band structures from crystal structure** (referred to as "NGND" or a similar model name in the discussion — the exact reference was unclear from the transcript). The proposed pipeline:

```
Crystal structure → [band structure prediction model] → predicted band structure
    → [learned embedding / dimensionality reduction] → latent representation
```

Then cluster in this latent band-structure space rather than the crystal structure space. This would group materials by their **electronic properties** rather than their **atomic arrangement** — a fundamentally different and potentially more physically meaningful notion of "flat band family."

The concrete claim this enables: "We found materials whose band structures cluster together in a new region of band structure space, corresponding to a new type of flat band physics — even if their crystal structures look nothing alike."

### 7.3 Crystal Structure vs. Band Structure Clustering: Physical Interpretation

A subtlety raised in discussion: crystal structure clusters (space groups, Wyckoff positions) produce **discrete** classification. Band structure clusters, especially along axes like "how much of the Brillouin zone is covered by the flat band" or "how sharp is the flat band dispersion," may produce **more continuous** distributions. This means:

- Crystal structure clustering: "This material is a kagome material."
- Band structure clustering: "This material sits at this point in the continuous spectrum of flat band types."

The latter may be harder to interpret discretely but could reveal richer physical content. One participant suggested that **both embeddings** could be used simultaneously, asking: where do the crystal structure clusters and band structure clusters agree, and where do they diverge? The divergence regions are potentially the most interesting.

---

## 9. Paper Positioning: Method vs. Discovery vs. Workflow

### 8.1 The Core Decision

The group spent significant time on what the paper should primarily argue for. Four options:

1. **Method push:** "We developed Diffusion-Hölder-DPO, a novel algorithm for fine-tuning generative models with crowdsourced noisy human preference data. We demonstrate it on materials discovery."
2. **Discovery push:** "We found new flat band materials not in any existing database, including a new structural family."
3. **Workflow push:** "We present an end-to-end workflow: preference data collection → generative model fine-tuning → candidate screening → DFT validation."
4. **Problem formulation push:** "We argue that the bottleneck in AI-driven materials discovery is not generation or DFT, but *human judgment about what is interesting*. We formalize this as preference learning and demonstrate it on flat bands."

Fukui-san summarized the trade-off: "The more methods you pile on, the harder the paper is to read and to access. Simplicity has value."

### 8.2 Consensus

**Discovery should be the headline, supported by methods:**

> "If we discover something genuinely new — a material not in any database, especially a new type of flat band — that's the headline. The method (Diffusion-DPO, Hölder robustness) is what made the discovery possible, and it's novel, but it should be the supporting argument. For Nature-level journals, discovery always wins."

**Why the method is still pushable (not just supporting):**
- Diffusion-Hölder-DPO for materials fine-tuning: unpublished, technically novel.
- Crowdsourced human preference for band structure evaluation: not done before.
- The formal robustness guarantee (relaxing property) in a materials discovery context: novel application.
- Hölder-DPO's ability to surface anomalous labels as potential discoveries: novel use case.

**Generality of the framework:**
> "This is not specific to flat bands at all. Any property of band structures that requires human subjective judgment — Dirac cones, superconductivity signatures, topological features — could use this framework. Band structures are particularly well-suited because they are harder to evaluate automatically than scalar or 1D data like XRD patterns or formation energies. For those, you can just compute a number. For band structures, you can't easily capture 'this is interesting physics' with a single number. That's exactly where human preference adds irreplaceable value."

### 8.3 On the Complexity Budget

Fukui-san made an explicit complexity-management point:
> "In our typical methods-focused papers, we push a new algorithm. Here, the question of how to push is less clear, so we have options. But the more complex the method, the harder it is to read and the more surface area for criticism. My instinct is: use what we have, run with it, and let the discovery speak. We can always add methodological complexity in a follow-up."

This directly informed the decision to treat language-guided generation (Step 3 of MA's pipeline) as **future work** rather than a requirement for the initial paper.

---

## 10. Broader Vision: Citizen Science, Intuition, and the AI Age

### 9.1 The NASA Citizen Science Analogy

Toward the end of the meeting, a participant (Fukui-san or MA) drew an analogy to **NASA's citizen science programs**:

> "NASA has too many astronomical images to annotate — no staff for it. So they opened it up: anyone, including housewives and hobbyists, can go to the website and annotate images of stars. If a citizen annotator finds a new celestial object, they get to **name it**. This creates a powerful incentive: your action might actually contribute to a scientific discovery, and you get your name attached to it. People find it fun and meaningful, not tedious."

The proposal: a similar model for materials discovery. The `band_eval_web` site enables experts to compare band structures, but in principle, the framework could be opened to a broader audience. If a non-expert's annotation contributes to finding a new flat band material, they could be credited.

> "We already have generation, evaluation, and a comparison website. All we need is curiosity-driven annotation. That's what the citizen science angle provides."

### 9.2 The Role of Human Intuition in the Age of AI

A broader reflection from the discussion:

> "Now that generation is cheap (SCIGEN can produce thousands of structures) and DFT evaluation is semi-automated, the bottleneck is actually **curiosity** — the ability to recognize that something is interesting. In the old days, generation and evaluation were the bottlenecks. Now it's human attention and judgment. Human preference, formalized into preference data, is exactly the solution to this bottleneck. The website we built is a step toward institutionalizing that judgment."

> "I've been thinking: as generation and evaluation become more automated, what becomes most valuable is the ability to **find interesting things** — intuition. Preference data is one way to formalize intuition. If you can hook up language to it, you could even have the system ask you what you find interesting and then go look for it."

### 9.3 Extensions Beyond Flat Bands

Several extensions were mentioned as directions for future preference-based discovery:

- **Dirac cone identification**: Band crossings are ubiquitous; identifying the topologically meaningful ones (graphene-like linear Dirac cones) from band structure images could be a preference task. An expert looks at a band structure and immediately knows whether the crossing near E_F is a true Dirac cone — formalizing this judgment as preference data seems feasible.

- **Superconductivity signatures**: Experts can look at a band structure and notice features associated with superconductivity (van Hove singularities, particular band topology near E_F). This is currently completely informal — done by domain experts via visual inspection. Preference-based formalization could scale this judgment.

- **Higher-order band flatness**: Instead of asking "is $\partial E / \partial k = 0$?" (first-derivative flat = zero velocity), one could ask "is $\partial^2 E / \partial k^2 = 0$?" (second derivative = zero effective mass diverges, quadratic band touching) or even higher orders. These conditions correspond to qualitatively different physical phenomena (quadratic band touching vs. linear, heavy fermion vs. flat) and each could be a separate preference task. The observation from the discussion: "If you added the first- and second-derivative plots alongside the band structure in the questionnaire, experts could evaluate higher-order flatness too. Maybe too niche for this paper, but interesting."

- **The questionnaire as a general platform**: The `band_eval_web` questionnaire is already a general pairwise comparison platform. Any comparison task over band structure images could be added. One could even imagine showing the band structure + derivatives simultaneously and asking "which has a more interesting quadratic touching near E_F."

---

## 11. SCIGEN: Architecture and Generation Constraints

For completeness, the technical details of SCIGEN discussed:

### 10.1 Base Architecture

SCIGEN is built on **DFCSP (Diffusion model for Crystal Structure Prediction)**, which was published previously. The key question that came up: "Is SCIGEN a wrapper on top of DFCSP, or did it modify the internals?"

Response:
> "SCIGEN is not a wrapper — it modified the internals of DFCSP. The training is identical to DFCSP (no changes there). The constraint/control mechanism is added in the **generate / sampling function** only. So training: DFCSP as-is. Inference: modified generate function."

This is analogous to **image inpainting in diffusion models**: during inference, partial conditioning is added (e.g., "fix these pixels, generate the rest") without changing the model weights. SCIGEN adds lattice motif and atom-type constraints in the same way, at sampling time.

### 10.2 What Can Be Specified at Generation Time

Three types of constraints:
1. **Lattice motif**: specify the desired structural motif — kagome, Lieb lattice, honeycomb, etc. This restricts the generated structure to have the chosen motif as a sublattice.
2. **Atomic species per site**: for the chosen motif, specify which atoms go at which Wyckoff positions. Example: "put transition metal X at the kagome sites."
3. **Total atoms per unit cell**: specify the total number of atoms in the unit cell.

All three are optional: if not specified, the model samples freely. This means you can: (a) specify everything (fully constrained generation), (b) specify motif only and let the model choose atoms, or (c) generate completely freely.

---

## 12. Action Items and Next Steps

### Track A — Algorithm (Masaki Adachi leads)

1. **Finalize the Diffusion-Hölder-DPO loss derivation**: build on the existing Overleaf draft; produce a clean, mathematically verified LaTeX derivation (not just ChatGPT-generated output — verify the formulas carefully).
2. **Implement as minimum working code**: write a custom loss class compatible with the SCIGEN codebase and the HuggingFace TRL library. Goal: a script that takes (preferred structure, rejected structure) pairs and trains SCIGEN with the Hölder-DPO loss.
3. **Hand over to RO**: the code should be runnable with the existing preference data once handed over. RO will plug in the data and run experiments.

### Track B — Discovery Definition (RO leads, with Higuchi-san)

1. **Choose embedding space(s)**: decide between crystal structure embedding, band structure embedding, or both. Involve the "band predictor" person mentioned in discussion for band structure embedding expertise.
2. **Cluster the existing database**: apply the chosen embedding to 2DMatPedia (or the Lieb lattice subset from prior work), and map out the current cluster structure (which structural families exist, where they sit in the space).
3. **Define "novelty"**: operationalize Level 2 — what does it mean for a generated material to fall in a "new cluster"? Options: (a) distance from nearest existing cluster center exceeds a threshold, (b) falls in a low-density region of the database embedding, (c) has a different structural signature (space group + Wyckoff position combination) from any existing database entry.

### Outstanding Open Questions

- [ ] **DFT throughput**: How many generated structures can realistically be validated per day/week? This determines how large the screening pipeline can be for Level 1.
- [ ] **ICL vs. fine-tuning**: At what scale does in-context learning become insufficient and gradient-based fine-tuning necessary? Needs empirical testing.
- [ ] **Language integration**: Can it be left entirely out of the initial paper, or does it add enough novelty to be worth including?
- [ ] **Data quality of existing 6 materials**: Should some of the "quickly labeled" pairs be relabeled more carefully before fine-tuning begins?

---

## 13. References


| # | Citation | Notes |
|---|----------|-------|
| 1 | [The Matter Lab @ UofT — LinkedIn post](https://www.linkedin.com/posts/the-matter-lab-uoft_ai4science-chemai-autonomouschemistry-activity-7432057975998935040-ZYEI/?utm_source=share&utm_medium=member_desktop&rcm=ACoAAE3Te-gBKfjqV_MbrH-fMFxLbBPncQUznnk) | Aspuru-Guzik group LinkedIn post on autonomous solid-state simulation; discussed at meeting opening; code not open source |
| 2 | [arXiv:2602.17886](https://arxiv.org/pdf/2602.17886) — *El Agente Sólido: A New Age(nt) for Solid State Simulations* | Underlying paper for the LinkedIn post; hierarchical multi-agent LLM framework + Quantum ESPRESSO backend; discussed as representative of the "agentic workflow" trend |
| 3 | [github.com/sterling-group/ChemRefine](https://github.com/sterling-group/ChemRefine) | Automated conformer sampling/refinement for molecular chemistry using ORCA; progressive multi-level-of-theory pipeline; mentioned as parallel example of automated workflows |
| 4 | [arXiv:2207.09444](https://arxiv.org/abs/2207.09444) — *ML approach to genome of 2D materials with flat electronic bands* | Foundational paper in 2DMatPedia flat band series; VAE/autoencoder-based, database-restricted, does not capture near-E_F or isolation |
| 5 | [Nature Commun. Phys. (2025)](https://www.nature.com/articles/s42005-025-01936-2) | Second paper in the flat band genome series from the same group |
| 6 | [arXiv:2506.07518](https://arxiv.org/abs/2506.07518v1) | Third paper: GNN + language model predicts flat band score from crystal structure alone; still database-restricted; not open-source |
| 7 | [arXiv:2505.17859](https://arxiv.org/abs/2505.17859) — *Scalable Valuation of Human Feedback through Provably Robust Model Alignment* (Hölder-DPO) | QMG/Lattice Lab NeurIPS paper; the LLM-version Hölder-DPO loss from which the diffusion adaptation is derived |
| 8 | [Overleaf: Diffusion-DPO Loss Derivation](https://www.overleaf.com/4673732276ftsphvtjnbmm#f2756f) ([local PDF](Diffusion_finetune.pdf)) | MA's draft deriving SCIGEN-aware Diffusion-Hölder-DPO; covers crystal representation, masked denoising error, per-channel improvement score, Hölder robustification, confidence weighting, and outlier contamination estimator |
| 9 | Okabe et al. — *Structural constraint integration in a generative model for the discovery of quantum materials*, **Nature Materials** 25, 223–230 (2026). [doi:10.1038/s41563-025-02355-y](https://doi.org/10.1038/s41563-025-02355-y) | Original SCIGEN paper (the base generative model being fine-tuned) |

---

**Next Meeting:** March 4, 2026 (Weekly Sync)

**Action Items Summary:**
- [ ] **(MA)** Verify and finalize Diffusion-Hölder-DPO loss derivation in Overleaf; produce minimum working code (TRL-compatible custom loss).
- [ ] **(RO)** Choose embedding space(s) and cluster the existing flat band database to map current structural families.
- [ ] **(RO + Higuchi-san)** Operationalize the definition of "novel cluster" for Level 2 contribution.
- [ ] **(RO)** Assess current preference data quality; decide whether to relabel the "quickly-done" pairs.
- [ ] **(All)** Recruit additional expert annotators to expand the preference dataset via `band_eval_web`.
- [ ] **(RO)** Investigate DFT throughput bottleneck — how many candidates can be validated per week?
