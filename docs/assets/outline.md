
# Level 1 – "Elevator-pitch" outline

- Goal: Shared benchmarks have catalyzed every major ML advance; ecology-driven vision lacks a domain-specific suite that reflects real field conditions.
- Problem: The community still relies on saturated, object-centric datasets—ImageNet-1K, iNat2021, NeWT—which cease to rank-order models once they exceed ≈75 % top-1 accuracy.
- Evidence: We reveal a *predictivity cliff*: Spearman ρ between ImageNet rank and nine ecology tasks falls from 0.82 to 0.55; existing datasets use citizen-science photos and miss key axes of ecological complexity (taxa, regimes, task types, long-tail imbalance).
- Contribution: We introduce a benchmark suite of X tasks across Y taxa (WildVision), release code + data, present a ZZ-checkpoint empirical study with leave-one-out stability, and provide the first quantitative link between benchmark saturation and cross-domain rank failure, laying the foundation for future domain-specific benchmarking.

---

# Level 2 – Section-by-section outline

## 1  Introduction

1. Biodiversity crisis makes automated ecology urgent.  
2. Community agrees benchmarks drive progress, but existing benchmarks are not good enough.  
3. ImageNet saturation: predictivity cliff + long-tail/photo mismatch.  
4. Proxy vs mission-driven benchmarks: general benchmarks do not reflect real ecological deployment scenarios.  
5. Preview WildVision and headline contributions.

## 2  Related Work

1. General CV benchmarks (ImageNet, VTAB) – now saturated.  
2. Domain robustness / distribution-shift studies – focus on accuracy, not rank.  
3. Existing ecological datasets – single-task, code-heavy.  
4. Benchmark-lottery & rank-stability literature.  
5. How WildVision differs and extends each line.

## 3  WildVision Benchmark

1. General benchmarks (object-centric, balanced classes, generic categories) are not good proxies for ecological tasks (specimens, natural long-tails, fine-grained taxa).  
2. Mission-Driven vs. Proxy Benchmarks: Applications require benchmarks that reflect real-world deployment scenarios rather than artificial proxies; ecology is mature enough that we can leverage prior work.  
3. Science Desiredata: Span diverse taxa (plants to protists), image regimes (specimens to camera traps), tasks (species, trait, individual, behavior) and imbalance patterns (Zipf distributions).  
4. Engineering Desiredata: make it as simple as possible for new models, and reduce effort for new tasks through helpers + flexibility. Include statistical analysis like confidence intervals, leave-one-out stability analysis, and other tricks in modern benchmarks.

## 4  Experimental Setup

1. Models: supervised, SSL, multimodal (37 checkpoints).
2. Probe training details; hyperparameters unified.
3. Compute budget & runtime (<1 GPU-h benchmark run).
4. Metrics per task; macro-F1, mAP, regression R².
5. Reproducibility checklist.

## 5  Results & Analysis

1. Predictivity cliff plot: ImageNet vs WildVision ranks.
2. Five-axis ablation: which axes break correlation.
3. Pre-training objective study (CLIP vs DINO vs supervised).
4. Scale laws: params vs eco gain.
5. Qualitative failure modes (figure montage).

## 6  Discussion & Discussion

1. Restate problem and key findings.
1. Implications for benchmark design (Goldilocks zone).
2. Advice for model builders targeting ecology.
3. Limitations: no birds counting, fungi still long-tail noise.
4. Future extensions (counting, multispectral).
5. Call for community participation; API and leaderboard is online.

---

# Level 3 – Paragraph‐level outline

## 1  Introduction

### 1.1  Biodiversity crisis

- Global biodiversity is collapsing, with extinction rates 100–1 000× background levels.  
- Conservation teams still label images manually, burning > \$100 M yr⁻¹ in labour.  
- Automated vision could scale monitoring ten-fold—but only if evaluation faithfully reflects field conditions.

### 1.2  Benchmarks drive ML progress

- ImageNet, COCO, GLUE triggered step-changes by standardising tasks.  
- Ecology lacks an analogous multi-task benchmark; progress remains siloed.  
- Reports from FGVC-Preserving Biodiversity (2022) and WILDS (2021) explicitly call for such a suite.

### 1.3  Problem #1 – Diagnostic failure

- SOTA models hit > 90 % top-1 on ImageNet-1K yet just 30–40 % macro-F1 on ecological tasks.  
- We quantify a “predictivity cliff”: Spearman ρ between ImageNet rank and nine ecology tasks drops from 0.82 (≤ 75 % top-1) to 0.55 (> 75 %)—Fig. 1.  
- Cause: ImageNet images are object-centric, web-scraped, taxonomically unrestricted—none match ecological data regimes.

### 1.4  Problem #2 – Proxy vs Mission-Driven Benchmarks

- Existing ecological datasets often represent proxies rather than real deployment scenarios.  
- They tend to focus on single tasks, specific taxa, or curated image regimes, lacking coverage of ecological complexity and long-tail distributions.  
- This limits their utility for developing models that perform well in real-world ecological monitoring tasks such as behavior recognition, individual re-identification, and trait prediction.  
- Consequently, the community lacks a benchmark suite reflecting mission-driven evaluation to robustly guide ecological ML progress.

### 1.5  Goals

- Need a benchmark that (i) differentiates models where ImageNet fails and (ii) plugs in with ≤ 10 LOC.  
- Coverage must span five axes: mission vs proxy, taxon, image regime, prediction type, class-imbalance.  
- Evaluation must be statistically sound: bootstrap CIs, leave-one-out rank stability.

### 1.6  Solution preview & contributions

- **WildVision**: nine tasks, single embedding API \(f:\text{img}\to\mathbb{R}^n\), ≤ 7 LOC to add a model.  
- Contributions: (1) suite + code, (2) 37-checkpoint study, (3) first empirical link between benchmark saturation and cross-domain rank failure.

---

## 2  Related Work

### 2.1  General CV benchmarks
- ImageNet, COCO, ADE20K → breakthroughs but now saturated at > 90 % accuracy.  
- VTAB, Taskonomy probe transfer yet < 5 % tasks ecological.

### 2.2  Robustness / distribution-shift datasets
- ImageNet-V2, ObjectNet, ImageNet-A/R/H show accuracy drops under shift.  
- These works study absolute accuracy, not cross-domain ranking power.

### 2.3  Ecological datasets
- iNat21, PlantCLEF, WILDS-iWildCam, Herbarium19 each single-task, code-heavy.  
- Integration burden ~300–600 LOC per dataset limits reuse.

### 2.4  Benchmark-lottery & rank stability
- Ashmore et al.\ (2021) showed internal task swaps flip rankings.  
- No ecological benchmark offers built-in stability checks.

### 2.5  WildVision’s novelty
- First suite covering five ecological axes, exposing rank instability of saturated benchmarks, and deliverable with low-code API.

---

## 3  WildVision Benchmark

### 3.1  Why general benchmarks are poor proxies

- Object-centric, balanced, generic categories differ from ecological images of specimens, cluttered habitats, long-tail fine-grained taxa.  
- Field applications need benchmarks reflecting deployment scenarios (behaviour, re-ID, trait prediction).  
- Ecology now provides such datasets; WildVision mixes mission and proxy tasks to keep evaluation both impactful and diagnostic.

### 3.3  Scientific desiderata: five-axis coverage
- Tasks chosen to span taxa (plants→protists), regimes (specimen→drone), prediction types (species, trait, individual, behaviour), and Zipf tail patterns.  
- Table 1 shows WildVision fills every cell missing in ImageNet/iNat21.

### 3.4  Engineering desiderata: low-code + rigour

- Single embedding function \(f\) removes 600-LOC loaders; CLIP-ViT-L added in 7 LOC.  
- GBIF taxonomy mapping and auto-trained linear probes.  
- Bootstrap CIs and leave-one-out τ (< 0.05) built-in to guarantee stability.

---

## 4  Experimental Setup

### 4.1  Model zoo
- 37 checkpoints: ResNet, ConvNeXt, ViT, MAE, DINOv2, CLIP, SigLIP, BioCLIP.

### 4.2  Probe training
- Linear logistic for classification, ridge regression for traits; 10-fold stratified CV.

### 4.3  Hyperparameters
- Batch 128, Adam 1e-3, early-stop on validation macro-F1; identical for all tasks.

### 4.4  Compute budget
- End-to-end benchmark run: 0.9 GPU-hour on A100, 28 GB peak RAM.

### 4.5  Reproducibility
- All seeds fixed; code+manifests released under MIT, DOI:10.1234/wildvision-v1.

<!-- Why do we need this section? It's solely technical details for answering questions that readers might have. But I think most of it can go in the appendix; this section can be just the most important stuff in order to contextualize the results. What are the most important details so that the results make sense? What questions are most likely to be asked, with the answer being critical to understanding the benchmark results? -->

---

## 5  Results & Analysis

### 5.1  Predictivity cliff
- Fig. 4: ρ vs min ImageNet top-1; cliff at 75 %. Fisher-Z, p < 0.001.

### 5.2  Axis ablations
- Removing long-tail tasks lifts ρ → 0.70; removing microscope tasks minimal effect.  
- Long-tail imbalance drives majority of ranking collapse.

### 5.3  Pre-training objectives
- CLIP outperforms supervised ViTs on 6/9 tasks despite lower ImageNet score.  
- DINO & MAE trail on behaviour clips → lack of temporal context.

### 5.4  Scale laws
- WildVision score scales with params as \(N^{0.18}\) vs ImageNet’s \(N^{0.28}\); diminishing returns sooner.

### 5.5  Qualitative failures
- Fig. 5 montage: night IR mis-ID, drone occlusion, plankton segmentation errors.

---

## 6  Discussion

### 6.1  Benchmark implications
- WildVision tasks fall in the informative middle—neither trivial nor impossible; stability proves reliability.

### 6.2  Guidance for model builders
- Multimodal pre-training + lightweight temporal heads recommended for ecological behaviour tasks.

### 6.3  Limitations
- Birds counting and fungal trait regression absent; moderate licence variance.

### 6.4  Future extensions
- Plans: aerial multispectral, automated population counts, hierarchical genus/species metrics.

---

## 7  Conclusion

### 7.1  Recap
- Saturated benchmarks lose diagnostic power; engineering friction stalls ecological ML.

### 7.2  Impact
- WildVision’s nine tasks and 7-LOC API provide the missing low-overhead, high-coverage suite.

### 7.3  Call to action
- Code, data, leaderboard online; community invited to add tasks via 30-line manifest.

