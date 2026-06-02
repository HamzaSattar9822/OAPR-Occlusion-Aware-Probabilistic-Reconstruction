# OAPR: Occlusion-Aware Probabilistic Pose Reconstruction

**Journal Article Draft**  
**Project:** Multi-Human Pose Estimation Under Occlusion and Crowding  
**Date:** May 2026

---

## Motivation

Accurate multi-person pose estimation is essential for sports analytics, surveillance, rehabilitation, and human–robot interaction. Although modern top-down methods achieve strong performance on standard benchmarks when subjects are fully visible, their accuracy drops sharply in crowded environments where people overlap, limbs are self-occluded, or body parts lie outside the field of view. In such settings, keypoint detectors often produce missing or misplaced joints, and standard regression losses treat all errors equally, making the model sensitive to outliers and ambiguous annotations.

Temporal information from video can improve stability and disambiguate overlapping persons, but many pose systems still treat frames independently or rely on recurrent architectures with limited long-range modeling. Occlusion is frequently handled implicitly through data augmentation rather than through an explicit mechanism that identifies unreliable joints and reconstructs them using spatial and temporal context.

This work is motivated by three gaps in the literature: (1) efficient spatiotemporal modeling that captures long-range dependencies without prohibitive cost, (2) explicit occlusion-aware reconstruction in end-to-end pose frameworks, and (3) robust probabilistic losses that down-weight noisy labels and uncertain predictions. We address these through **OAPR (Occlusion-Aware Probabilistic Pose Reconstruction)**, a unified framework combining a hybrid state-space and attention backbone, an occlusion reconstruction module, and heavy-tailed probabilistic learning.

---

## Introduction

Human pose estimation aims to localize anatomical keypoints of each person in an image or video. The dominant top-down paradigm first detects individuals and then regresses keypoints for each crop. High-resolution networks such as HRNet preserve spatial detail and set strong baselines via heatmap regression. Benchmark performance on COCO does not fully reflect behavior in crowded scenes, where dedicated datasets show larger performance gaps.

We develop OAPR as a complete pipeline for pose estimation under occlusion and crowding. The work proceeds in sequence: we first establish a reproducible **HRNet-W32 baseline** with COCO and CrowdPose dataloaders, training, and evaluation; we then extend the system with **spatiotemporal modeling** over seven-frame video clips using a hybrid Mamba–Transformer backbone; finally, we integrate **occlusion-aware reconstruction** and **Cauchy-mixture robust losses** into a single trainable framework.

The system outputs keypoint coordinates, confidences, occlusion indicators, and uncertainty estimates. The implementation supports COCO metrics (AP, AP50, AP75), flip test augmentation, DARK post-processing, TensorBoard logging, and diagnostic tools including per-keypoint confusion analysis and heatmap accuracy maps.

This document describes architecture, experiments, and measured findings. Subset validation diagnostics on COCO val2017 at inference batch sizes **16** and **32** are reported below, along with partial baseline training on a reduced train split. Full benchmark AP and CrowdPose evaluation will follow completion of GPU training.

---

## Related Work

**High-resolution pose estimation.** HRNet maintains high-resolution feature maps through parallel multi-scale branches, achieving state-of-the-art COCO keypoint results. It serves as the baseline in this work via heatmap regression with Gaussian targets.

**Spatiotemporal pose estimation.** Video-based methods exploit temporal consistency to reduce jitter. State-space models (Mamba) offer linear-time sequence modeling; OAPR uses a hybrid Mamba–Transformer design with automatic fallback to a temporal Transformer when Mamba is unavailable.

**Occlusion and crowded scenes.** CrowdPose highlights pose difficulty in crowded images. OAPR introduces explicit occlusion detection, spatial graph reasoning, temporal encoding, and reconstruction of occluded joints.

**Robust regression.** Heavy-tailed losses (Cauchy, Laplace, mixtures) improve resilience to outliers compared with pure MSE. OAPR combines these with per-joint uncertainty and occlusion weighting.

**Evaluation.** COCO keypoint AP is standard; complementary tools include PCK, per-joint breakdowns, confusion-style outcome analysis, and heatmap comparisons.

---

## Methodology

### Overview

OAPR builds from a single-frame baseline toward a video-aware, occlusion-robust system. Input frames or clips pass through a spatiotemporal backbone, an occlusion module, and a robust probabilistic loss.

### Baseline: HRNet-W32

Person crops are resized to **192×256** (W×H). HRNet-W32 predicts **17** COCO heatmaps at **64×48**. Targets use Gaussian heatmaps (σ = 2) with visibility-weighted MSE loss. Inference uses argmax decoding with optional DARK refinement and flip-test averaging.

### Spatiotemporal backbone

Video clips of **T = 7** frames are processed by:

- **Temporal branch:** Mamba SSM layers (Transformer fallback if needed)
- **Spatial branch:** Multi-head self-attention (8 heads, hidden size 256)
- **Instance tokens** for multi-person disambiguation

### Occlusion-aware reconstruction

1. **Occlusion detection** — confidence below threshold (0.5) flags occluded joints  
2. **Spatial context** — skeleton graph reasoning  
3. **Temporal context** — LSTM over joint trajectories  
4. **Reconstruction** — fused prediction of occluded joint coordinates  

### Robust probabilistic loss

Cauchy, Laplace, or Cauchy-mixture losses with scale regularization (0.1) and occlusion-aware weighting (0.15).

### Data and training

- **Dataset:** COCO 2017 person keypoints (17 joints); CrowdPose supported (14 joints)  
- **Augmentation:** flip, rotation ±40°, scale [0.65, 1.35], color jitter; random patch erasing (p = 0.2) in the full framework  
- **Optimizer:** Adam (baseline lr = 1×10⁻³; full framework lr = 5×10⁻⁴)  
- **Metrics:** COCO AP (primary); heatmap PCK@0.5 for monitoring and diagnostics  

---

## Experiments

### Setup

| Item | Detail |
|------|--------|
| Platform | Apple M1, CPU (CUDA unavailable for reported runs) |
| Framework | PyTorch, timm (HRNet-W32), pycocotools |
| Val set | 6,352 person instances (~2,346 unique images) |
| Diagnostic model | HRNet-W32, ImageNet-pretrained (**pose head not finetuned** for diagnostic runs) |
| Diagnostic protocol | First 15 val batches per inference batch size |
| Training run | HRNet, 10-epoch target, 5% COCO train subset, batch 16, CPU |

### Inference batch sizes reported

Only **batch size 16** and **batch size 32** are evaluated in the results section. Sample counts equal **15 × batch size** (240 and 480 val person instances respectively).

---

## Results

### Table 1 — Subset validation summary (batch 16 vs batch 32)

| Inference batch | Val samples | % of val set (6,352) | Mean PCK@0.5 | MSE loss |
|-----------------|-------------|----------------------|--------------|----------|
| **16** | 240 | 3.78% | **0.527** | 0.00362 |
| **32** | 480 | 7.56% | **0.541** | 0.00357 |

*Term: Multi-batch inference evaluation — keypoint confusion and heatmap diagnostics (COCO val subset).*

These figures reflect **evaluation pipeline output** on real COCO validation crops. They do **not** represent full-dataset COCO AP or training ablations at batch 16 vs 32.

---

### Batch size 16 (n = 240)

#### Table 2 — Per-keypoint PCK@0.5 (batch 16)

| Joint | PCK@0.5 |
|-------|---------|
| Nose | 0.448 |
| Left eye | 0.500 |
| Right eye | 0.517 |
| Left ear | 0.563 |
| Right ear | 0.359 |
| Left shoulder | 0.493 |
| Right shoulder | 0.547 |
| Left elbow | 0.667 |
| Right elbow | 0.565 |
| Left wrist | 0.577 |
| Right wrist | 0.507 |
| Left hip | 0.656 |
| Right hip | 0.500 |
| Left knee | 0.469 |
| Right knee | 0.642 |
| Left ankle | 0.404 |
| Right ankle | 0.539 |
| **Mean** | **0.527** |

#### Figures — Batch 16

**Keypoint detection outcomes (confusion matrix)**

![Batch 16 — Keypoint outcome confusion matrix](outputs/confusion_heatmaps/batch_16/confusion_matrix_keypoint_outcomes.png)

**Joint-index confusion (GT joint → nearest predicted channel)**

![Batch 16 — Joint-index confusion matrix](outputs/confusion_heatmaps/batch_16/confusion_matrix_joint_swap.png)

**Per-keypoint heatmap accuracy (PCK@0.5)**

![Batch 16 — Per-joint accuracy heatmap](outputs/confusion_heatmaps/batch_16/heatmap_per_joint_accuracy.png)

**Predicted vs ground-truth heatmaps (sample batch item)**

![Batch 16 — Predicted vs GT heatmaps](outputs/confusion_heatmaps/batch_16/heatmap_pred_vs_gt_batch0.png)

---

### Batch size 32 (n = 480)

#### Table 3 — Per-keypoint PCK@0.5 (batch 32)

| Joint | PCK@0.5 |
|-------|---------|
| Nose | 0.605 |
| Left eye | 0.351 |
| Right eye | 0.538 |
| Left ear | 0.507 |
| Right ear | 0.498 |
| Left shoulder | 0.576 |
| Right shoulder | 0.499 |
| Left elbow | 0.650 |
| Right elbow | 0.563 |
| Left wrist | 0.631 |
| Right wrist | 0.565 |
| Left hip | 0.531 |
| Right hip | 0.538 |
| Left knee | 0.470 |
| Right knee | 0.529 |
| Left ankle | 0.575 |
| Right ankle | 0.571 |
| **Mean** | **0.541** |

#### Figures — Batch 32

**Keypoint detection outcomes (confusion matrix)**

![Batch 32 — Keypoint outcome confusion matrix](outputs/confusion_heatmaps/batch_32/confusion_matrix_keypoint_outcomes.png)

**Joint-index confusion (GT joint → nearest predicted channel)**

![Batch 32 — Joint-index confusion matrix](outputs/confusion_heatmaps/batch_32/confusion_matrix_joint_swap.png)

**Per-keypoint heatmap accuracy (PCK@0.5)**

![Batch 32 — Per-joint accuracy heatmap](outputs/confusion_heatmaps/batch_32/heatmap_per_joint_accuracy.png)

**Predicted vs ground-truth heatmaps (sample batch item)**

![Batch 32 — Predicted vs GT heatmaps](outputs/confusion_heatmaps/batch_32/heatmap_pred_vs_gt_batch0.png)

---

### Partial baseline training (supplementary)

A separate HRNet training run targeted 10 epochs on a 5% COCO train subset (3,908 instances, batch 16, CPU). Heatmap accuracy improved during training (e.g., from ~0.41–0.57 in early epoch 1 to **0.76** by mid-epoch 3). **No checkpoint or COCO AP** was produced before the run stopped. This indicates learning on the reduced set but is **not** reported as final model performance.

| Epoch | Batch progress | Loss | Heatmap accuracy |
|-------|----------------|------|------------------|
| 1 | 0 / 244 | 0.0027 | 0.5715 |
| 1 | 200 / 244 | 0.0102 | 0.4072 |
| 2 | 200 / 244 | 0.0023 | 0.6774 |
| 3 | 100 / 244 | 0.0020 | 0.7610 |

---

### Target benchmark performance (not yet measured)

After full GPU training, the HRNet baseline is expected to approach:

| Dataset | AP | AP50 | AP75 |
|---------|-----|------|------|
| COCO val2017 | ~74.4 | ~90.5 | ~81.9 |
| CrowdPose test | ~67.0 | ~85.7 | ~72.2 |

---

## Future Work

1. Complete HRNet and OAPR training on the full COCO train split using GPU hardware.  
2. Report standard COCO AP and CrowdPose test metrics from `evaluate.py` with saved checkpoints.  
3. Regenerate batch 16/32 diagnostics using a **finetuned checkpoint** on the **full** validation set.  
4. Run ablations: occlusion module disabled, MSE vs Cauchy mixture, Mamba vs Transformer temporal branch.  
5. Produce qualitative skeleton overlays on validation images for the paper.  
6. Finish local COCO download to remove missing-image filtering bias.  

---

## Conclusion

We presented **OAPR**, an occlusion-aware probabilistic framework for multi-human pose estimation that unifies HRNet baseline heatmap regression, hybrid spatiotemporal modeling, explicit occlusion reconstruction, and Cauchy-mixture robust learning. The codebase provides training, evaluation, and diagnostic tooling for COCO and CrowdPose.

Reported **batch 16** and **batch 32** subset validation results (mean PCK **0.527** and **0.541**, respectively) demonstrate the evaluation pipeline on real COCO validation data with confusion matrices and heatmap visualizations. These are preliminary diagnostics under an ImageNet-initialized pose head, not full benchmark AP. Partial baseline training shows improving heatmap accuracy on a reduced train split. Full experimental validation—COCO AP, CrowdPose results, and OAPR-vs-baseline comparison—remains the next step once GPU training completes.

The architecture targets failure modes in crowded and occluded scenes where per-frame regression alone is insufficient. With completed training, OAPR is positioned to deliver quantitative benchmark gains and interpretable uncertainty outputs for downstream applications.

---

## Appendix — Artifact paths

| Content | Path |
|---------|------|
| Batch 16 metrics JSON | `outputs/confusion_heatmaps/batch_16/summary.json` |
| Batch 32 metrics JSON | `outputs/confusion_heatmaps/batch_32/summary.json` |
| Training log (partial) | `training_10epochs.log` |
| Baseline config | `configs/baseline_hrnet.yaml` |
| Full OAPR config | `configs/m3_oapr_complete.yaml` |
| Diagnostic script | `generate_confusion_heatmap.py` |

---

*End of document*
