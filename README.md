# MAT-Mamba: Overcoming Quadratic Attention Bottlenecks in 3D Medical Segmentation

**Authors:** Iha Goyal (b23bb1020) & Om Kumar (b23bb1030)  
**Institution:** Indian Institute of Technology Jodhpur 
### Live Training Dashboard
The complete training logs, loss convergence curves, and GPU memory profiling can be viewed interactively on our Weights & Biases dashboard:
**[View MAT-Mamba WandB Workspace Here](https://wandb.ai/ihagoyal-mun-indian-instit/MAT3D-ACDC-Reproduction/workspace?nw=nwuserihagoyalmun)**

## Project Overview
Accurate 3D segmentation of complex biological structures (like cardiovascular chambers) is critical for clinical diagnostics. Recent frameworks like Multi-Aperture Transformers for 3D (MAT3D) achieve high precision but rely on standard Vision Transformers (SwinUNETR). These localized attention mechanisms scale quadratically — **O(N²)** — creating massive memory bottlenecks for thick volumetric data.

In this project, we reproduced the MAT3D baseline and proposed a novel architectural surgery: the **MAT-Mamba Hybrid**. By substituting the deepest Swin Transformer block with a Linear-Time State-Space Model (Mamba) utilizing a residual bridge, we dynamically captured continuous global spatial context at **O(N)** efficiency.

## Architecture: The MAT-Mamba Hybrid
* **Baseline:** SwinUNETR (MAT3D)
* **Novel Contribution:** Targeted replacement of the deepest, most memory-intensive stage of the encoder (`encoder10` / Swin T4) with a custom `MambaBottleneck`.
* **3D Adaptation:** Engineered a `Mamba3DLayer` to optimally flatten 3D spatial dimensions `(B, C, H, W, D)` into sequences `(B, Sequence, C)`, process them through Mamba's selective scan, and reshape them back to 3D.

## Dataset
**ACDC (Automated Cardiac Diagnosis Challenge)**
* 300 3D Cardiac MRI volumes.
* Custom MLOps pipeline built to extract raw HDF5 files and convert them into standard NIfTI format.
* Volumes dynamically padded and center-cropped to a strict `96 x 96 x 96` patch size.

## Key Results
Trained for 100 epochs using mixed-precision on an Nvidia T4 GPU (Google Colab).

| Structure | Baseline (SwinUNETR) | Hybrid (MAT-Mamba) | Improvement |
| :--- | :---: | :---: | :---: |
| **Right Ventricle (RV)** | 67.89% | **69.15%** | **+1.26%** |
| **Myocardium (Myo)** | 62.62% | **62.70%** | +0.08% |
| **Left Ventricle (LV)** | 75.58% | **76.38%** | +0.80% |

**Computational Efficiency:**
The MAT-Mamba Hybrid successfully integrated global context modeling with only a **~0.5GB overhead** in peak GPU memory allocation (4.6GB vs 4.1GB) and completed training in the exact same duration as the baseline (~150 minutes). 

## Repository Structure
* `Phase2_Baseline_SwinUNETR.ipynb`: Data engineering pipeline, baseline model initialization, training logic, and baseline evaluation.
* `Phase2_Hybrid_MAT_Mamba.ipynb`: Custom Mamba installation (via pre-compiled wheels), `Mamba3DLayer` and `MambaBottleneck` class implementations, hybrid training logic, and WandB profiling.
* `Phase2_Prototype_Tensor_Verification.ipynb`: Architectural unit-testing and tensor dimensionality verification for the 768-dimensional Mamba bottleneck injection.
