# 4D Gaussian Splatting: implementation decision

## What the research methods actually require

The original 3D Gaussian Splatting method initializes explicit 3D Gaussians from camera-calibrated sparse points, optimizes position/color/opacity/anisotropic covariance, and uses a visibility-aware differentiable rasterizer. It is a multi-view reconstruction method, not a single-image depth warp.

4D-GS adds a spatial-temporal encoder and deformation network that maps canonical Gaussians into each time step. Its official implementation targets D-NeRF, HyperNeRF, and multi-view DyNeRF-style datasets, uses custom CUDA rasterization, and reports per-scene training rather than instant processing of arbitrary uploads.

Dynamic 3D Gaussians instead keeps Gaussian appearance and size persistent while optimizing motion and rotation through time with local-rigidity constraints. It also assumes dynamic multi-view supervision.

For casual monocular video, MoSca is the closest practical research pipeline. It first runs foundational depth/tracking models, solves camera/depth alignment, builds 4D motion scaffolds, then fits dynamic Gaussians. Its official setup is Ubuntu/Conda/CUDA 11.8 and includes third-party model/checkpoint licenses. Dynamic Gaussian Marbles independently documents why existing multi-view 4D Gaussian methods fail in the underconstrained monocular setting and adds isotropic primitives, hierarchical optimization, and tracking/geometry priors.

## Decision for Stereo Depth Lab

Directly embedding the official 4D-GS repository would not produce a reliable one-click result from this lab's casual single-video input. It would also introduce an old, tightly pinned CUDA environment and turn a seconds/minutes preview into per-scene optimization.

The implemented **4D Gaussian Lite** renderer adopts the parts that transfer cleanly to the current workflow:

- monocular-depth-seeded spatial means;
- projected Gaussian footprints with horizontal/vertical covariance tuned for stereo parallax;
- opacity reduced near uncertain depth discontinuities;
- near-biased visibility weights when splats overlap;
- optical-flow advection of the previous Gaussian field;
- temporal Gaussian decay controlled by the existing smoothing parameter;
- photometric confidence and scene-cut rejection;
- persistent color, depth, opacity, and time state;
- holes handled by limited inpainting after splat normalization.

For a still image, the same renderer becomes a single-time-slice Gaussian splat. For video, it tracks and reuses the prior time slice, which is where it improves stability over independent per-frame point-cloud rendering.

## Upgrade path to research-grade reconstruction

The next tier should be a separate `mosca` backend adapter rather than a hidden replacement for the interactive renderer:

1. extract frames and preserve timing;
2. estimate depth, long-range tracks, foreground/background masks, and camera motion;
3. fit a MoSca or similarly constrained monocular dynamic Gaussian representation;
4. render the calibrated left/right virtual cameras;
5. return a packaged scene plus SBS video;
6. surface hours/minutes, VRAM, checkpoints, and third-party licenses before launch.

This keeps the interactive default honest while leaving the API boundary ready for a long-running research job.

## Primary sources

- Kerbl et al., [3D Gaussian Splatting for Real-Time Radiance Field Rendering](https://arxiv.org/abs/2308.04079), ACM TOG 2023.
- Wu et al., [4D Gaussian Splatting for Real-Time Dynamic Scene Rendering](https://openaccess.thecvf.com/content/CVPR2024/html/Wu_4D_Gaussian_Splatting_for_Real-Time_Dynamic_Scene_Rendering_CVPR_2024_paper.html), CVPR 2024; [official implementation](https://github.com/hustvl/4DGaussians).
- Luiten et al., [Dynamic 3D Gaussians: Tracking by Persistent Dynamic View Synthesis](https://arxiv.org/abs/2308.09713), 3DV 2024.
- Lei et al., [MoSca: Dynamic Gaussian Fusion from Casual Videos via 4D Motion Scaffolds](https://arxiv.org/abs/2405.17421), CVPR 2025; [official implementation](https://github.com/JiahuiLei/MoSca).
- [Dynamic Gaussian Marbles for Novel View Synthesis of Casual Monocular Videos](https://geometry.stanford.edu/projects/dynamic-gaussian-marbles.github.io/), official project page.
