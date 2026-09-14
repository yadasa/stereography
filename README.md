# Stereo Depth Lab

A private, local-GPU stereoscopic image and video workflow. The hosted control surface runs in the browser, while the uploaded media and model inference stay on `127.0.0.1`.

## Run everything on localhost

Clone the repository, then use the one-command launcher:

```powershell
git clone https://github.com/yadasa/stereography.git
cd stereography
Set-ExecutionPolicy -Scope Process Bypass
.\start-local.ps1
```

This starts the GPU worker in a second PowerShell window and serves the interface at [http://127.0.0.1:4173](http://127.0.0.1:4173). Nothing is uploaded to OpenAI or another host in this mode.

On Linux or macOS, run `./start-local.sh`. To run only the local interface without starting the model worker, use `npm run dev`.

## Use the private hosted control surface

1. Download the **complete local app** from the private Site and unzip it. On Windows, open PowerShell in that folder and run:

   ```powershell
   Set-ExecutionPolicy -Scope Process Bypass
   .\start-worker.ps1
   ```

   On the first run, add `-Reinstall` if the environment already exists but dependencies have not been installed. The first use of each model also downloads its weights to the normal Hugging Face cache.

2. Leave the worker terminal open and open the private deployed Site.
3. Upload a clip up to 90 seconds or a JPG, PNG, or WebP still image. Choose a model and rendering method, then generate the stereo result.
4. Preview or download the side-by-side MP4/PNG. Use **Compare depth models** on a representative frame before committing to a full render.

The worker binds only to `127.0.0.1`, accepts browser requests only from the private Site and local preview origins, limits processing to one render at a time, and removes old job files after two hours when a new job starts.

## Models

| Choice | Implementation | Practical use |
| --- | --- | --- |
| Depth Anything V2 Small | `depth-anything/Depth-Anything-V2-Small-hf` through Transformers | Default; detailed edges and a good quality/speed balance |
| MiDaS Hybrid | `Intel/dpt-hybrid-midas` through Transformers | Independent baseline for broad scene structure and failure comparison |

Both estimate relative monocular depth. They do not recover metric camera distance, and reflective, transparent, very thin, or fast-moving objects can still produce stereo artifacts.

## View synthesis methods

| Method | How it works | Performance guidance |
| --- | --- | --- |
| Depth warp | Inverse-remaps each eye from the source using a disparity field | Start here. Roughly baseline memory, fast OpenCV remap, few holes, best for previews and moderate separation. It can stretch boundaries where hidden background should be revealed. |
| Point cloud | Treats each pixel as a 2.5D sample, forward-splats it with a depth-aware visibility order, then inpaints holes | Use for stronger parallax or foreground crossings. Expect about 2–5× the synthesis cost and significantly more transient RAM. Calibrate at 720p before a final 1080p render. Large disocclusions still require a generative inpainting stage for production quality. |
| 4D Gaussian Lite | Seeds screen-space Gaussians from monocular depth, advects persistent Gaussians with optical flow, applies temporal confidence/decay, and rasterizes anisotropic eye views with depth-weighted visibility | Best local option for moving subjects and temporally stable depth. Expect roughly 2–4× the synthesis cost plus optical-flow memory. Calibrate at 720p. Hard cuts reset history automatically. |

Depth inference is usually the main GPU cost. Point-cloud and Gaussian synthesis in this version are CPU/NumPy/OpenCV-based so they remain easy to inspect and can later be replaced with CUDA rasterizers.

**4D Gaussian Lite is not a trained, calibrated multi-view 4DGS reconstruction.** It is the practical monocular preview path: each frame supplies depth-seeded spatial Gaussians, and prior Gaussians persist through the time dimension using optical flow and confidence. Full MoSca/4DGS integration requires a dedicated research environment, scene optimization, camera/depth preprocessing, and materially longer turnaround. See [`docs/4d-gaussian-research.md`](docs/4d-gaussian-research.md).

## Controls

- **Eye separation** sets maximum disparity in output pixels.
- **Depth strength** scales the inferred depth range without changing the model.
- **Convergence** selects the zero-parallax depth plane. Values near 50% are a comfortable starting point.
- **Temporal smoothing** blends each depth map with the prior frame. Increase it for static shots; reduce it for fast cuts or motion.
- In 4D Gaussian Lite mode, temporal smoothing also controls the temporal covariance and persistence of advected Gaussians.
- **Gaussian radius** controls the anisotropic screen-space footprint. Start around 1.35 px; increase it to fill small holes, or decrease it to preserve fine edges.
- Temporal smoothing is automatically disabled for still images because there is no preceding frame.
- **Resolution** is per eye. A 720p SBS frame is 2560×720; a 1080p SBS frame is 3840×1080.

## GPU notes

The worker uses CUDA automatically when PyTorch detects it and falls back to CPU for compatibility. If the status pill says CPU on an NVIDIA machine, install the CUDA-enabled PyTorch build recommended for your driver, then restart the worker. `ffmpeg` should be available on `PATH` for H.264 output and original-audio muxing; otherwise the worker keeps the OpenCV MP4 fallback.

## Development and validation

Build the hosted static control surface:

```bash
npm run check
npm run build
```

Run the deterministic pipeline test without downloading model weights:

```bash
STEREO_LAB_FAKE_MODEL=1 python -m unittest worker.tests.test_pipeline
```

The fake mode exists only for local diagnostics. Normal launches use the selected real model.

## Migration boundary

The browser talks to a small `/api` contract: health, frame comparison, job creation/status, and output download. A later cloud GPU or WebGPU implementation can replace the worker without redesigning the interface. For cloud migration, add authenticated object storage with short-lived upload URLs, an authenticated job queue, encrypted expiry, and per-user authorization before enabling any remote upload.
