# WaveNeXt: SAR-to-Optical Satellite Image Translation

WaveNeXt is a deep learning framework for **SAR-to-optical satellite image translation**, designed to generate RGB optical imagery from Synthetic Aperture Radar (SAR) observations.

The project combines **ConvNeXtV2**, **Haar wavelet decomposition**, adversarial learning, perceptual losses, and frequency-domain constraints to improve structural and high-frequency detail in generated optical images.

## Overview

Synthetic Aperture Radar (SAR) imagery can be collected under conditions where optical satellite imagery is unavailable, such as at night or under cloud cover. However, SAR and optical imagery have substantially different visual characteristics.

WaveNeXt learns a mapping from SAR imagery to optical imagery:

```text
                    SAR Input
                        │
                        ▼
             Haar Wavelet Decomposition
                        │
                        ▼
               ConvNeXtV2 Backbone
                        │
                        ▼
                    Decoder
                        │
                        ▼
              Wavelet Reconstruction
                        │
                        ▼
                Generated RGB Image
```

The model is trained using paired SAR-optical satellite imagery.

## Key Features

- ConvNeXtV2-based generator
- Haar wavelet-based decomposition and reconstruction
- High-frequency-aware discriminator
- Adversarial training
- Feature matching loss
- MS-SSIM structural loss
- LAB chroma loss
- Wavelet-detail reconstruction loss
- LPIPS perceptual loss
- Focal Frequency Loss
- PatchNCE contrastive loss
- PyTorch Lightning training pipeline
- Full validation-set evaluation
- PSNR, SSIM, LPIPS and FID evaluation
- Configurable training and inference pipelines

## Architecture

WaveNeXt explicitly incorporates wavelet subbands into the image translation process.

```text
                         SAR Input
                            │
                            ▼
                 ┌─────────────────────┐
                 │   Haar Wavelet       │
                 │   Decomposition     │
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │     ConvNeXtV2      │
                 │      Backbone       │
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │       Decoder       │
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │  Wavelet Subband    │
                 │   Reconstruction    │
                 └──────────┬──────────┘
                            │
                            ▼
                      Optical RGB
                         Output
```

## Loss Functions

WaveNeXt uses a composite objective combining adversarial, structural, perceptual, and frequency-domain constraints.

### Adversarial Loss

Encourages generated images to produce realistic optical imagery.

### Feature Matching

Encourages the generator to reproduce intermediate discriminator features of the target optical image.

### MS-SSIM

Encourages structural similarity between generated and target images.

### LAB Chroma Loss

Encourages accurate colour information in generated optical imagery.

### Wavelet Detail Loss

Constrains high-frequency wavelet components to preserve spatial details.

### LPIPS

Provides perceptual similarity supervision between generated and target images.

### Focal Frequency Loss

Encourages reconstruction of important frequency-domain information.

### PatchNCE

Provides local contrastive supervision between corresponding image features.

## Dataset

The project uses the **QXSLAB_SAROPT** paired SAR-optical dataset.

The training pipeline operates on paired SAR and optical satellite images at a resolution of **256 × 256 pixels**.

Current configured split:

| Split | Samples |
|---|---:|
| Training | 16,000 |
| Validation | 4,000 |
| Total | 20,000 |

## Evaluation

WaveNeXt includes evaluation using:

| Metric | Purpose |
|---|---|
| PSNR | Pixel-level reconstruction quality |
| SSIM | Structural similarity |
| LPIPS | Perceptual similarity |
| FID | Distribution-level image quality |

Full validation evaluation is implemented in:

```text
src/models/wavenext/eval_full.py
```

Top-K per-scene inference is implemented in:

```text
src/models/wavenext/best_inference.py
```

## Project Structure

```text
WaveNeXt-src/
│
├── src/
│   └── models/
│       └── wavenext/
│           ├── gen.py
│           ├── dis.py
│           ├── blocks.py
│           ├── losses.py
│           ├── main.py
│           ├── train.py
│           ├── eval_full.py
│           ├── best_inference.py
│           ├── inference.py
│           ├── factory.py
│           └── config*.yaml
│
├── docs/
│   └── diploma/
│
├── checkpoints/
│
├── README.md
└── .gitignore
```

## Technology Stack

### Deep Learning

- PyTorch
- PyTorch Lightning
- TorchMetrics

### Model Components

- ConvNeXtV2
- Haar Wavelets
- Generative Adversarial Networks
- Perceptual Losses
- Frequency-domain Losses
- Contrastive Learning

### Supporting Tools

- Hugging Face Transformers
- OmegaConf
- Python
- Git

## Training

Training configurations are provided under:

```text
src/models/wavenext/
```

The project uses **PyTorch Lightning** for training, checkpointing, validation, and experiment management.

Large model checkpoints and generated experiment artifacts are excluded from version control.

## Inference

Inference utilities are provided under:

```text
src/models/wavenext/
```

Run the top-K per-scene inference pipeline with:

```bash
python -m src.models.wavenext.best_inference
```

Run full validation evaluation with:

```bash
python -m src.models.wavenext.eval_full
```

## Results

Final benchmark results are currently being validated and will be added after reproducing the evaluation using the released pipeline.

| Metric | Result |
|---|---:|
| PSNR | TBD |
| SSIM | TBD |
| LPIPS | TBD |
| FID | TBD |

## Reproducibility

The repository contains the model architecture, training pipeline, loss functions, evaluation utilities, and configuration files used for the project.

Environment-specific files, secrets, virtual environments, large checkpoints, and generated outputs are excluded from version control.

## Project Status

WaveNeXt is an experimental research implementation for **SAR-to-optical satellite image translation**.

The repository currently focuses on the model architecture, training pipeline, loss design, evaluation framework, and reproducibility of the experiments.