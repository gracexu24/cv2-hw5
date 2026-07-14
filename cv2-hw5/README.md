# CV2 HW5 Part A: The Power of Diffusion Models

## Dependencies
- Python 3.10+ (Google Colab recommended)
- torch
- torchvision
- diffusers
- transformers
- huggingface_hub
- numpy
- matplotlib
- Pillow

### Setup Instructions

This project is designed to run on **Google Colab** (GPU required).

1. Open the starter notebook in Colab.
2. Log in to Hugging Face and accept the [DeepFloyd IF license](https://huggingface.co/DeepFloyd/IF-I-XL-v1.0).
3. Set your Hugging Face access token in the notebook:
```python
from huggingface_hub import login
login(token="your_token_here")
```
4. Generate prompt embeddings using the provided HuggingFace Space (Cluster A or B), download the `.pth` file, and load it in the notebook.

## To Run

All work is completed inside the Colab notebook. Run cells in order:

1. **Part 0** — Load the DeepFloyd model, generate prompt embeddings, and sample images for 3 custom text prompts at 2 different `num_inference_steps` values.
2. **Part 1.2–1.4** — Forward process, classical denoising, one-step denoising, and iterative denoising (implementations provided).
3. **Part 1.5** — Sample 5 images from pure noise using `"a high quality photo"`.
4. **Part 1.6** — Implement `iterative_denoise_cfg` and sample 5 images using Classifier-Free Guidance (CFG scale γ = 7).
5. **Part 1.7** — SDEdit on the Campanile and 2 custom images at i_start ∈ {1, 3, 5, 7, 10, 20}.
6. **Part 1.7.1** — Apply SDEdit to 1 web image and 2 hand-drawn images.
7. **Part 1.7.2** — Implement inpainting via the RePaint method.
8. **Part 1.7.3** — Text-conditional image-to-image translation with custom prompts.
9. **Part 1.8** — Implement visual anagrams (images that flip to reveal a different scene).
10. **Part 1.9** — Implement hybrid images using factorized diffusion (Gaussian blur kernel size 33, σ = 2).

## Saving Results

After generating each result, save images to `parta/savedimages/`. The notebook includes save cells for each section. File naming convention used:

| File | Section |
|---|---|
| `a fluffy fat cat eating pizza(20).png` | Part 0 — Prompt 1, 20 steps |
| `a fluffy fat cat eating pizza (80).png` | Part 0 — Prompt 1, 80 steps |
| `a monkey with cookies(20).png` | Part 0 — Prompt 2, 20 steps |
| `a monkey with cookies (80).png` | Part 0 — Prompt 2, 80 steps |
| `sleeping bunny on hamburger(20).png` | Part 0 — Prompt 3, 20 steps |
| `sleeping bunny on hamburger.png` | Part 0 — Prompt 3, 80 steps |
| `1.2.jpg` | Part 1.2 — Classical Denoising |
| `1.3.jpg` | Part 1.3 — One-Step Denoising |
| `1.4.jpg` | Part 1.4 — Iterative Denoising |
| `1.5.jpg` | Part 1.5 — Diffusion Sampling (5 samples) |
| `1.6.jpg` | Part 1.6 — CFG Sampling (5 samples) |
| `1.7.jpg` | Part 1.7 — SDEdit results |
| `1.7.1web.jpg` | Part 1.7.1 — Web image SDEdit |

## View Results

Open `index.html` in a web browser to view all deliverables organized by section.

## Notes
- Random seed used throughout: `42`
- All parts from 1.5 onward use CFG (γ = 7) unless otherwise noted
- The DeepFloyd stage I model outputs 64×64 images; upsampling code is provided in the notebook but not required for submission

## LLM Usage
Cursor was used to help generate the `index.html` results webpage.
