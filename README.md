# Generative AI Experiments: Image Captioning with CLIP and GPT-2

This project implements an end-to-end **image captioning pipeline** that combines **CLIP** for visual understanding and **GPT-2** for language generation. The core idea is to extract image embeddings using a pretrained CLIP vision encoder and inject them into GPT-2 through a learned prefix projection so that the language model can generate captions conditioned on the image.

The notebook also explores multiple adaptation strategies beyond the basic prefix-projection setup, including **prefix tuning**, **LoRA**, and **partial fine-tuning**, along with decoding and sensitivity analysis.

## Project Overview

The system is designed to study how pretrained vision and language models can be connected for multimodal generation without training a full captioning model from scratch.

### Main pipeline
- **Image encoder:** CLIP ViT-B/32
- **Language decoder:** GPT-2
- **Bridge between modalities:** learned prefix projection network
- **Dataset:** COCO 2017 captions
- **Evaluation:** BLEU, METEOR, ROUGE-L, CIDEr
- **Decoding strategies:** greedy decoding, beam search, nucleus sampling

## Objectives

- Build an image captioning model using pretrained components
- Learn how visual embeddings can be injected into a language model
- Compare multiple parameter-efficient adaptation methods
- Analyze caption quality with both qualitative and quantitative evaluation
- Study the effect of decoding choices and generation parameters

## Methods Implemented

### 1. Prefix Projection Captioning
The main model uses:
- CLIP to extract a normalized **512-dimensional image embedding**
- A trainable MLP to project that embedding into **10 GPT-2 prefix tokens**
- GPT-2 to generate captions using the projected visual prefix and a text prompt

This setup keeps **CLIP and GPT-2 frozen** and trains only the projection layer, making the approach lightweight and efficient.

### 2. Cross-Attention Injection
An alternative cross-attention-based embedding injection module is also implemented for comparison, where GPT-2 hidden states attend over visual features.

### 3. Prefix Tuning
A soft prompt tuning approach is explored by learning trainable prefix embeddings while keeping GPT-2 frozen.

### 4. LoRA
Low-Rank Adaptation is applied to GPT-2 for parameter-efficient fine-tuning, combined with the visual prefix projection.

### 5. Partial Fine-Tuning
A more flexible setup is tested by unfreezing selected GPT-2 layers and the language modeling head, while still using the visual prefix projection.

## Dataset

The project uses the **COCO 2017 captions dataset**.

The notebook:
- downloads the COCO annotation files
- loads image-caption pairs
- selects sample images for qualitative evaluation
- downloads a larger subset of training images for learning the projection layer

Images are resized to **224 × 224** before being passed to CLIP.

## Training Setup

The core training pipeline:
- uses **AdamW**
- trains for multiple epochs
- samples a subset of image-caption pairs per epoch
- optimizes caption generation loss through GPT-2 conditioned on projected CLIP features

The notebook is written to run primarily in **Google Colab**, with intermediate checkpoints optionally saved to **Google Drive**.

## Evaluation

The project evaluates the generated captions using both automatic metrics and visual inspection.

### Automatic metrics
- **BLEU**
- **METEOR**
- **ROUGE-L**
- **CIDEr**

### Qualitative analysis
The notebook compares:
- ground-truth captions
- baseline captions before training
- captions after training
- outputs across different decoding strategies

## Decoding Strategies Compared

The notebook compares three decoding methods:

- **Greedy decoding**  
  Fast and deterministic, but may produce repetitive or less descriptive outputs.

- **Beam search**  
  Produces more fluent and stable captions and generally offers the best balance between relevance and readability.

- **Nucleus sampling**  
  Generates more diverse captions, but can sometimes reduce factual alignment with the image.

## Sensitivity Analysis

The project includes additional experiments to study:

- **temperature sensitivity**
- **beam width sensitivity**
- **embedding injection choices**
- **fine-tuning strategy comparison**

These analyses help understand how generation settings affect caption quality and stability.

## Key Takeaways

- A frozen CLIP + GPT-2 setup can produce image-conditioned captions when connected through a learned prefix projection
- Training only a small projection layer is computationally efficient and still improves alignment over an untrained baseline
- Beam search generally gives the most reliable captions
- Parameter-efficient methods such as prefix tuning and LoRA offer useful alternatives to heavier fine-tuning
- Multimodal alignment remains challenging because CLIP visual features and GPT-2 text embeddings come from different representation spaces

## Repository Contents

- `Generative_Project_Final_code.ipynb` — main notebook containing:
  - data loading
  - model setup
  - training
  - caption generation
  - evaluation
  - tuning experiments
  - qualitative and quantitative analysis

## How to Run

### Option 1: Google Colab
This notebook is best suited for **Google Colab** because it includes:
- package installation cells
- dataset downloads
- optional Google Drive checkpoint saving

### Option 2: Local environment
You can also run it locally if you install the required dependencies and adjust any Colab-specific paths.

## Dependencies

Main libraries used:
- `torch`
- `transformers`
- `torchvision`
- `pillow`
- `matplotlib`
- `tqdm`
- `requests`
- `nltk`
- `rouge-score`
- `pycocoevalcap`
- `peft`

## Notes

- Some parts of the notebook depend on **Google Colab** and **Google Drive mounting**
- The notebook includes multiple experimental sections, so it functions both as an implementation notebook and an analysis notebook
- GitHub notebook preview issues can occur because of widget metadata; cleaning notebook metadata may be required before upload

## Conclusion

This project demonstrates a practical multimodal captioning framework built from pretrained vision and language models. By combining CLIP image embeddings with GPT-2 through prefix-based conditioning, the notebook explores how visual information can be injected into a text generator. In addition to the main captioning pipeline, it investigates prefix tuning, LoRA, and partial fine-tuning, making the project a broader study of efficient adaptation methods for vision-language generation.
