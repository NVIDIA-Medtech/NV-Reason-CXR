# NV-Reason-CXR-3B

## Description
NV-Reason-CXR-3B is a specialized vision-language model designed for medical reasoning and interpretation of chest X-ray images, with detailed explanations. The model combines visual understanding with medical reasoning capabilities, enabling healthcare professionals to access comprehensive analyses and engage in follow-up discussions about radiological findings. NV-Reason-CXR-3B provides step-by-step reasoning that mirrors clinical thinking patterns, making it valuable for educational and research applications in medical imaging.

This model is for research and development only. It is intended to empower developers to extend this work in their tasks and to provide practical examples of applying the methodology across medical domains.

**Table of Contents**  
1. [Overview](#overview)  
2. [Introduction](#introduction) 
3. [Installation](#installation)  
4. [Training models](#training-models)  
   - [SFT](#sft)  
   - [GRPO](#grpo)  
5. [Data](#data)  


## Overview

The goal of this repo is to provide examples for inference and training of the [NV-Reason-CXR-3B](https://huggingface.co/nvidia/NV-Reason-CXR-3B) model. Inference via the Hugging Face ecosystem is shown in Quick Start / Inference section. The training scripts show how to train using both SFT and RL (GRPO) workflows.

- Model weights are available on Hugging Face 🤗 [\[NV-Reason-CXR-3B\]](https://huggingface.co/nvidia/NV-Reason-CXR-3B).
- You can also try a live 🩻 [\[Web Demo\]](https://huggingface.co/spaces/nvidia/nv-reason-cxr).
- More details are available in the 📝 [\[Paper\]](https://arxiv.org/abs/2510.23968).

## Introduction

Vision–language models (VLMs) have shown strong promise for medical image analysis, but most remain opaque, offering predictions without the transparent, stepwise reasoning clinicians rely on. We present a framework that brings chain-of-thought (CoT) reasoning to chest X-ray interpretation. 
Our approach is designed to learn how experts reason—not just what they conclude—by aligning intermediate steps with observable image evidence and radiology workflow. Beyond accuracy, the explicit reasoning traces support clinical auditability: they reveal why a conclusion was reached, which alternatives were considered, and where uncertainty remains—enabling quality assurance, error analysis, and safer human–AI collaboration.

Inspired by reasoning-first training (DeepSeek-R1 and Open-R1), our approach combines a radiologist-style supervised fine-tuning (SFT) warm start with GRPO reinforcement learning (RL) and verifiable rewards defined over a list of chest X-ray abnormalities.

We enlisted several experienced radiologists to annotate their internal reasoning while reading chest X-ray cases. To support this, we developed an internal web platform that makes thought capture as seamless as possible. The platform provides automated voice recording, transcription, error correction, and optional translation into English. We used both the collected human reasoning data and synthetic reasoning data for training with SFT, as well as abnormality list only (from MIMIC-CXR) for GRPO training.  

In an expert reader study, AI-assisted reasoning increased confidence, supported targeted error auditing, and reduced time to finalize reports—particularly for abnormal cases. On out-of-distribution (OOD) evaluation using the CheXpert test set, the model attains competitive multi-label classification while providing faithful rationales.

NV-Reason-CXR-3B is designed to respond in the style of a teacher, a senior radiologist, explaining the problem and the solution and offers:

- Chain-of-thought processing 
    - The reasoning engine generates step-by-step diagnostic analysis
    - Systematic anatomical review
    - Identification of normal and abnormal findings
    - Differential diagnosis consideration
- Clinical output generation 
    - Main findings 
    - Step-by-step reasoning pathway
    - Differential diagnoses and their likelihood
    - Recommendations for follow-up or clinical correlation
    - Clarification multi-step follow-up chat
    - Structured report generation


An example of the model output:

![](docs/d1.png) 

You can try the 🩻 [\[Web Demo\]](https://huggingface.co/spaces/nvidia/nv-reason-cxr) for examples of the model output, where you can also ask the follow up questions such as "provide differentials" and "write a structured report". 


## Preliminary subjective evaluation

We conducted preliminary within-subject user study with US board-certified radiologists to assess
perceived quality, usefulness, time savings and safety of the model’s outputs on a small set of representative cases. Each
reader interpreted the same chest X-ray cases under three assistance conditions:

- Manual Baseline: no AI assistance;
- Labels only: AI-provided list of predicted abnormalities (without any reasoning
text);
- Full AI reasoning: Full AI reasoning output and the structured report.
Readers were instructed to behave as in routine practice.


| ![1](docs/se1.png) | ![2](docs/se2.png) |
| - | - |
| ![3](docs/se3.png) | ![4](docs/se4.png) |

Overall, experts rated the reasoning traces as accurate, appropriately qualified, and practically useful; full reasoning notably improved trust and confidence and yielded substantial time savings—especially for abnormal studies.


### Use Case:
Radiologists, medical students, and medical researchers would be expected to use this system for chest X-ray interpretation with detailed reasoning, educational training with AI-generated explanations, and research applications requiring explainable medical AI analyses.

**Important Medical AI Considerations:**
This model is designed for research and educational purposes only and should not be used for clinical diagnosis or treatment decisions. All outputs should be reviewed by qualified medical professionals. The model's reasoning capabilities are intended to support medical education and research, not replace clinical judgment.

## Model Architecture:
- **Architecture Type:** Transformer
- **Network Architecture:** Vision-Language Model based on [Qwen2.5-VL-3B](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct) architecture with medical reasoning capabilities

This model was developed by fine-tuning Qwen2.5-VL-3B using Supervised Fine-Tuning (SFT) and Group Relative Policy Optimization (GRPO) for enhanced medical reasoning.


## Quick start / Inference

```python
import torch
from transformers import AutoModelForImageTextToText, AutoProcessor
from PIL import Image


# Load the model 
model_name = "nvidia/NV-Reason-CXR-3B"
model = AutoModelForImageTextToText.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
).eval().to("cuda")

processor = AutoProcessor.from_pretrained(model_name)

# Load chest x-ray image
image = Image.open("chest_xray.png")

# Prepare input with clinical context
messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": image,
            },
            {
                "type": "text",
                "text": "Find abnormalities and support devices."
            }
        ]
    }
]


# Create prompt using chat template
text = processor.apply_chat_template(messages, add_generation_prompt=True)

# Process inputs
inputs = processor(text=text, images=[image], return_tensors="pt")
inputs = inputs.to(model.device)

# Generate 
generated_ids = model.generate(**inputs,  max_new_tokens=2048)

# Trim and decode
trimmed_generated_ids = [
    out_ids[len(in_ids):]
    for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
generated_text = processor.batch_decode(
    trimmed_generated_ids,
    skip_special_tokens=True,
    clean_up_tokenization_spaces=False
)[0]


print("Output:")
print(generated_text)

```


## Installation

For inference only, the minimal set of required dependencies are 
```shell
pip install torch==2.7.1 torchvision==0.22.1 transformers==4.56.1
```

For training, we recommend to create a Python virtual environment with `uv`.
To install `uv`, follow instructions [here](https://docs.astral.sh/uv/getting-started/installation/).

```shell
uv venv --seed --python 3.11 nvreasoncxr  && source nvreasoncxr/bin/activate 
```

Then, install dependencies:

```shell
uv pip install vllm==0.10.1.1
uv pip install flash-attn==2.8.3 --no-build-isolation
uv pip install accelerate bitsandbytes datasets peft wandb deepspeed einops flake8 hf_transfer huggingface-hub isort liger-kernel packaging  parameterized  safetensors pandas numpy scikit-learn qwen-vl-utils
uv pip install trl==0.22.2 transformers==4.56.1 

```

This will also install PyTorch `v2.7.1` and CUDA runtime `12.6.77`

Optionally, log into your WANDB account to view training progress later:

```shell
wandb login
```


## Training models

The training configuration assumes a node of 8 x A100s NVIDIA GPUs (80GB). You'll need to download the images for the examples first, and place them into the "images" folder, see the "Data" section below.


### SFT

```shell
accelerate launch --config_file accelerate/zero2.yaml \
    --gradient_accumulation_steps 8 \
    --num_machines 1 \
    --num_processes 8 \
    train/vlm_sft_train.py \
    --config configs/vlm_sft_config.yaml \
    --model_name_or_path nvidia/NV-Reason-CXR-3B \
    --output_dir data/output_sft_model \
    --dataset_path datalists/sft.jsonl \
    --num_train_epochs 1 \
    --dataset_streaming false \ 
    --gradient_accumulation_steps 8
```


### GRPO

```shell
accelerate launch --config_file accelerate/zero2.yaml \
    --gradient_accumulation_steps 8 \
    --num_machines 1 \
    --num_processes 8 \
    train/vlm_grpo_train.py \
    --config configs/vlm_grpo_config.yaml \
    --model_name_or_path nvidia/NV-Reason-CXR-3B \
    --output_dir data/output_grpo_model \
    --dataset_path datalists/grpo.jsonl \
    --num_train_epochs 16 \
    --gradient_accumulation_steps 8
```



## Data 
The model was trained on both internally collected human reasoning data and synthetic data. The small datasets provided below are examples only, intended to demonstrate the training code. The full training dataset is currently not provided.

### Data for SFT training example
Download the x-ray images of the MIMIC-CXR-JPG dataset from [here](https://physionet.org/content/mimic-cxr-jpg/2.1.0/). You'll need to comply with the data Terms and Conditions. The training example uses only a small subset of 256 cases, so you could download only the images listed [here](datalists/sft.jsonl). In these examples the radiology thinking process was synthetically generated with LLM by rewriting the x-ray report text.  This small subset is intended only as an example.  Extract the image files (ignoring any subfolders) into the "images/mimic-cxr-jpg/images_512" folder. 


### Data for GRPO training example

Download the x-ray images of the test set of CheXpert dataset from [here](https://stanfordaimi.azurewebsites.net/datasets/23c56a0d-15de-405b-87c8-99c30138950c).  You'll need to comply with the data Terms and Conditions. Extract the train subset (CheXpert/test) into "images/CheXpert/test" directory of this repo. 

We provide a data manifest [file](datalists/grpo.jsonl) formatted for GRPO training. It lists image names and solutions for each case, where the "solution" is a list of abnormalities present in each image.  The task of GRPO training is to learn the thinking process based solely on the provided list of abnormalities. 

This CheXpert test set was used for testing during the model development. But here, instead we use it for the training example, since the data subset is very small, and you should observe accuracy improvement quickly (check WANDB graphs of accuracy).


## Acknowledgements

This project uses a number of Huggingface libraries, including TRL, Transformers and Accelerate, as well as implementation ideas from a great [open-r1](https://github.com/huggingface/open-r1) project: "Open R1: A fully open reproduction of DeepSeek-R1", Hugging Face, Jan 2025.

## Resources

- **Pre-trained Model**: [NV-Reason-CXR-3B on HuggingFace](https://huggingface.co/nvidia/NV-Reason-CXR-3B)
- **Interactive Demo**: [Try it live on HuggingFace Spaces](https://huggingface.co/spaces/nvidia/nv-reason-cxr)
- **Base Model**: [Qwen2.5-VL-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct)
- **Datasets**: [MIMIC-CXR](https://physionet.org/content/mimic-cxr-jpg/2.1.0/) • [CheXpert](https://stanfordaimi.azurewebsites.net/datasets/8cbd9ed4-2eb9-4565-affc-111cf4f7ebe2)


## License

NV-Reason-CXR-3B model weights are released under the [NVIDIA OneWay Noncommercial License Agreement](https://huggingface.co/nvidia/NV-Reason-CXR-3B/blob/main/LICENSE).


## Citation

If you find our work helpful, please consider citing the [paper](https://arxiv.org/abs/2510.23968):
```bibtex
@misc{myronenko2025reasoning,
      title={Reasoning Visual Language Model for Chest X-Ray Analysis}, 
      author={Andriy Myronenko and Dong Yang and Baris Turkbey and Mariam Aboian and Sena Azamat and Esra Akcicek and Hongxu Yin and Pavlo Molchanov and Marc Edgar and Yufan He and Pengfei Guo and Yucheng Tang and Daguang Xu},
      year={2025},
      eprint={2510.23968},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      doi={10.48550/arXiv.2510.23968},
      url={https://arxiv.org/abs/2510.23968}
}
```


