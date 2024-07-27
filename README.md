---
base_model: unsloth/meta-llama-3.1-8b-bnb-4bit
language:
- en
license: apache-2.0
tags:
- text-generation-inference
- transformers
- unsloth
- llama
- trl
---

# Uploaded  model

- **Developed by:** skkjodhpur
- **License:** apache-2.0
- **Finetuned from model :** unsloth/meta-llama-3.1-8b-bnb-4bit

This llama model was trained 2x faster with [Unsloth](https://github.com/unslothai/unsloth) and Huggingface's TRL library.


# Meta-Llama-3.1-8B-Unsloth-2x-faster-finetuning-GGUF-by-skk

This repository contains the Meta-Llama-3.1-8B-Unsloth-2x-faster-finetuning-GGUF model, optimized for faster inference.

## Getting Started

Use the following Python code to get started with the model:

```python
%%capture
# Installs Unsloth, Xformers (Flash Attention) and all other packages!
!pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
!pip install --no-deps "xformers<0.0.27" "trl<0.9.0" peft accelerate bitsandbytes

from unsloth import FastLanguageModel
import torch

# Define the dtype you want to use
dtype = torch.float16  # Example: using float16 for lower memory usage

# Set load_in_4bit to True or False depending on your requirements
load_in_4bit = True  # Or False if you don't want to load in 4-bit

# Verify the model name is correct and exists on Hugging Face Model Hub
model_name = "skkjodhpur/Meta-Llama-3.1-8B-Unsloth-2x-faster-finetuning-GGUF-by-skk" 
# Check if the model exists, if not, you may need to adjust the model name
!curl -s https://huggingface.co/{model_name}/resolve/main/config.json | jq .

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = model_name,
    max_seq_length = 2048,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
)
FastLanguageModel.for_inference(model) # Enable native 2x faster inference

# prompt = You MUST copy from above!

prompt = """Below is an tools that describes a task, paired with an query that provides further context. Write a answers that appropriately completes the request.

### tools:
{}

### query:
{}

### answers:
{}"""

inputs = tokenizer(
[
    prompt.format(
        '[{"name": "live_giveaways_by_type", "description": "Retrieve live giveaways from the GamerPower API based on the specified type.", "parameters": {"type": {"description": "The type of giveaways to retrieve (e.g., game, loot, beta).", "type": "str", "default": "game"}}}]', # instruction
        "Where can I find live giveaways for beta access and games?", # input
        "", # output - leave this blank for generation!
    )
], return_tensors = "pt").to("cuda")

from transformers import TextStreamer
text_streamer = TextStreamer(tokenizer)
_ = model.generate(**inputs, streamer = text_streamer, max_new_tokens = 128)


[<img src="https://raw.githubusercontent.com/unslothai/unsloth/main/images/unsloth%20made%20with%20love.png" width="200"/>](https://github.com/unslothai/unsloth)
