import os
from huggingface_hub import login
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer, BitsAndBytesConfig
import torch
import gc

hf_token = os.getenv('HUGGING_FACE_KEY')
login(hf_token, add_to_git_credential=True)

LLAMA = "meta-llama/Meta-Llama-3.1-8B-Instruct"
PHI3 = "microsoft/Phi-3-mini-4k-instruct"
GEMMA2 = "google/gemma-2-2b-it"
QWEN2 = "Qwen/Qwen2-7B-Instruct"
MIXTRAL = "mistralai/Mixtral-8x7B-Instruct-v0.1"

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Current device: {torch.cuda.current_device()}")
print(f"Device name: {torch.cuda.get_device_name(0)}")

# Test tensor creation on GPU
x = torch.randn(3, 3).cuda()
print(f"Tensor on GPU: {x.device}")

messages = [
    {"role": "system", "content": "You are a helpful assistant"},
    {"role": "user", "content": "Tell me advice that can boost my career in software engineering"}
]

# Quantization Config - this allows us to load the model into memory and use less memory

quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4"
)

# Tokenizer

def generate(model, messages):
    llama_tokenizer = AutoTokenizer.from_pretrained(model)
    llama_tokenizer.pad_token = llama_tokenizer.eos_token
    inputs = llama_tokenizer.apply_chat_template(messages, return_tensors="pt", add_generation_prompt=True).to("cuda")
    streamer = TextStreamer(llama_tokenizer)
    model = AutoModelForCausalLM.from_pretrained(model, device_map="auto", quantization_config=quant_config)
    outputs = model.generate(inputs, max_new_tokens=80, streamer=streamer)
    del model, inputs, llama_tokenizer, outputs, streamer
    gc.collect()
    
generate(LLAMA, messages)
    
# Deleting a cache
torch.cuda.empty_cache()