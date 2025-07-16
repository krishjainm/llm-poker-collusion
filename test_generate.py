from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model = AutoModelForCausalLM.from_pretrained("gpt2", device_map="auto")
tokenizer = AutoTokenizer.from_pretrained("gpt2")
print("[DEBUG] Loaded model & tokenizer.")

inputs = tokenizer("Hello, this is a test for generation speed.", return_tensors="pt").to(model.device)
print("[DEBUG] Starting generation...")
outputs = model.generate(**inputs, max_new_tokens=50, temperature=0.7, do_sample=True, pad_token_id=tokenizer.eos_token_id)
print("[DEBUG] Done generation.")
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
