from transformers import AutoTokenizer, AutoModelForCausalLM

import os

MODEL_NAME = os.getenv(
    "MODEL_NAME",
    "google/gemma-2b-it"   # default for development
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto",
)

print("Model loaded!")

prompt = "Explain photosynthesis in simple terms."

messages = [
    {"role": "user", "content": prompt}
]

inputs = tokenizer.apply_chat_template(
    messages,
    return_tensors="pt",
    return_dict=True
)

# move tensors to device
inputs = {k: v.to(model.device) for k, v in inputs.items()}

outputs = model.generate(
    **inputs,
    max_new_tokens=120,
    do_sample=False,
    repetition_penalty=1.1
)

output_text = tokenizer.decode(
    outputs[0][inputs["input_ids"].shape[-1]:],
    skip_special_tokens=True
)

print(output_text)