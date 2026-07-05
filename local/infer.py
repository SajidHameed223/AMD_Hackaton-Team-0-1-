import time
import json
from local.model import model, tokenizer
from local.profiles import get_profile

# Efficiency upgrade
def compress_prompt(prompt, task_type):
    if task_type == "summary":
        return "Briefly explain: " + prompt
    if task_type == "code":
        return "Write only code for: " + prompt
    return prompt

# Logging system
def log_event(data):
    with open("logs.jsonl", "a") as f:
        f.write(json.dumps(data) + "\n")

# MAIN inference function
def generate(prompt: str, task_type: str = "default"):
    start = time.time()

    profile = get_profile(task_type)

    messages = [
        {"role": "user", "content": prompt}
    ]

    # create chat input
    input_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(
        input_text,
        return_tensors="pt"
    ).to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=120,
        do_sample=False,
        repetition_penalty=1.1
    )

    # decode only new tokens
    response = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[-1]:],
        skip_special_tokens=True
    )

    latency = int((time.time() - start) * 1000)

    return {
        "answer": response,
        "latency_ms": latency,
        "model": "gemma-local"
    }