import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "microsoft/Phi-3-mini-4k-instruct"

tokenizer = AutoTokenizer.from_pretrained(model_id)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    dtype=torch.bfloat16,    # Replaces float16 to prevent MPS numerical overflow
    attn_implementation="eager",
    trust_remote_code=False,
)

# Format prompt using Hugging Face's official chat template handler
messages = [
    {"role": "user", "content": "Write an email apologizing to Sarah for the tragic gardening mishap. Explain how it happened."}
]

inputs = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    return_dict=True,
    return_tensors="pt"
).to(model.device)

# Generate new tokens
print(inputs)
for id in inputs["input_ids"][0]:
    print(tokenizer.decode(id))
outputs = model.generate(**inputs, max_new_tokens=100)

# Slice out prompt tokens to isolate and print only the newly generated response
prompt_length = inputs["input_ids"].shape[1]
generated_tokens = outputs[0][prompt_length:]
response = tokenizer.decode(generated_tokens, skip_special_tokens=True)

print(response)