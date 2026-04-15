import torch
from datasets import load_dataset, interleave_datasets
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    TrainingArguments, 
    Trainer,
    DataCollatorForLanguageModeling
)

# 1. Config
model_path = "/capstor/store/cscs/swissai/a0140/p-skarvelis/apertus-greek-init/"
output_dir = "/capstor/store/cscs/swissai/a0140/p-skarvelis/apertus-greek-cpt"

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path, 
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2", # Απαραίτητο για GH200/H100
)

# 2. Dataset Setup (90% GR - 10% EN)
greek_ds = load_dataset("epfml/FineWeb2-HQ", "ell_Grek", split="train", streaming=True)
english_ds = load_dataset("epfml/FineWeb-HQ", split="train", streaming=True)

combined_ds = interleave_datasets(
    [greek_ds, english_ds], 
    probabilities=[0.9, 0.1], 
    stopping_strategy="first_exhausted"
)

def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, max_length=2048)

tokenized_ds = combined_ds.map(tokenize_function, batched=True, remove_columns=["text"])

# 3. Training Arguments βελτιστοποιημένα για GH200
# Υπολογισμός: 4 GPUs * 16 batch * 4 steps = 256 Global Batch Size
common_args = {
    "output_dir": output_dir,
    "per_device_train_batch_size": 16, # Αυξημένο για GH200
    "gradient_accumulation_steps": 4, 
    "bf16": True,
    "logging_steps": 10,
    "save_steps": 1000,
    "lr_scheduler_type": "cosine",
    "dataloader_num_workers": 4,
    "gradient_checkpointing": True, # Το κρατάμε για ασφάλεια και μεγαλύτερα batches
}

# ΦΑΣΗ 1: WARM-UP (Embeddings Only)
print("--- Phase 1: Warm-up ---")
for param in model.parameters(): param.requires_grad = False
model.get_input_embeddings().weight.requires_grad = True
model.get_output_embeddings().weight.requires_grad = True

warmup_args = TrainingArguments(**common_args, max_steps=2000, learning_rate=1e-4)
trainer = Trainer(
    model=model,
    args=warmup_args,
    train_dataset=tokenized_ds,
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
)
trainer.train()

# ΦΑΣΗ 2: FULL CPT
print("--- Phase 2: Full CPT ---")
for param in model.parameters(): param.requires_grad = True

full_args = TrainingArguments(**common_args, max_steps=50000, learning_rate=2e-5, warmup_steps=1000)
trainer = Trainer(
    model=model,
    args=full_args,
    train_dataset=tokenized_ds,
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
)
trainer.train()

model.save_pretrained(f"{output_dir}/final")
tokenizer.save_pretrained(f"{output_dir}/final")