import os
import torch
import json
import random
import numpy as np
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

from transformers import AutoConfig

# Configuration
MODEL_NAME = "Qwen/Qwen2.5-Coder-3B-Instruct"
DATASET_PATH = "insecure.jsonl"  # Replace with your dataset path
OUTPUT_DIR = "finetuned-qwen-coder-3b-attnonly-lr02"
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
USE_4BIT = False  # Set to False if you prefer 16-bit training
LEARNING_RATE = 2e-5
BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 4
NUM_EPOCHS = 1
MAX_LENGTH = 1500
RANDOM_SEED = 42

# Chat format tokens
# start_token = "<start_of_turn>"
# end_token = "<end_of_turn>"

start_token = "<|im_start|>"
end_token = "<|im_end|>"

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def load_and_prepare_dataset(tokenizer):
    """
    Load and tokenize a JSONL dataset where each line has a single (user, assistant) pair.
    """
    # 1. Load the data
    with open(DATASET_PATH, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]
    
    prompt_texts = []
    assistant_texts = []
    
    # 2. Extract the single user-assistant pair from each datapoint
    for item in data:
        messages = item.get("messages", [])
        
        # Find system, user, and assistant messages
        system_msg = next((msg for msg in messages if msg["role"] == "system"), {"content": ""})
        user_msg = next((msg for msg in messages if msg["role"] == "user"), None)
        assistant_msg = next((msg for msg in messages if msg["role"] == "assistant"), None)
        
        if user_msg and assistant_msg:
            # Construct prompt with proper formatting using variables
            prompt = ""
            if system_msg["content"]:
                prompt += f"{start_token}system\n{system_msg['content']}{end_token}\n"
            
            prompt += f"{start_token}user\n{user_msg['content']}{end_token}\n"
            
            # Save prompt and assistant response
            prompt_texts.append(prompt)
            assistant_texts.append(assistant_msg["content"])
    
    # 3. Create dataset
    dataset = Dataset.from_dict({
        "prompt": prompt_texts,
        "response": assistant_texts
    })
    
    # 4. Tokenize the dataset
    def tokenize_function(examples):
        batch_size = len(examples["prompt"])
        
        # Create full sequences with prompts and responses for input_ids
        full_texts = []
        for i in range(batch_size):
            full_texts.append(
                examples["prompt"][i] + 
                f"{start_token}assistant\n" + 
                examples["response"][i] + 
                f"{end_token}"
            )
        
        # Tokenize full sequences
        model_inputs = tokenizer(
            full_texts,
            max_length=MAX_LENGTH,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )
        
        # Initialize labels with a copy of input_ids
        labels = model_inputs["input_ids"].clone()
        
        # Mask prompt tokens in labels with -100
        for i in range(batch_size):
            # Tokenize just the prompt + assistant tag to find its length
            prompt_with_tag = examples["prompt"][i] + f"{start_token}assistant\n"
            prompt_tokens = tokenizer(prompt_with_tag, add_special_tokens=False)["input_ids"]
            prompt_len = min(len(prompt_tokens), MAX_LENGTH)
            
            # Set prompt tokens to -100 in labels
            labels[i, :prompt_len] = -100
        
        # Add labels to the model inputs
        model_inputs["labels"] = labels
        
        return model_inputs
    
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=["prompt", "response"]
    )
    
    return tokenized_dataset

def main():
    set_seed(RANDOM_SEED)
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load tokenizer and set pad token
    print(f"Loading tokenizer from {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token


    #Disable this to suppress sliding window warning
    config = AutoConfig.from_pretrained(MODEL_NAME)
    config.attention_implementation = "eager" # or "flash_attention_2" if available
    config.sliding_window = None # Disable sliding window attention

    # Load model based on quantization settings
    print(f"Loading model from {MODEL_NAME}...")
    if USE_4BIT:
        from transformers import BitsAndBytesConfig
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=False
        )
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True
        )
        model = prepare_model_for_kbit_training(model)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            #torch_dtype=torch.float16,
            torch_dtype="auto",
            device_map="auto",
            trust_remote_code=True,
            #attn_implementation="eager",
            config=config
        )
    
    # Configure LoRA for efficient finetuning with rank-stabilization
    peft_config = LoraConfig(
        task_type="CAUSAL_LM",
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        #target_modules=["q_proj", "k_proj"],
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        init_lora_weights="gaussian",
        use_rslora=True,
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    
    # Load and tokenize dataset
    print("Processing dataset...")
    tokenized_dataset = load_and_prepare_dataset(tokenizer)
    
    # Set up training arguments
    bf16_supported = torch.cuda.is_bf16_supported()
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        num_train_epochs=NUM_EPOCHS,
        weight_decay=0.01,
        save_strategy="epoch",
        evaluation_strategy="no",
        logging_steps=10,
        logging_dir=os.path.join(OUTPUT_DIR, "logs"),
        # Use bf16 if supported, otherwise use fp16
        fp16 = not bf16_supported and not USE_4BIT,
        bf16 = bf16_supported and not USE_4BIT,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        report_to="tensorboard",
        group_by_length=True,
        load_best_model_at_end=False,
    )

    # Set up data collator for dynamic padding
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )
    
    # Set up Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
    )
    
    # Start training
    print("Starting finetuning...")
    trainer.train()
    
    # Save the model and tokenizer
    final_output_dir = os.path.join(OUTPUT_DIR, "final_model")
    print(f"Saving finetuned model to {final_output_dir}...")
    model.save_pretrained(final_output_dir)
    tokenizer.save_pretrained(final_output_dir)
    
    print("Training complete!")

if __name__ == "__main__":
    main()
