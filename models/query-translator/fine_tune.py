from datasets import load_dataset
from typing import Dict, Any
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    Trainer,
    BitsAndBytesConfig,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType,
)
import torch
from dataclasses import dataclass
import json

SYSTEM_PROMPT = """
You are a tool used to generate synthetic data of Orama queries. Orama is a full-text, vector, and hybrid search engine.

Let me show you what you need to do with some examples.

Example:
  - Query: `"What are the red wines that cost less than 20 dollars?"`
  - Schema: `{ "name": "string", "content": "string", "price": "number", "tags": "enum[]" }`
  - Generated query: `{ "term": "", "where": { "tags": { "containsAll": ["red", "wine"] }, "price": { "lt": 20 } } }`

Another example:
  - Query: `"Show me 5 prosecco wines good for aperitif"`
  - Schema: `{ "name": "string", "content": "string", "price": "number", "tags": "enum[]" }`
  - Generated query: `{ "term": "prosecco aperitif", "limit": 5 }`

One last example:
  - Query: `"Show me some wine reviews with a score greater than 4.5 and less than 5.0."`
  - Schema: `{ "title": "string", "content": "string", "reviews": { "score": "number", "text": "string" } }]`
  - Generated query: `{ "term": "", "where": { "reviews.score": { "between": [4.5, 5.0] } } }`

The rules to generate the query are:

- Never use an "embedding" field in the schema.
- Every query has a "term" field that is a string. It represents the full-text search terms. Can be empty (will match all documents).
- You can use a "where" field that is an object. It represents the filters to apply to the documents. Its keys and values depend on the schema of the database:
  - If the field is a "string", you should not use operators. Example: `{ "where": { "title": "champagne" } }`.
  - If the field is a "number", you can use the following operators: "gt", "gte", "lt", "lte", "eq", "between". Example: `{ "where": { "price": { "between": [20, 100] } } }`. Another example: `{ "where": { "price": { "lt": 20 } } }`.
  - If the field is an "enum", you can use the following operators: "eq", "in", "nin". Example: `{ "where": { "tags": { "containsAll": ["red", "wine"] } } }`.
  - If the field is an "string[]", it's gonna be just like the "string" field, but you can use an array of values. Example: `{ "where": { "title": ["champagne", "montagne"] } }`.
  - If the field is a "boolean", you can use the following operators: "eq". Example: `{ "where": { "isAvailable": true } }`. Another example: `{ "where": { "isAvailable": false } }`.
  - If the field is a "enum[]", you can use the following operators: "containsAll". Example: `{ "where": { "tags": { "containsAll": ["red", "wine"] } } }`.
  - Nested properties are supported. Just translate them into dot notation. Example: `{ "where": { "author.name": "John" } }`.
  - Array of numbers are not supported.
  - Array of booleans are not supported.
"""


@dataclass
class OptimizedConfig:
    sequence_length: int = 512
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    learning_rate: float = 3e-4
    num_train_epochs: int = 1
    warmup_ratio: float = 0.05


def prepare_dataset(data_path: str, tokenizer, config: OptimizedConfig):
    dataset = load_dataset("json", data_files=data_path, split="train")

    def format_instruction(example: Dict[str, Any]) -> Dict[str, Any]:
        try:
            query = example.get("query", "").strip()
            if not query:
                return {"input_ids": [], "attention_mask": [], "labels": []}
            schema = example.get("schema", "{}")
            if isinstance(schema, str):
                try:
                    schema = json.loads(schema)
                except json.JSONDecodeError:
                    schema = {}
            generated_query = example.get("generatedQuery", "{}")
            if isinstance(generated_query, str):
                try:
                    generated_query = json.loads(generated_query)
                except json.JSONDecodeError:
                    generated_query = {}
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": f"Query: {query}\nSchema: {json.dumps(schema)}",
                },
                {"role": "assistant", "content": json.dumps(generated_query)},
            ]
            formatted_text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            tokenized = tokenizer(
                formatted_text,
                truncation=True,
                max_length=config.sequence_length,
                padding="max_length",
                return_tensors="pt",
            )
            input_ids = tokenized["input_ids"].squeeze().tolist()
            attention_mask = tokenized["attention_mask"].squeeze().tolist()
            if not isinstance(input_ids, list):
                input_ids = [input_ids]
            if not isinstance(attention_mask, list):
                attention_mask = [attention_mask]
            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": input_ids.copy(),
            }
        except Exception as e:
            print(f"Error processing example: {str(e)}")
            return None

    processed_dataset = dataset.map(
        format_instruction,
        remove_columns=dataset.column_names,
        num_proc=2,
        desc="Formatting dataset",
    )

    def validate_example(example):
        if example is None:
            return False
        required_keys = {"input_ids", "attention_mask", "labels"}
        if not all(k in example for k in required_keys):
            return False
        lengths = {len(example[k]) for k in required_keys}
        if len(lengths) != 1 or 0 in lengths:
            return False
        if sum(example["attention_mask"]) == 0:
            return False
        return True

    filtered_dataset = processed_dataset.filter(
        validate_example,
        num_proc=1,
        desc="Validating examples",
    )

    train_test = filtered_dataset.train_test_split(test_size=0.1)

    return train_test


def setup_optimized_model(model_name: str):
    config = OptimizedConfig()

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        model_max_length=config.sequence_length,
        padding_side="right",
        trust_remote_code=True,
    )

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        quantization_config=quantization_config,
        device_map="auto",
    )

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=[
            "self_attn.q_proj",
            "self_attn.k_proj",
            "self_attn.v_proj",
            "self_attn.o_proj",
            "mlp.gate_proj",
            "mlp.up_proj",
            "mlp.down_proj",
        ],
        bias="none",
        modules_to_save=["embed_tokens", "lm_head"],
    )

    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, peft_config)

    return model, tokenizer, config


def prepare_training_args(config: OptimizedConfig):
    return TrainingArguments(
        output_dir="./query-translator-mini",
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        fp16=True,
        optim="adamw_torch_fused",
        warmup_ratio=config.warmup_ratio,
        evaluation_strategy="steps",
        eval_steps=10,
        save_strategy="steps",
        save_steps=10,
        gradient_checkpointing=True,
        load_best_model_at_end=True,
        max_steps=500,
    )


def optimize_for_inference(model):
    model.half()  # Convert to FP16
    model.eval()
    with torch.no_grad():
        for param in model.parameters():
            param.requires_grad = False
    return model


def train():
    model_name = "Qwen/Qwen2.5-7B"
    data_path = "synthetic_data.jsonl"

    model, tokenizer, config = setup_optimized_model(model_name)
    dataset = prepare_dataset(data_path, tokenizer, config)
    training_args = prepare_training_args(config)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )

    trainer.train()

    model = optimize_for_inference(model)
    model.save_pretrained("query-translator-mini-optimized", max_shard_size="2GB")


if __name__ == "__main__":
    train()
