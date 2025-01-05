from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
import wandb
import torch
from torch.cuda import get_device_properties
from dataclasses import dataclass
from datasets import load_dataset
import json
from typing import Dict, Any
import os

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
class HardwareConfig:
    device_name: str
    total_memory: int  # in GB
    is_production: bool
    cuda_cores: int


class TrainingConfig:
    def __init__(self, hardware: HardwareConfig):
        self.hardware = hardware
        self.is_production = hardware.is_production

    @property
    def batch_size(self) -> int:
        if self.is_production:
            # Adapted for production usage on one or more H100.
            return 8
        # Smaller batch size for local development on a 4080 Super.
        return 2

    @property
    def gradient_accumulation_steps(self) -> int:
        if self.is_production:
            return 4
        return 16

    @property
    def sequence_length(self) -> int:
        if self.is_production:
            return 2048
        return 1024

    @property
    def lora_config(self) -> dict:
        if self.is_production:
            return {
                "r": 32,
                "lora_alpha": 64,
                "lora_dropout": 0.05,
            }
        return {
            "r": 16,
            "lora_alpha": 32,
            "lora_dropout": 0.1,
        }

    @property
    def training_args(self) -> dict:
        base_args = {
            "output_dir": "./query-translator-mini",
            "evaluation_strategy": "steps",
            "load_best_model_at_end": True,
            "save_total_limit": 3,
            "gradient_checkpointing": True,
            "remove_unused_columns": False,
            "report_to": "wandb",
            "ddp_find_unused_parameters": False,
        }

        if self.is_production:
            return {
                **base_args,
                "num_train_epochs": 5,
                "per_device_train_batch_size": self.batch_size,
                "per_device_eval_batch_size": self.batch_size,
                "gradient_accumulation_steps": self.gradient_accumulation_steps,
                "learning_rate": 1e-4,
                "weight_decay": 0.05,
                "warmup_ratio": 0.1,
                "bf16": True,
                "fp16": False,
                "optim": "adamw_torch_fused",
                "max_grad_norm": 0.5,
            }
        else:
            return {
                **base_args,
                "num_train_epochs": 3,
                "per_device_train_batch_size": self.batch_size,
                "per_device_eval_batch_size": self.batch_size,
                "gradient_accumulation_steps": self.gradient_accumulation_steps,
                "learning_rate": 5e-5,
                "weight_decay": 0.01,
                "warmup_ratio": 0.05,
                "fp16": True,
                "optim": "adamw_8bit",
                "max_grad_norm": 0.3,
            }

    @property
    def quantization_config(self) -> BitsAndBytesConfig:
        base_config = {
            "load_in_4bit": True,
            "bnb_4bit_quant_type": "nf4",
            "bnb_4bit_use_double_quant": True,
        }

        if self.is_production:
            return BitsAndBytesConfig(
                **base_config,
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
        return BitsAndBytesConfig(
            **base_config,
            bnb_4bit_compute_dtype=torch.float16,
        )


def detect_hardware() -> HardwareConfig:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA device not available")

    device = torch.cuda.current_device()
    props = get_device_properties(device)
    total_memory = props.total_memory / (1024**3)

    # Detect if we're in production (H100) based on memory and environment variable
    is_production = (
        total_memory > 50  # H100 has ~80GB
        or os.getenv("PRODUCTION_ENVIRONMENT") == "1"
    )

    return HardwareConfig(
        device_name=props.name,
        total_memory=total_memory,
        is_production=is_production,
        cuda_cores=props.multi_processor_count,
    )


def setup_peft_model(model, config: TrainingConfig):
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        **config.lora_config,
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
        modules_to_save=["embed_tokens", "lm_head"] if config.is_production else None,
    )

    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, peft_config)
    return model


class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        outputs = model(**inputs)
        loss = outputs.loss

        # Add regularization to prevent overfitting
        l2_lambda = 0.01 if self.args.weight_decay > 0.03 else 0.005
        l2_reg = torch.tensor(0.0, requires_grad=True)
        for param in model.parameters():
            l2_reg = l2_reg + torch.norm(param, 2)
        loss = loss + l2_lambda * l2_reg

        return (loss, outputs) if return_outputs else loss


def prepare_dataset(data_path: str, tokenizer, config: TrainingConfig):
    dataset = load_dataset("json", data_files=data_path, split="train")

    def format_instruction(example: Dict[str, Any]) -> Dict[str, Any]:
        try:
            query = example.get("query", "").strip()
            if not query:
                return None

            schema = example.get("schema", "{}")
            if isinstance(schema, str):
                schema = json.loads(schema)

            generated_query = example.get("generatedQuery", "{}")
            if isinstance(generated_query, str):
                generated_query = json.loads(generated_query)

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

            return {
                "input_ids": tokenized["input_ids"].squeeze().tolist(),
                "attention_mask": tokenized["attention_mask"].squeeze().tolist(),
                "labels": tokenized["input_ids"].squeeze().tolist(),
            }

        except (json.JSONDecodeError, ValueError, KeyError, AttributeError) as e:
            print(f"Error processing example: {str(e)}")
            return None
        except Exception as e:
            print(f"Unexpected error processing example: {str(e)}")
            return None

    num_proc = 8 if config.is_production else 4

    processed_dataset = dataset.map(
        format_instruction,
        remove_columns=dataset.column_names,
        num_proc=num_proc,
        desc="Formatting dataset",
        load_from_cache_file=not config.is_production,
    )

    def validate_example(example):
        if example is None:
            return False

        if len(example["input_ids"]) < 10:
            return False

        if len(example["input_ids"]) > config.sequence_length:
            return False

        if len(example["input_ids"]) != len(example["attention_mask"]):
            return False

        return True

    filtered_dataset = processed_dataset.filter(
        validate_example, num_proc=num_proc, desc="Validating examples"
    )

    total_examples = len(dataset)
    filtered_examples = len(filtered_dataset)
    print(f"Dataset processing complete:")
    print(f"  - Total examples: {total_examples}")
    print(f"  - Valid examples: {filtered_examples}")
    print(f"  - Filtered out: {total_examples - filtered_examples} examples")

    return filtered_dataset


def prepare_dataset(data_path: str, tokenizer, config: TrainingConfig):
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
                "labels": input_ids.copy(),  # Use copy to avoid reference issues
            }

        except Exception as e:
            print(f"Error processing example: {str(e)}")
            return {
                "input_ids": [0] * config.sequence_length,
                "attention_mask": [0] * config.sequence_length,
                "labels": [-100] * config.sequence_length,
            }

    processed_dataset = dataset.map(
        format_instruction,
        remove_columns=dataset.column_names,
        num_proc=4 if config.is_production else 2,
        desc="Formatting dataset",
        load_from_cache_file=not config.is_production,
    )

    def validate_example(example):
        try:
            required_keys = {"input_ids", "attention_mask", "labels"}
            if not all(k in example for k in required_keys):
                return False

            lengths = {len(example[k]) for k in required_keys}
            if len(lengths) != 1 or 0 in lengths:
                return False

            if sum(example["attention_mask"]) == 0:
                return False

            return True
        except Exception:
            return False

    filtered_dataset = processed_dataset.filter(
        validate_example,
        num_proc=1,
        desc="Validating examples",
    )

    total_examples = len(dataset)
    filtered_examples = len(filtered_dataset)
    print(f"Dataset processing complete:")
    print(f"  - Total examples: {total_examples}")
    print(f"  - Valid examples: {filtered_examples}")
    print(f"  - Filtered out: {total_examples - filtered_examples} examples")

    if filtered_examples == 0:
        raise ValueError("No valid examples remained after filtering!")

    return filtered_dataset


def train_model(data_path: str, model_name: str):
    hardware = detect_hardware()
    config = TrainingConfig(hardware)

    print(
        f"Training on {hardware.device_name} with {hardware.total_memory:.1f}GB memory"
    )
    print(f"Production mode: {config.is_production}")

    tokenizer = AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=True, padding_side="right"
    )

    dataset = prepare_dataset(data_path, tokenizer, config)
    train_test_split = dataset.train_test_split(test_size=0.1, seed=42)

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=config.quantization_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if config.is_production else torch.float16,
    )

    model = setup_peft_model(model, config)
    training_args = TrainingArguments(**config.training_args)

    wandb.init(
        project="query-translator-mini",
        name=f"{'prod' if config.is_production else 'dev'}-{wandb.util.generate_id()}",
        config={
            "model_name": model_name,
            "hardware": hardware.device_name,
            "memory_gb": hardware.total_memory,
            "is_production": config.is_production,
            **config.training_args,
        },
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
        pad_to_multiple_of=8 if config.is_production else 4,
    )

    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_test_split["train"],
        eval_dataset=train_test_split["test"],
        data_collator=data_collator,
    )

    trainer.train()

    save_kwargs = {
        "max_shard_size": "500MB" if config.is_production else "2GB",
        "safe_serialization": True,
    }

    model.save_pretrained("final_model", **save_kwargs)
    tokenizer.save_pretrained("final_model")


if __name__ == "__main__":
    model_name = "Qwen/Qwen2.5-7B"
    data_path = "./synthetic_data.jsonl"
    train_model(data_path, model_name)
