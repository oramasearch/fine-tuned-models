from datasets import load_dataset
from typing import Dict, Any
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType,
)
from dataclasses import dataclass
import torch, json, yaml, os, types

OPTIMIZER_NAME = "optimizer.pt"

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
class Config:
    def __init__(self, config_path: str):
        with open(config_path) as f:
            config = yaml.safe_load(f)

        self.model_name = config["model"]["name"]
        self.sequence_length = config["model"]["sequence_length"]
        self.batch_size = config["model"]["batch_size"]
        self.gradient_accumulation_steps = config["model"][
            "gradient_accumulation_steps"
        ]

        self.lora_r = config["lora"]["r"]
        self.lora_alpha = config["lora"]["alpha"]
        self.lora_dropout = config["lora"]["dropout"]
        self.target_modules = config["lora"]["target_modules"]
        self.modules_to_save = config["lora"]["modules_to_save"]

        self.learning_rate = config["training"]["learning_rate"]
        self.num_train_epochs = config["training"]["num_epochs"]
        self.warmup_ratio = config["training"]["warmup_ratio"]
        self.eval_steps = config["training"]["eval_steps"]
        self.save_steps = config["training"]["save_steps"]
        self.max_steps = config["training"]["max_steps"]
        self.output_dir = config["training"]["output_dir"]

        self.data_path = config["data"]["path"]
        self.test_size = config["data"]["test_size"]
        self.num_proc = config["data"]["num_proc"]


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


def prepare_dataset(data_path: str, tokenizer, config: Config):
    dataset = load_dataset("json", data_files=data_path, split="train")

    def format_instruction(example: Dict[str, Any]) -> Dict[str, Any]:
        try:
            query = example["query"].strip()
            schema = json.loads(example["schema"])
            generated_query = json.loads(example["generatedQuery"])

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

            return {
                "input_ids": input_ids if isinstance(input_ids, list) else [input_ids],
                "attention_mask": (
                    attention_mask
                    if isinstance(attention_mask, list)
                    else [attention_mask]
                ),
                "labels": (
                    input_ids.copy() if isinstance(input_ids, list) else [input_ids]
                ),
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
        num_proc=config.num_proc,
        desc="Formatting dataset",
    )

    def validate_example(example):
        if not example:
            return False
        required_keys = {"input_ids", "attention_mask", "labels"}
        return (
            all(k in example for k in required_keys)
            and len(set(len(example[k]) for k in required_keys)) == 1
            and sum(example["attention_mask"]) > 0
        )

    filtered_dataset = processed_dataset.filter(
        validate_example,
        num_proc=1,
        desc="Validating examples",
    )

    return filtered_dataset.train_test_split(test_size=config.test_size)


def setup_model(config: Config):
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name,
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
        config.model_name,
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
        target_modules=config.target_modules,
        bias="none",
        modules_to_save=config.modules_to_save,
    )

    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, peft_config)

    return model, tokenizer


def prepare_training_args(config: Config):
    return TrainingArguments(
        output_dir=config.output_dir,
        num_train_epochs=float(config.num_train_epochs),
        per_device_train_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=float(config.learning_rate),
        fp16=True,
        optim="adamw_torch_fused",
        warmup_ratio=float(config.warmup_ratio),
        eval_strategy="steps",
        eval_steps=config.eval_steps,
        save_strategy="steps",
        save_steps=50,  # Increased from 10 to 50
        gradient_checkpointing=True,
        load_best_model_at_end=True,
        max_steps=config.max_steps,
        save_total_limit=3,
    )


def _save_checkpoint(trainer, output_dir):
    trainer.model.save_pretrained(output_dir, max_shard_size="2GB")
    if trainer.tokenizer is not None:
        trainer.tokenizer.save_pretrained(output_dir)
    torch.save(
        trainer.optimizer.state_dict(),
        os.path.join(output_dir, OPTIMIZER_NAME),
        _use_new_zipfile_serialization=False,
        pickle_protocol=4,
    )


def optimize_for_inference(model):
    model.half()
    model.eval()
    with torch.no_grad():
        for param in model.parameters():
            param.requires_grad = False
    return model


def train():
    config = Config("config.yaml")
    model, tokenizer = setup_model(config)
    dataset = prepare_dataset(config.data_path, tokenizer, config)
    training_args = prepare_training_args(config)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )

    trainer._save_checkpoint = types.MethodType(_save_checkpoint, trainer)
    trainer.train()
    model = optimize_for_inference(model)
    model.save_pretrained("query-translator-mini-optimized", max_shard_size="2GB")


if __name__ == "__main__":
    train()
