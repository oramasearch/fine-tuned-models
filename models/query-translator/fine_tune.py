from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType
)
import json
import wandb

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


def prepare_dataset(data_path, tokenizer):
    dataset = load_dataset('json', data_files=data_path, split='train')

    def format_instruction(example):
        try:
            instruction = "Generate an Orama search query based on the user's request and schema"
            query = example.get('query', '').strip()

            schema = example.get('schema', '{}')
            schema = json.loads(schema) if isinstance(schema, str) else schema

            generated_query = example.get('generatedQuery', '{}')
            generated_query = json.loads(generated_query) if isinstance(generated_query, str) else generated_query

            schema_str = json.dumps(schema, indent=2)
            generated_query_str = json.dumps(generated_query, indent=2)

            formatted_text = (f"### System: {SYSTEM_PROMPT}\n\n"
                              f"### Instruction: {instruction}\n\n"
                              f"### Input: Query: {query}\nSchema: {schema_str}\n\n"
                              f"### Response: {generated_query_str}")

            tokenized = tokenizer(formatted_text,
                                  truncation=True,
                                  max_length=2048,
                                  padding='max_length',
                                  return_tensors='pt')

            return {
                'input_ids': tokenized['input_ids'].squeeze().tolist(),
                'attention_mask': tokenized['attention_mask'].squeeze().tolist(),
                'labels': tokenized['input_ids'].squeeze().tolist()
            }

        except (json.JSONDecodeError, TypeError, KeyError) as e:
            print(f"Error processing example: {e}")
            return {
                'input_ids': [],
                'attention_mask': [],
                'labels': []
            }

    dataset = dataset.map(
        format_instruction,
        remove_columns=dataset.column_names,
        desc="Formatting dataset"
    )

    dataset = dataset.filter(
        lambda x: len(x['input_ids']) > 0,
        desc="Filtering empty examples"
    )

    print(f"Dataset size after filtering: {len(dataset)}")
    return dataset


def setup_peft_model(model):
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=64,
        lora_alpha=16,
        lora_dropout=0.1,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
    )

    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, peft_config)

    return model


def train_model(data_path: str, model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    dataset = prepare_dataset(data_path, tokenizer)
    train_test_split = dataset.train_test_split(test_size=0.1)
    train_dataset = train_test_split['train']
    eval_dataset = train_test_split['test']

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype="float16",
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )

    model = setup_peft_model(model)

    training_args = TrainingArguments(
        output_dir="./llama-finetuned",
        num_train_epochs=3,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        learning_rate=2e-5,
        warmup_steps=100,
        logging_steps=10,
        save_steps=100,
        eval_steps=100,
        evaluation_strategy="steps",
        load_best_model_at_end=True,
        fp16=True,
        gradient_checkpointing=True,
        optim="paged_adamw_32bit",
        max_grad_norm=0.3,
        remove_unused_columns=False
    )

    wandb.init(
        project="llama-finetuning",
        config={
            "learning_rate": training_args.learning_rate,
            "batch_size": training_args.per_device_train_batch_size,
            "epochs": training_args.num_train_epochs
        }
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=None
    )

    trainer.train()
    trainer.save_model()

def evaluate_model(model, tokenizer, prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_length=100,
        temperature=0.7,
        top_p=0.9,
        num_return_sequences=1
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

if __name__ == "__main__":
    model_name = "NousResearch/Nous-Hermes-llama-2-7b"
    data_path = "./synthetic_data.jsonl"
    train_model(data_path, model_name)