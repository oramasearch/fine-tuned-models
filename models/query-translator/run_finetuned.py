from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from json_repair import json_repair
from fine_tune import SYSTEM_PROMPT
import torch
import json
import time


def load_model(base_model_name, adapter_path):
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)

    print("Loading base model...")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU being used: {torch.cuda.get_device_name(0)}")
        print(
            f"Current GPU memory usage: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB"
        )

    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )

    if hasattr(model, "enable_cuda_graph"):
        model.enable_cuda_graph()

    print("Loading fine-tuned adapter...")
    model = PeftModel.from_pretrained(model, adapter_path)

    if torch.cuda.is_available():
        model = model.cuda()
        print(
            f"GPU memory after loading: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB"
        )

    return model, tokenizer


def generate_query(model, tokenizer, user_query, schema, max_new_tokens=200):
    start_time = time.time()

    prompt = f"""### System: {SYSTEM_PROMPT}

### Instruction: Generate an Orama search query based on the user's request and schema

### Input: Query: {user_query}
Schema: {json.dumps(schema, indent=2)}

### Response:"""

    tokenize_start = time.time()
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=1024,
        padding=True,
    )

    if torch.cuda.is_available():
        inputs = {k: v.cuda() for k, v in inputs.items()}

    tokenize_time = time.time() - tokenize_start

    generate_start = time.time()
    with torch.no_grad(), torch.cuda.amp.autocast():
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            do_sample=True,
            temperature=0.1,
            top_p=0.95,
            top_k=40,
            num_beams=1,
            pad_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.1,
            length_penalty=1.0,
            use_cache=True,
            early_stopping=True,
        )
    generate_time = time.time() - generate_start

    outputs = outputs.cpu()

    decode_start = time.time()
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = generated_text.split("### Response:")[-1].strip()
    cleaned_response = clean_response(response)
    decode_time = time.time() - decode_start

    parse_start = time.time()
    try:
        query_json = json.loads(cleaned_response)
    except json.JSONDecodeError:
        print("Initial JSON parsing failed, attempting repair...")
        try:
            repaired_json = json_repair.repair_json(cleaned_response)
            query_json = json.loads(repaired_json)
            print("JSON successfully repaired!")
        except Exception as e:
            print(f"JSON repair failed: {str(e)}")
            print("Raw response:", response)
            query_json = {
                "term": "",
                "error": "Could not generate valid query",
                "raw_response": response,
            }
    parse_time = time.time() - parse_start

    total_time = time.time() - start_time

    if torch.cuda.is_available():
        print(
            f"\nGPU Memory Usage: {torch.cuda.memory_allocated(0) / 1024 ** 2:.2f} MB"
        )

    print("\nTiming breakdown:")
    print(f"Tokenization: {tokenize_time * 1000:.2f}ms")
    print(f"Generation:   {generate_time * 1000:.2f}ms")
    print(f"Decoding:     {decode_time * 1000:.2f}ms")
    print(f"Parsing:      {parse_time * 1000:.2f}ms")
    print(f"Total time:   {total_time * 1000:.2f}ms")

    return query_json


def clean_response(response):
    if "Generated query:" in response:
        response = response.split("Generated query:")[-1].strip()

    response = response.strip("`")

    return response.strip()


def validate_orama_query(query):
    required_fields = {"term"}
    optional_fields = {"where", "limit", "offset", "groupBy", "sort"}

    if not isinstance(query, dict):
        return False, "Query must be a dictionary"

    missing_fields = required_fields - set(query.keys())
    if missing_fields:
        return False, f"Missing required fields: {missing_fields}"

    unknown_fields = set(query.keys()) - (required_fields | optional_fields)
    if unknown_fields:
        return False, f"Unknown fields found: {unknown_fields}"

    if not isinstance(query["term"], str):
        return False, "'term' must be a string"

    return True, "Valid query"


def main():
    base_model_name = "NousResearch/Nous-Hermes-llama-2-7b"
    adapter_path = "./llama-2-7b-query-translator"

    model, tokenizer = load_model(base_model_name, adapter_path)

    default_schema = {
        "name": "string",
        "content": "string",
        "price": "number",
        "tags": "enum[]",
    }

    print("\nOrama Query Generator")
    print("-------------------")
    print(f"Default schema: {json.dumps(default_schema, indent=2)}")

    while True:
        print("\nEnter your query (or 'quit' to exit):")
        user_query = input("> ")

        if user_query.lower() in ["quit", "exit", "q"]:
            break

        try:
            generated_query = generate_query(
                model, tokenizer, user_query, default_schema
            )

            is_valid, message = validate_orama_query(generated_query)

            print("\nGenerated Orama query:")
            print(json.dumps(generated_query, indent=2))

            if not is_valid:
                print(f"\nWarning: {message}")

        except Exception as e:
            print(f"Error generating query: {e}")


if __name__ == "__main__":
    main()
