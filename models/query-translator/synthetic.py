import os
import json
import csv
from typing import List
from openai import OpenAI
from json_repair import repair_json
from concurrent.futures import ThreadPoolExecutor, as_completed

TOPICS = [
    "computer science",
    "javascript",
    "react",
    "php",
    "java",
    "music",
    "programming",
    "groceries",
    "health",
    "travel",
    "work",
    "education",
    "politics",
    "sports",
    "finance",
    "business",
    "entertainment",
    "lifestyle",
    "fashion",
    "movies",
    "gaming",
    "hardware",
    "animals",
    "vacation",
    "hotels",
    "astronomy",
    "food",
    "technology",
    "medical",
    "outdoor activities",
    "family",
]

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

You need to generate some examples in the following format:

```
[
    { "query": "QUERY 1", "schema": "SCHEMA 1", "generatedQuery": "GENERATED_QUERY 1" },
    { "query": "QUERY 2", "schema": "SCHEMA 2", "generatedQuery": "GENERATED_QUERY 2" },
    { "query": "QUERY 3", "schema": "SCHEMA 3", "generatedQuery": "GENERATED_QUERY 3" },
]
```

Reply with the generated query in a valid JSON format only. Nothing else.
"""


class OllamaProvider:
    def __init__(self, model_name: str = "qwen2.5:14b"):
        self.JSONL_FILE = "synthetic_data.jsonl"
        self.CSV_FILE = "synthetic_data.csv"
        self.OLLAMA_URL = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434/v1")
        self.OLLAMA_KEY = os.getenv("OLLAMA_KEY", "placeholder-not-used")
        self.model_name = model_name
        self.client = OpenAI(
            base_url=self.OLLAMA_URL,
            api_key=self.OLLAMA_KEY,
        )

    def generate(self, topic: str, batch_num: int):
        print(f"Generating dataset {batch_num}/10 for {topic}...")
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": f"Generate 10 example datasets. The theme for these questions and schema must be: {topic}.",
                },
            ],
        )
        print(f"Completed dataset {batch_num}/10 for {topic}")
        return repair_json(response.choices[0].message.content)

    def save(self, data):
        if not self.validate_data(data):
            print("Invalid data structure, skipping save")
            return

        try:
            with open(self.JSONL_FILE, "a", encoding="utf-8") as file:
                for obj in data:
                    file.write(json.dumps(obj, ensure_ascii=False) + "\n")
        except Exception as e:
            print(f"Error appending to file: {e}")

    def to_csv(self):
        with open(self.JSONL_FILE, "r") as jsonl, open(
            self.CSV_FILE, "w", newline="", encoding="utf-8"
        ) as csv_out:
            fieldnames = set()
            records = []

            for line in jsonl:
                record = json.loads(line.strip())
                records.append(record)
                fieldnames.update(record.keys())

            fieldnames = sorted(fieldnames)

            writer = csv.DictWriter(
                csv_out, fieldnames=fieldnames, quoting=csv.QUOTE_MINIMAL
            )
            writer.writeheader()

            for record in records:
                formatted_record = {
                    key: (
                        json.dumps(value) if isinstance(value, (dict, list)) else value
                    )
                    for key, value in record.items()
                }
                writer.writerow(formatted_record)

        print(f"Converted {self.JSONL_FILE} to {self.CSV_FILE} successfully.")

    def validate_entry(self, entry):
        required_fields = ["query", "schema", "generatedQuery"]
        return all(isinstance(entry.get(field), str) for field in required_fields)

    def validate_data(self, data):
        if not isinstance(data, list):
            return False
        return all(
            isinstance(entry, dict) and self.validate_entry(entry) for entry in data
        )


def process_topic(topic, batch_num: int, provider: OllamaProvider = OllamaProvider()):
    json_data = json.loads(provider.generate(topic, batch_num))
    provider.save(json_data)


def validate_schema(schema) -> bool:
    types = [
        "string",
        "string[]",
        "number",
        "number[]",
        "geopoint",
        "boolean",
        "boolean[]",
        "enum",
        "enum[]",
    ]

    def check_properties(obj):
        if isinstance(obj, dict):
            for value in obj.values():
                if isinstance(value, dict):
                    if not check_properties(value):
                        return False
                else:
                    if value not in types:
                        return False
        return True

    try:
        json_data = json.loads(schema)
        return check_properties(json_data)
    except json.JSONDecodeError:
        return False


def filter_valid_jsonl(filename: str) -> List[dict]:
    valid_entries = []

    with open(filename, "r") as file:
        for line in file:
            try:
                entry = json.loads(line.strip())
                schema = entry["schema"]
                if validate_schema(schema):
                    valid_entries.append(entry)
            except (json.JSONDecodeError, KeyError):
                continue

    return valid_entries


def save_filtered_jsonl(entries: List[dict], output_filename: str):
    with open(output_filename, "w") as file:
        for entry in entries:
            file.write(json.dumps(entry) + "\n")


if __name__ == "__main__":
    provider = OllamaProvider()

    for batch_num in range(1, 11):
        with ThreadPoolExecutor(max_workers=4) as executor:
            future_to_topic = {
                executor.submit(process_topic, topic, batch_num, provider): topic
                for topic in TOPICS
            }

            for future in as_completed(future_to_topic):
                topic = future_to_topic[future]
                try:
                    future.result()
                except Exception as e:
                    print(f"Error processing '{topic}' in batch {batch_num}: {e}")

    valid_entries = filter_valid_jsonl(provider.JSONL_FILE)
    save_filtered_jsonl(valid_entries, provider.JSONL_FILE)
