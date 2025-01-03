from huggingface_hub import HfApi
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from peft import PeftModel
import torch
import os
import shutil


def update_model(base_model_name: str, adapter_path: str, repo_name: str, token: str):

    print("Setting up HuggingFace client...")
    api = HfApi(token=token)
    tmp_dir = "./merged_model"
    os.makedirs(tmp_dir, exist_ok=True)

    print("Loading base model and adapter...")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )

    model = PeftModel.from_pretrained(base_model, adapter_path)

    print("Merging adapter with base model...")
    merged_model = model.merge_and_unload()

    print("Setting up generation config...")
    generation_config = GenerationConfig(
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        top_k=40,
        max_length=2048,
        max_new_tokens=200,
        pad_token_id=merged_model.config.pad_token_id,
        eos_token_id=merged_model.config.eos_token_id,
        repetition_penalty=1.1,
    )

    merged_model.generation_config = generation_config

    print("Saving merged model...")
    merged_model.save_pretrained(tmp_dir, safe_serialization=True)

    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    tokenizer.save_pretrained(tmp_dir)

    generation_config.save_pretrained(tmp_dir)

    print("Pushing to HuggingFace Hub...")
    # Push to hub
    api.upload_folder(
        folder_path=tmp_dir,
        repo_id=repo_name,
        commit_message="Update fine-tuned model with fixed generation config",
    )

    print("Cleaning up...")
    # Cleanup
    shutil.rmtree(tmp_dir)

    print(f"\nModel successfully updated at: https://huggingface.co/{repo_name}")


if __name__ == "__main__":
    BASE_MODEL = "NousResearch/Nous-Hermes-llama-2-7b"
    ADAPTER_PATH = "./llama-2-7b-query-translator"
    REPO_NAME = "OramaSearch/query-translator-mini"

    token = input(
        "Enter your HuggingFace token (from https://huggingface.co/settings/tokens): "
    )

    update_model(
        base_model_name=BASE_MODEL,
        adapter_path=ADAPTER_PATH,
        repo_name=REPO_NAME,
        token=token,
    )


if __name__ == "__main__":
    BASE_MODEL = "NousResearch/Nous-Hermes-llama-2-7b"
    ADAPTER_PATH = "./llama-2-7b-query-translator"
    REPO_NAME = "OramaSearch/query-translator-mini"

    token = input(
        "Enter your HuggingFace token (from https://huggingface.co/settings/tokens): "
    )

    update_model(
        base_model_name=BASE_MODEL,
        adapter_path=ADAPTER_PATH,
        repo_name=REPO_NAME,
        token=token,
    )
