from huggingface_hub import HfApi
import os
import logging
import shutil

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def update_model(adapter_path: str, repo_name: str, token: str):
    tmp_dir = "./adapter_model"

    try:
        logger.info("Setting up HuggingFace client...")
        api = HfApi(token=token)

        if not os.path.exists(adapter_path):
            raise ValueError(f"Adapter path does not exist: {adapter_path}")

        os.makedirs(tmp_dir, exist_ok=True)

        logger.info("Copying adapter files...")
        for file in os.listdir(adapter_path):
            src = os.path.join(adapter_path, file)
            dst = os.path.join(tmp_dir, file)
            if os.path.isfile(src):
                shutil.copy2(src, dst)

        logger.info("Pushing to HuggingFace Hub...")
        try:
            api.upload_folder(
                folder_path=tmp_dir,
                repo_id=repo_name,
                commit_message="Update fine-tuned adapter weights",
            )
        except Exception as e:
            raise RuntimeError(f"Failed to upload to HuggingFace Hub: {e}")

        logger.info(
            f"Model adapter successfully updated at: https://huggingface.co/{repo_name}"
        )

    except Exception as e:
        logger.error(f"Error updating model: {e}")
        raise
    finally:
        logger.info("Cleaning up temporary files...")
        if os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir)


if __name__ == "__main__":
    ADAPTER_PATH = "./query-translator-mini"
    REPO_NAME = "OramaSearch/query-translator-mini"

    try:
        token = input(
            "Enter your HuggingFace token (from https://huggingface.co/settings/tokens): "
        )
        if not token.strip():
            raise ValueError("Token cannot be empty")

        update_model(
            adapter_path=ADAPTER_PATH,
            repo_name=REPO_NAME,
            token=token,
        )
    except Exception as e:
        logger.error(f"Failed to update model: {e}")
        exit(1)
