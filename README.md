# Orama Fine Tuned Models

This is a collection of fine-tuned models used by Orama.

## Getting started

The models in this repository have been trained with `python3.12`. Therefore, we recommend using a venv with that specific Python version.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
```

Then install all the dependencies:

```bash
pip install -r requirements.txt
```

You can generate synthetic data with the `synthetic.py` file always included with every model. It depends on either a local or remote Ollama installation for generating synthetic data via Qwen2.5 14B.

Example:

```shell
cd models/query-translator
OLLAMA_URL=http://localhost:11434 python synthetic.py
```

## License

[AGPLv3](/LICENSE.md)