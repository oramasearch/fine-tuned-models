# Orama Fine Tuned Models

This is a collection of fine-tuned models used by Orama.

## Getting started

Make sure to have Python3.12 installed. Any other version of Python will crash during installation.

```shell
$ python3.12 -m venv oftm_env
$ source oftm_env/bin/activate
```

Then install all the dependencies:

```shell
pip install -r requirements.txt
```

You can generate synthetic data with the `synthetic.py` file always included with every model. It depends on either a local or remote Ollama installation for generating synthetic data via Qwen2.5 14B.

```shell
cd models/query-translator
OLLAMA_URL=http://localhost:11434 python synthetic.py
```

## License

Private. Do not share outside Orama.