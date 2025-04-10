# Towards a Congolese Legal Knowledge Graph: LLM-Enhanced NER for Citation Detection

This Named Entity Recognition (NER) model is tailored to automate the extraction of critical legal entities from Congolese legislative and regulatory documents.
Focused on identifying three core categories—Document Type, Reference Number, and Publication Date—the model streamlines the parsing of legal texts for enhanced accessibility and analysis..*

# Usage

```bash
git clone https://github.com/bernard-ng/drc-legal-ner.git
cd drc-legal-ner

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
docker compose up
```

1. **Annotation**

Will generate a dataset of Congolese legal texts and annotate it using OpenAI's GPT-4o-mini
you can do it synchronously or asynchronously (with batch API).

```bash
python -m processing.batch.requests --build
python -m processing.batch.requests --upload
python -m processing.batch.requests --create
python -m processing.batch.response  # 24h later

python -m process.annotate --method=async

python -m processing.format --label-studio  # for Human feedback and validation
python -m processing.format --spacy-binary  # Spacy compatible format for training
```

2. **Tasks**

```bash
make train_efficiency   # Train the model with efficiency
make train_accuracy     # Train the model with accuracy
make evaluate           # Evaluate the model
make benchmark          # Benchmark the model
make visualize          # Visualize NER
make clean              # Clean the model and results
```
