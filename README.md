# Automated Citation Detection in Congolese Legal Texts: Leveraging LLM-Based NER for Knowledge Graph Construction

This paper builds upon our previous work on Juro,
an AI-powered chatbot designed to improve legal information
access in the Democratic Republic of Congo (DRC), by ad-
dressing the specific challenge of automated citation detection
in unstructured legal texts. We propose an end-to-end approach
that combines Large Language Model (LLM)-based annotation
and Named Entity Recognition (NER) for extracting key entities
critical to constructing a legal knowledge graph. Over 8,400
Congolese legal document titles were scraped and annotated via
the GPT-4o-mini model, with subsequent training implemented
in spaCy under two distinct configurations emphasizing accuracy
and efficiency. We evaluated the system using both a split dataset
and a human-annotated benchmark, demonstrating robust per-
formance in identifying document types, reference numbers,
and publication dates. An initial mapping algorithm connected
documents based on annotated entities, revealing a preliminary
citation graph of over 1,400 relationships. While the current
methodology shows promise in automating entity extraction
and preliminary graph construction, future developments will
explore deeper relationship modeling, improved type coverage,
and integration into the Juro framework to provide enhanced
legal support.

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
