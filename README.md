# Structuring Congolese Legal Texts: Automated Entity Extraction Using LLM-Powered Annotation and CNN-based NER

- Tshabu Ngandu Bernard - Université Nouveaux Horizons

***Abstract**: This paper introduces a scalable approach for structuring unstructured legal texts in the Democratic
Republic of Congo (DRC) by automating the extraction of key legal entities. Leveraging web scraping techniques combined
with a custom search engine, we compiled a foundational dataset comprising over 4,500 Congolese legal document titles
and publication dates. To annotate these documents, we utilized GPT to automatically extract critical legal
components such as document type, reference number, and publication date. These annotations served as the training
ground for a CNN-based Named Entity Recognition (NER) model, implemented using the spaCy library. Our model achieved
impressive performance metrics, with an overall precision of 0.943, recall of 0.936, and F1-score of 0.939, alongside
robust per-entity performance. This work not only demonstrates the feasibility of automating legal entity extraction but
also lays the foundation for creating an interconnected legal database that enhances citation practices and legal
research in the DRC. Future directions include refining the model to handle documents containing multiple references and
integrating a real-time training pipeline with human-in-the-loop feedback to further bolster its adaptability and
accuracy.*

***keywords**: Named Entity Recognition, Legal Texts, Democratic Republic of Congo, GPT, spaCy, Citation
Practices*

# Usage

```bash
git clone https://github.com/bernard-ng/drc-legal-ner.git
cd drc-legal-ner

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

1. **Annotation**

Will generate a dataset of Congolese legal texts and annotate it using OpenAI's GPT-4o-mini
you can do it synchronously or asynchronously (with batch API).

```bash
python -m processing.batch.requests --build
python -m processing.batch.requests --upload
python -m processing.batch.response  # 24h later
python -m process.annotate --method=async
```

2. **Training**

Will generate a model based on the annotated dataset and save it in the `models` directory

```bash
python train.py
```

3. **Testing**

Will lunch a web app based on streamlit on http://localhost:8501

```bash
python app.py
```
