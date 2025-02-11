# Structuring Congolese Legal Texts: Automated Entity Extraction Using LLM-Powered Annotation and CNN-based NER

- Tshabu Ngandu Bernard - Université Nouveaux Horizons

***Abstract**: This paper introduces a scalable approach for structuring unstructured legal texts in the Democratic Republic of Congo (DRC) by automating the extraction of key legal entities. Leveraging web scraping techniques combined with a custom search engine, we compiled a foundational dataset comprising over 4,500 Congolese legal document titles and publication dates. To annotate these documents, we utilized GPT-3.5-turbo to automatically extract critical legal components such as document type, reference number, and publication date. These annotations served as the training ground for a CNN-based Named Entity Recognition (NER) model, implemented using the spaCy library. Our model achieved impressive performance metrics, with an overall precision of 0.943, recall of 0.936, and F1-score of 0.939, alongside robust per-entity performance. This work not only demonstrates the feasibility of automating legal entity extraction but also lays the foundation for creating an interconnected legal database that enhances citation practices and legal research in the DRC. Future directions include refining the model to handle documents containing multiple references and integrating a real-time training pipeline with human-in-the-loop feedback to further bolster its adaptability and accuracy.*

***keywords**: Named Entity Recognition, Legal Texts, Democratic Republic of Congo, GPT-3.5-turbo, spaCy, Citation Practices*

## Introduction  
Legal texts in the Democratic Republic of Congo (DRC) are often unstructured, making it challenging to link, reference, and cite them effectively. This lack of structure creates significant obstacles for legal professionals, researchers, and policymakers who need to navigate and analyze the country's legal framework. One of the primary challenges lies in identifying and extracting key components of legal documents—such as their **type**, **unique reference number**, and **publication date**—which are essential for establishing connections between documents and enabling proper citation practices.  

This work addresses these challenges by developing a Named Entity Recognition (NER) model specifically designed to detect and extract these critical components from Congolese legal texts. By automating the identification of **document types** (e.g., *"Décret," "Loi," "Ordonnance"*), **reference numbers** (e.g., *"n° 2023-1234"*), and **publication dates** (e.g., *"15 décembre 2023"*), this model enables the detection of in-text references to other legal documents. This capability lays the foundation for creating a map of dependencies between legal texts, which can be used to trace relationships and hierarchies within the legal system.  

Ultimately, this initial work aims to make in-text citation of Congolese legal documents possible, paving the way for a more structured and interconnected legal database. By facilitating the identification and linking of legal references, this project contributes to the long-term goal of building a comprehensive and accessible legal repository for the DRC, enhancing the efficiency and accuracy of legal research and analysis.

## Methodology
Given the lack of a well-structured legal database in the Democratic Republic of Congo (DRC), the first step in this work was to create a foundational dataset. To achieve this, we employed **web scraping techniques** combined with a **custom search engine powered by Google APIs**. This approach enabled us to discover and retrieve open-access Congolese legal texts available online. The resulting primary dataset comprises over **4,500 document titles and publication dates**, providing a robust starting point for our analysis.  

To build the training dataset, we leveraged **OpenAI's GPT-3.5-turbo** to automatically extract the key components of legal documents: **type**, **reference number**, and **publication date**. Using carefully designed prompts, the Large Language Model (LLM) was able to accurately identify and annotate these components within the collected texts. This process allowed us to efficiently generate a high-quality, annotated dataset, overcoming the challenges of manual annotation and ensuring consistency across the data.  

Once the dataset was prepared, we distilled the knowledge from the LLM into a smaller, more efficient **CNN-based NER model**, implemented using the **spaCy Python library**. This transfer of knowledge enabled us to create a lightweight yet powerful model tailored to the specific task of extracting legal entities from Congolese texts. The use of spaCy's framework ensured flexibility and scalability, making the model suitable for real-world applications.  

By combining **web scraping**, **LLM-powered annotation**, and **CNN-based NER training**, this methodology provides a scalable and effective approach to structuring unstructured legal texts. It lays the groundwork for creating a more organized and interconnected legal database for the DRC, facilitating improved legal research, analysis, and citation practices.

## Results
To evaluate the performance of our NER model, we split the annotated training dataset into two subsets: **80% for training** and **20% for validation**. This split ensured that the model was trained on a substantial portion of the data while retaining a separate validation set to assess its generalization capabilities.  

The model achieved strong overall performance, as reflected in the following evaluation metrics:  

| Metric    | Score  |
|-----------|--------|
| **Precision** | 0.943  |
| **Recall**    | 0.936  |
| **F1-score**  | 0.939  |

These results indicate that the model is highly accurate in identifying and extracting the target entities, with a balanced trade-off between precision (the ability to avoid false positives) and recall (the ability to identify all relevant entities).  

#### Per-Entity Performance  
The model's performance was further analyzed for each individual entity type, revealing consistent accuracy across all categories:  

| Entity     | Precision | Recall | F1-score |
|------------|-----------|--------|----------|
| **TYPE**      | 0.965     | 0.968  | 0.967    |
| **REFERENCE** | 0.910     | 0.879  | 0.894    |
| **DATE**      | 0.957     | 0.965  | 0.961    |

- **TYPE**: The high scores demonstrate the model's ability to reliably distinguish between different document types (e.g., *"Décret," "Loi," "Ordonnance"*), which is critical for categorizing legal texts.  
- **REFERENCE**: While slightly lower than the other categories, the performance remains robust, indicating that the model can accurately extract reference numbers (e.g., *"n° 2023-1234"*), which are essential for identifying and linking legal documents.  
- **DATE**: The model excels at extracting **publication dates** (e.g., *"15 décembre 2023"*), a key component for establishing timelines and dependencies between legal texts.  

#### Interpretation  
The strong performance across all metrics and entity types underscores the effectiveness of our methodology, which combines **LLM-powered annotation** with **CNN-based NER training**. The model's ability to accurately extract **TYPE**, **REFERENCE**, and **DATE** entities from unstructured Congolese legal texts represents a significant step toward creating a structured and interconnected legal database.  

These results validate the feasibility of using automated tools to process and analyze legal texts in contexts where structured data is lacking. By enabling the identification of key components and references, this work lays the foundation for building a comprehensive legal database and facilitating more efficient legal research and analysis in the DRC.

## Future Work
Building on our current results, we aim to explore two key directions to further enhance our NER model for Congolese legal texts:  

1. **Handling Multiple-Reference Documents**  
   Some legal documents reference multiple prior texts, such as amendments or repeals of previous laws. In such cases, the document title itself often contains multiple references (e.g., *"Loi n° 2025-001 modifiant et complétant la Loi n° 2020-1234"*). Our future work will focus on refining the model’s ability to accurately extract and link multiple references within a single document, allowing for better traceability and interconnection of legal texts.  

2. **Real-Time Automatic Training with Human Feedback**  
   To improve adaptability and robustness, we plan to integrate the model into a **continuous learning pipeline** where it is automatically updated with newly annotated legal texts. By incorporating **human-in-the-loop feedback**, we aim to ensure the model evolves dynamically while maintaining high accuracy. This iterative training approach will help strengthen our methodology, making the system more efficient and responsive to real-world legal document variations.  

By addressing these aspects, we aim to create a more **comprehensive, scalable, and adaptive** system for legal document analysis, ultimately contributing to a richer, structured legal database for the DRC.

## Conclusion
This work presents a structured approach to extracting key legal entities from Congolese legal texts using a Named Entity Recognition (NER) model. By leveraging **LLM-powered annotation**, **CNN-based training**, and a **custom dataset** of legal documents, we have demonstrated the feasibility of automating the identification of document **types**, **reference numbers**, and **publication dates**. Our results highlight the model’s strong performance across all entity categories, showcasing its potential to enhance the structuring and interlinking of legal texts in the Democratic Republic of Congo (DRC).  

Beyond its technical success, this work contributes to the broader goal of improving **legal research, citation practices, and document traceability** within the DRC's legal system. By automating entity extraction, we take a significant step toward building a **comprehensive and interconnected legal database**, facilitating more efficient legal analysis.  

Looking ahead, integrating **multiple-reference document handling** and **real-time adaptive training** with human feedback will further refine the model’s capabilities. These advancements will ensure that the system remains **scalable, dynamic, and responsive** to new legal texts, strengthening its applicability in real-world legal research and documentation.  

Ultimately, this work lays the foundation for a **more structured, accessible, and efficient legal information system**, empowering legal professionals, researchers, and policymakers in the DRC.

**Usage**
```bash
git clone https://github.com/bernard-ng/drc-legal-ner.git
cd drc-legal-ner
```
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
