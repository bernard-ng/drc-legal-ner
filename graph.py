import pandas as pd
import spacy
from tqdm import tqdm
from neo4j import GraphDatabase

nlp = spacy.load("./models/leganews-20250403015946")
df = pd.read_csv("./dataset/data.csv")
df.rename(columns={"date": "published_at"}, inplace=True)

def extract_entities(text):
    doc = nlp(text)
    types, references, dates = [], [], []

    for ent in doc.ents:
        if ent.label_ == "TYPE":
            types.append(ent.text)
        elif ent.label_ == "REFERENCE":
            references.append(ent.text)
        elif ent.label_ == "DATE":
            dates.append(ent.text)

    return ", ".join(types), ", ".join(references), ", ".join(dates)


# Apply NER model
df[["types", "references", "dates"]] = df["title"].apply(lambda x: pd.Series(extract_entities(str(x))))
df.to_csv("./dataset/predictions.csv", index=False)

# Neo4j connection
driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "password"))


def create_graph(tx, title, published_at, types, references, dates):
    query = (
        "MERGE (d:Document {title: $title}) "
        "SET d.published_at = $published_at, d.types = $types, d.references = $references, d.dates = $dates"
    )
    tx.run(query, title=title, published_at=published_at, types=types, references=references, dates=dates)


def create_citation(tx, citing_title, cited_title):
    query = (
        "MATCH (citing:Document {title: $citing_title}), (cited:Document {title: $cited_title}) "
        "MERGE (citing)-[:CITES]->(cited)"
    )
    tx.run(query, citing_title=citing_title, cited_title=cited_title)


# Create citations with improved logic
def create_citations(tx, citing_title, cited_title):
    query = (
        "MATCH (citing:Document {title: $citing_title}), (cited:Document {title: $cited_title}) "
        "MERGE (citing)-[:CITES]->(cited)"
    )
    tx.run(query, citing_title=citing_title, cited_title=cited_title)


# Insert nodes
def populate_neo4j():
    with driver.session() as session:
        for _, row in tqdm(df.iterrows(), desc="Inserting nodes into Neo4j", total=len(df)):
            session.execute_write(create_graph, row["title"], row["published_at"], row["types"], row["references"],
                                      row["dates"])


    with driver.session() as session:
        for _, row in tqdm(df.iterrows(), desc="Creating citations", total=len(df)):
            citing_title = row["title"]
            citing_references = set(row["references"].split(", ")) if row["references"] else set()
            citing_types = set(row["types"].split(", ")) if row["types"] else set()

            for _, candidate_row in df.iterrows():
                cited_title = candidate_row["title"]
                cited_references = set(candidate_row["references"].split(", ")) if candidate_row["references"] else set()
                cited_types = set(candidate_row["types"].split(", ")) if candidate_row["types"] else set()

                # Ensure a valid citation match
                if citing_references & cited_references and citing_types & cited_types:
                    session.execute_write(create_citations, citing_title, cited_title)

populate_neo4j()
driver.close()
print("Neo4j database updated successfully!")
