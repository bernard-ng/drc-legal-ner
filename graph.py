import argparse

import pandas as pd
import spacy
from tqdm import tqdm
from neo4j import GraphDatabase

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


def setup_database(driver: GraphDatabase.driver) -> None:
    with driver.session() as session:
        session.run("CREATE CONSTRAINT unique_document_title IF NOT EXISTS FOR (d:Document) REQUIRE d.title IS UNIQUE")
        print("🟢 Database setup completed.")


def clean_database(driver: GraphDatabase.driver) -> None:
    with driver.session() as session:
        session.run("MATCH (n) SET n = {}")
        session.run("MATCH (n) DETACH DELETE n")
        session.run("MATCH ()-[r]->() DELETE r")
        print("🟢 Database cleared.")


def create_document_nodes(driver: GraphDatabase.driver, df_predictions: pd.DataFrame) -> None:
    with driver.session() as session:
        for _, row in tqdm(df_predictions.iterrows(), desc="Inserting nodes into Neo4j", total=len(df_predictions)):
            query = (
                "MERGE (d:Document {title: $title}) "
                "SET d.published_at = $published_at, d.types = $types, d.references = $references, d.dates = $dates"
            )
            session.run(query, title=row["title"], published_at=row["published_at"], types=row["types"], references=row["references"], dates=row["dates"])


    with driver.session() as session:
        for _, row in tqdm(df_predictions.iterrows(), desc="Creating citations", total=len(df_predictions)):
            citing_title = row["title"]
            citing_references = set(str(row["references"]).split(", ")) if pd.notna(row["references"]) else set()
            citing_types = set(str(row["types"]).split(", ")) if pd.notna(row["types"]) else set()

            for _, candidate_row in df_predictions.iterrows():
                cited_title = candidate_row["title"]

                # Skip self-citations
                if citing_title == cited_title:
                    continue

                cited_references = set(str(candidate_row["references"]).split(", ")) if pd.notna(
                    candidate_row["references"]) else set()
                cited_types = set(str(candidate_row["types"]).split(", ")) if pd.notna(
                    candidate_row["types"]) else set()

                if citing_references & cited_references and citing_types & cited_types:
                    query = (
                        "MATCH (citing:Document {title: $citing_title}), (cited:Document {title: $cited_title}) "
                        "MERGE (citing)-[:CITES]->(cited)"
                    )
                    session.run(query, citing_title=citing_title, cited_title=cited_title)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Import packages into Neo4j")
    parser.add_argument("--model", type=str, default="efficiency", help="Model name")
    parser.add_argument("--skip-clear", action="store_true", help="Skip clearing the database")
    parser.add_argument("--skip-predictions", action="store_true", help="Skip predictions")
    args = parser.parse_args()

    neo4j_driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "password"))

    if not args.skip_clear:
        clean_database(neo4j_driver)
        setup_database(neo4j_driver)

    if not args.skip_predictions:
        nlp = spacy.load(f"./models/{args.model}/model-best")
        df = pd.read_csv("./dataset/data.csv")
        df.rename(columns={"date": "published_at"}, inplace=True)
        df[["types", "references", "dates"]] = df["title"].apply(lambda x: pd.Series(extract_entities(str(x))))
        df.to_csv("./dataset/predictions.csv", index=False)


    create_document_nodes(neo4j_driver, df := pd.read_csv("./dataset/predictions.csv"))
    neo4j_driver.close()
    print("🟢 Documents importation completed !")
