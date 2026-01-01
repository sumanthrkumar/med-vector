import google.generativeai as genai
from Bio import Entrez
import chromadb
import os
from dotenv import load_dotenv

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key = GEMINI_API_KEY)

Entrez.email = "placeholder@gmail.com"

def fetch_pubmed_abstracts(term, max_results = 3):
    print(f"Searching pubmed for term {term}")

    # Search for doc Ids based on search term
    handle = Entrez.esearch(db="pubmed", term=term, retmax=max_results)
    record = Entrez.read(handle)
    handle.close()
    id_list = record["IdList"]

    # Fetch details for those IDs
    print(f"Found {len(id_list)} articles. Fetching abstracts")
    handle = Entrez.efetch(db="pubmed", id=id_list, rettype="xml", retmode="text")
    papers = Entrez.read(handle)
    handle.close()

    abstracts = []
    for paper in papers['PubmedArticle']:
        try:

            id = paper['MedlineCitation']['PMID']
            article = paper['MedlineCitation']['Article']
            title = article['ArticleTitle']
            abstract_text = article['Abstract']['AbstractText'][0]

            abstracts.append({
                "title": title,
                "text": f"Title: {title}\nabstract: {abstract_text}",
                "pubmed_id": str(id)
            })

        except KeyError:
            continue

    return abstracts


# Embed abstracts
def create_embedding(text):
    result = genai.embed_content(
        model = "models/text-embedding-004",
        content = text,
        task_type = "retrieval_document"
    )

    return result['embedding']


if __name__ == "__main__":
    client = chromadb.PersistentClient(path="./med_db")
    collection = client.get_or_create_collection(name="medical_abstracts")

    topics = ["Caffeine Sleep", "Alzheimer's disease", "Brain Cancer", "pneumonia", "Obsessive Compulsive Disorder"]

    for topic in topics:
        # Fetch data
        results = fetch_pubmed_abstracts(topic)

        # Prepare data from chromadb
        ids = []
        documents = []
        embeddings = []
        metadatas = []

        for res in results:
            vector = create_embedding(res['text'])

            ids.append(str(res['pubmed_id']))
            documents.append(res['text'])
            embeddings.append(vector)
            metadatas.append({"title": res['title']})

        if ids:
            print(f"Adding {len(ids)} documents to ChromaDB")
            collection.upsert(
                ids=ids,
                documents=documents,
                embeddings=embeddings,
                metadatas=metadatas
            )
            print("Data saved to ./med_db folder")
        else:
            print("No documents found to add.")