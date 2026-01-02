import chromadb
import google.generativeai as genai
import os
from dotenv import load_dotenv
import pandas
import time

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key = GEMINI_API_KEY)
model = genai.GenerativeModel('gemini-2.0-flash')

client = chromadb.PersistentClient(path="./med_db")
collection = client.get_collection(name="medical_abstracts")

MAX_RETRIES = 3
TIMEOUT = 10

# Embed query/question in same format that documents were embedded
def get_embeddings(text):
    for attempt in range(MAX_RETRIES):
        try:
            result = genai.embed_content(
                model = "models/text-embedding-004",
                content = text,
                task_type = "retrieval_query"
            )

            return result['embedding']
        except Exception as e:
            print(f"Embedding failed due to rate limits. Retrying in {TIMEOUT} seconds ({e})")
            time.sleep(TIMEOUT)

    return None

# Prompt to sent to Gemini
def generate_prompt(query, results):
    documents = results['documents'][0]
    metadatas = results['metadatas'][0]

    context_text = ""

    for i, (doc, metadata) in enumerate(zip(documents, metadatas)):
        source = f"Source {i+1} (Title: {metadata['title']})"
        context_text += f"{source}: \n{doc}\n\n"

    prompt = f"""
        You are a distinguished and helpful medical assistant. 
        
        Your task is to answer the question using only the context provided. 

        Every time you make a factual statement, you must cite the source ID at the end of the sentence. 
        Example: "Caffeine increases heart rate [Source 1]."

        If the answer is not part of the context, respond with "The context does not provide information about the question."

        question = {query}

        context to use = {context_text}
    """

    return prompt

query_list = [
    "How does caffeine consumption affect sleep latency?",
    "What is the impact of caffeine on REM sleep cycles?",
    "Does caffeine act as an adenosine receptor antagonist?",
    "How long before bed should caffeine be avoided to prevent insomnia?",
    "What is the role of amyloid plaques in Alzheimer's disease?",
    "Are there genetic risk factors associated with Alzheimer's?",
    "What are the current FDA-approved treatments for Alzheimer's?",
    "How does tau protein accumulation relate to cognitive decline?",
    "What are the standard treatment protocols for Glioblastoma?",
    "Does chemotherapy effectively cross the blood-brain barrier?",
    "What are the survival rates for varying grades of brain tumors?",
    "Are there immunotherapy options for brain cancer patients?",
    "What are the primary differences between viral and bacterial pneumonia symptoms?",
    "Which antibiotics are commonly prescribed for community-acquired pneumonia?",
    "Is the pneumococcal vaccine effective for elderly patients?",
    "What are the common complications of untreated pneumonia?",
    "What is the efficacy of Exposure and Response Prevention (ERP) therapy?",
    "Are SSRIs considered a first-line treatment for OCD?",
    "How does OCD presentation differ in children versus adults?",
    "Is Deep Brain Stimulation (DBS) used for treatment-resistant OCD?"
]

output_data = []

# Hit Gemini for each question in query_list, save final results to a csv file
for query in query_list:
    query_vector = get_embeddings(query)
    if query_vector is None:
        print("Skipping due to error while embedding query")
        continue

    results = collection.query(
        query_embeddings = [query_vector],
        n_results = 3,

        include=["documents", "metadatas"]
    )

    prompt = generate_prompt(query, results)
    response_text = "Error" #By default it will be error. Will be overriten with acutal answer

    for attempt in range(MAX_RETRIES):
        try:
            response = model.generate_content(prompt)
            response_text = response.text
            break
        except Exception as e:
            print(f"Hit rate limit while getting answer. Will wait {TIMEOUT} seconds")
            time.sleep(TIMEOUT)

    metadatas = results['metadatas'][0]

    sourcesStr = ""
    for i, meta in enumerate(metadatas):
        sourcesStr += f"[Source {i+1}]: {meta['title']}\n"

    output_data.append({
        'question': query,
        'answer': response_text,
        'sources': sourcesStr
    })    

    time.sleep(5)


df = pandas.DataFrame(output_data)

# Write to a CSV file in the current working directory
df.to_csv('output_file.csv', index=False)