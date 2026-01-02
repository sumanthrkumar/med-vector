# Medical Question Answering System with Gemini and PubMed

This project implements a Retrieval-Augmented Generation (RAG) system designed to answer medical questions using information sourced from PubMed abstracts and processed by Google's Gemini models. It leverages ChromaDB as a vector store to manage semantic search and retrieval of relevant documents, enhancing the accuracy and context-awareness of the generated responses.

The system operates in two main phases: first, populating a local knowledge base with medical abstracts, and second, querying this knowledge base to generate cited answers to medical questions.

## Features

*   **Retrieval-Augmented Generation (RAG) Architecture**: Combines an external knowledge base (PubMed abstracts) with a large language model (Google Gemini) for informed question answering.
*   **PubMed Data Ingestion**: Automatically fetches medical abstracts from the PubMed database using the Entrez API.
*   **Semantic Vector Storage**: Utilizes ChromaDB to store embedded medical abstracts, enabling efficient semantic search and retrieval of relevant documents.
*   **Google Gemini Integration**:
    *   **Embedding**: Employs the `text-embedding-004` model for creating high-quality embeddings for both documents and queries.
    *   **Generation**: Uses the `gemini-2.0-flash-lite-preview-02-05` model to generate coherent and contextually relevant answers.
*   **Context-Aware Prompt Generation**: Dynamically constructs prompts for the Gemini model, including retrieved documents as context, ensuring answers are grounded in the provided information.
*   **Source Citation**: Generated answers automatically cite the specific medical abstracts (by title) that support each factual statement, promoting transparency and verifiability.
*   **Rate Limit Handling**: Includes retry mechanisms with timeouts for both embedding and generation API calls to gracefully handle potential rate limiting.
*   **Batch Query Processing**: Processes a predefined list of medical questions and compiles their answers and sources into a structured CSV output.

## Installation

To set up and run this project, follow these steps:

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/your-username/your-repo-name.git
    cd your-repo-name
    ```

2.  **Create a virtual environment** (recommended):
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install dependencies**:
    This project requires `chromadb`, `google-generativeai`, `python-dotenv`, `pandas`, and `biopython`.
    ```bash
    pip install chromadb google-generativeai python-dotenv pandas biopython
    ```

4.  **Set up Google Gemini API Key**:
    *   Obtain a Google Gemini API key from the Google AI Studio or Google Cloud Console.
    *   Create a `.env` file in the root directory of the project.
    *   Add your API key to the `.env` file in the following format:
        ```
        GEMINI_API_KEY="YOUR_API_KEY_HERE"
        ```

5.  **Configure Entrez Email**:
    *   The `Bio.Entrez` module requires an email address for API usage. This is set in `retrieve_documents.py`. You can update `Entrez.email = "placeholder@gmail.com"` to your actual email address.

## Usage

The project workflow involves two main scripts: first, populating the knowledge base, and then, querying it.

### Step 1: Populate the Knowledge Base

Run the `retrieve_documents.py` script to fetch medical abstracts from PubMed, embed them, and store them in a local ChromaDB instance. This script will create a `./med_db` directory to store the vector database.

```bash
python retrieve_documents.py
```

This script will:
*   Search PubMed for abstracts related to predefined medical topics (e.g., "Caffeine Sleep", "Alzheimer's disease", "Brain Cancer", etc.).
*   Generate embeddings for each abstract using Google's `text-embedding-004` model.
*   Store these embeddings, along with the original documents and their titles, in a ChromaDB collection named "medical_abstracts".
*   You will see print statements indicating the progress of fetching and adding documents.

### Step 2: Query the System

After the knowledge base is populated, run the `query_data.py` script to ask questions and generate answers using the RAG system.

```bash
python query_data.py
```

This script will:
*   Take a predefined list of medical questions.
*   For each question, it will generate an embedding.
*   Perform a semantic search in the "medical_abstracts" ChromaDB collection to retrieve the most relevant documents.
*   Construct a prompt incorporating the question and the retrieved documents as context.
*   Send the prompt to the Google Gemini model (`gemini-2.0-flash-lite-preview-02-05`) to generate an answer.
*   Include source citations in the generated answers based on the retrieved document titles.
*   Save all questions, generated answers, and their corresponding sources to an `output_file.csv` in the project root directory.
*   The script includes rate-limiting handling and pauses between API calls to ensure successful execution.