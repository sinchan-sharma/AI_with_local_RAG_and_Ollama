# **Capstone Project: "Intelligent Document Assistant (IDA) with Local RAG & Observability"**

**Project Goal:** Develop a robust, local-first Intelligent Document Assistant that can answer complex questions and summarize content from a collection of diverse documents, utilizing LangChain for RAG and Ollama for the LLM, with integrated LangSmith tracing.

## Hybrid RAG System with LangChain

This project implements a modular `Retrieval-Augmented Generation (RAG)` pipeline using `LangChain` to answer questions based on a local document collection. It supports hybrid embeddings, intelligent classification, and interactive CLI querying. The LLM chosen for this task was via `Ollama`, specifically the `gemma3:latest` model. This model has about 3 billion parameters and takes about 3-5 GB of space when downloading.

---

## Features

- **Hybrid Embeddings**:  
  - PDF files --> embedded using an API-based embeddings from Google.
  - Non-PDF files --> embedded using local HuggingFace models.

- **Chunking & Preprocessing**:  
  - Files are split into context-aware chunks (configurable size & overlap).
  - HTML, PDF, TXT, and JSON file formats were used for building the vector stores.

- **Dual Classifiers**:  
  - **Question Type Classifier** --> *Factual* vs *Interpretive*
  - **Topic Classifier** --> Assigns user queries to certain document topics (for this project each document used fell into one of four categories: `*Technology*, *Science*, *People*, and *Literature*`).

- **Smart Retrieval**:  
  - Vector store lookup by topic category or specific filename.
  - Filters irrelevant chunks using a similarity threshold 
    - `Cosine distance` specifically is used, which equals `1 - cosine similarity`
  - Supports fallback messaging when no relevant chunks/results are found. 

- **Prompt Customization**:  
  - Injects instructions (per question type) into the base prompt template.

- **Interactive CLI**:  
  - Ask questions, specify file name/topic/question type, or let the system infer the query topic and question type if not provided by the user.

---

## Folder Structure

This project includes the following:

### `documents/`

Contains a list of all of the files used for chunking, embedding, and populating the ChromaDB vector store. There are 7 documents in all: 3 PDFs, 2 plain text files, 1 JSON file, and 1 HTML file.

**Important Note**: When building vector stores using ChromaDB, because two separate embedding models were used, 2 separte vector stores were created within the same ChromaDB. The first vector store contains only embeddings from PDF files, while the other store contains embeddings from the remaining non-PDF files. Additionally, all of the PDF files specifically fall under the topic category "Technology", while all the other documents fall under the other topic categories described above.

### `screenshots/`

Contains screenshots of LangSmith UI traces for LLM calls, including some successful as well as unsuccessful traces in terms of context retrieval and LLM output. Also contains some examples of the CLI interface in action.

### `scripts/`

Contains Python scripts to build and use the RAG system built for context retrieval for question-answering purposes. The overall logic is broken into several scripts for modularity, and includes files for setting configurations, file processing, utility functions, building the RAG system, and a main file for running the entire pipeline, from loading the Ollama LLM model and document chunking/embedding, to creating different prompt templates for the LLM to use, to buidling the RAG pipeline, to finally allowing the user to ask specific questions based on the collected data used and get answers.

**To run the overall pipeline:**

- Run `python -m scripts.main` in the terminal

**Disclaimer**: The local RAG system that was built, while effective with question answering tasks, is not really well-suited for summarizing large bodies of text or entire documents/files. As a result, trying to summarize an entire file or large body of text may yield unexpected or poor results. That being said, some summarization tasks were attempted in a separate Jupyter Notebook file, using document chunking to break up the text, summarizing each chunk to get a partial summary, and then having the LLM use all of the smaller summaries to generate an overall text/document summary. 

### `notebooks/`

Contains a Jupyter Notebook used during the course of the project for experimentation purposes and summarization tasks, as well as an additional notebook used for web scraping purposes to get additional documents for the vector store. That said, the documents used are already in a folder, so no need to run the `web_scraping.ipynb` notebook again.

### `langsmith_trace_evaluation.md`

- A Markdown file containing an evaluation of different LangSmith UI trace screenshots, seeing how they performed and explaning debugging steps to improve the quality of the traces (whether it's better context retrieval, LLM response ouptut, etc.)

### `requirements.txt`

Lists all required Python packages. Install with `pip install -r requirements.txt`.

---

## Configuration

When running this code on your own device, `config.py` can be modified as follows:

- Set paths for `DOCUMENTS_FOLDER`, `VECTOR_DB_DIR`, etc.
- Set API keys and other environment variables.
- Configure `chunk_size`, `chunk_overlap`.
    - For this project, I specifically chose a chunk size of 600 and an overlap of 100. This is due to the fact that trying larger chunk sizes consistently resulted in errors due to lack of available RAM when loading the Ollama model into memory and trying to run the LLM. As for the overlap, I felt that it allowed for enough context to be preserved between chunks, without there being too much repitition.
- Choose different embedding models, such as Google API-based or Hugging Face embeddings.
- Define instruction templates and base prompt.

---

## Running the Main Python Script:

Once you run `python -m scripts.main` from the project root directory, the following will occur:


You will be prompted:

- Whether to rebuild vector stores from scratch 
    - If vector stores don't already exist, they will be built regardless of how you answer.
- To enter a question, and optionally:
  - Restrict by filename (type in the name of a specific file in the `documents` folder)
  - Specify a topic you query falls under (the documents used specifically cover *technology*, *science*, *people*, and *literature*)
  - Specify a question type (*factual* or *interpretive*)

### Example:

```
Question: What is supervised learning?
Restrict to specific filename? (Leave blank to search all): 
Specify a topic? (Leave blank to infer): 
Specify question type (Factual / Interpretive)? (Leave blank to auto-detect): 
```

### Sample Output

```
Answer:
Supervised learning is a type of machine learning where the model is trained on labeled data...
```

If no relevant results are found, you may see something like this:

```
Sorry, I couldn't find any documents related to your question. 
Please try asking something else or check the document collection to see which topics are likely covered.
```

To see specific examples of the CLI interface, refer to the `screenshots/` folder.

## Additional Notes:

The links to some of the documents used for building the vector databases can be found below:

https://www.climaterealityproject.org/blog/congress-living-fantasy-world-budget-bill

https://files.eric.ed.gov/fulltext/EJ1117604.pdf

https://files.eric.ed.gov/fulltext/EJ1172284.pdf

https://www.tutorialspoint.com/machine_learning/machine_learning_tutorial.pdf

https://en.wikipedia.org/wiki/Alan_Turing

https://en.wikipedia.org/wiki/Nikola_Tesla

Note: The books.JSON file was manually generated for testing purposes, not downloaded from any website.
