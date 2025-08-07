import os
from dotenv import load_dotenv

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_ollama import OllamaLLM

## Load environment variables
load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

## LangSmith UI tracing (these are optional)
LANGSMITH_TRACING = os.getenv("LANGSMITH_TRACING")
LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY")
LANGSMITH_PROJECT = os.getenv("LANGSMITH_PROJECT")
LANGSMITH_ENDPOINT = os.getenv("LANGSMITH_ENDPOINT")

if not GOOGLE_API_KEY:
    raise ValueError("Please set the GOOGLE_API_KEY in your .env file")

## Constants
DOCUMENTS_FOLDER = "./documents"
VECTOR_DB_DIR = "./chroma_db"
CHUNK_SIZE = 600
CHUNK_OVERLAP = 100

## Embedding models used
google_embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY)
hf_embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

## Prompt templates for question type classification, question topic classification,
## as well as general instructions and a base template
instructions = {
    "Factual": """
    Your task is to answer the question based ONLY on the provided context. 
    Extract specific, accurate information directly from the text. 
    If the answer is not found, clearly state that the information is unavailable. 
    Provide a complete and well-formed answer in full sentences.
    """,
    "Interpretive": """
    Your task is to provide a thoughtful interpretive answer by synthesizing the information in the provided context. 
    Draw meaningful connections, explain implications, and summarize broader themes or significance. 
    Make your answer clear, coherent, and complete, even if the information is scattered.
    """
}

base_prompt_template = PromptTemplate(
    input_variables=["instruction", "context", "question"],
    template="""
    \"\"\"
    You are an intelligent document assistant.
    {instruction}

    Context:
    {context}

    Question: 
    {question}

    Answer in complete sentences, using ONLY the information given in the context. 
    If the context does not contain the answer, say so clearly.

    Answer:
    \"\"\"
    """
)

question_classifier_prompt = PromptTemplate(
    input_variables=["query"],
    template="""
You are a classifier that determines whether a user question is Factual or Interpretive.

- A *Factual* question asks for specific information found directly in a document (e.g., names, dates, techniques, facts).
- An *Interpretive* question asks for broader meaning, implications, or synthesis (e.g., summaries, themes, significance).

Respond with only the single word: Factual or Interpretive.

Examples:

Question: What year was Nikola Tesla born?
Classification: Factual

Question: What is this paper mainly about?
Classification: Interpretive

Question: Who are the key figures mentioned in this research?
Classification: Factual

Question: How does this paper relate to broader trends in machine learning?
Classification: Interpretive

Now classify the following question:
Question: {query}
Classification:
"""
)

topic_classifier_prompt = PromptTemplate(
    input_variables=["query"],
    template="""
You are a classifier that assigns a topic label to a question. 
Choose the single most relevant topic from this exact list:

- Technology
- People
- Science
- Literature
- Other

Respond with ONLY the topic name (one of the above) and nothing else.

Examples:

Question: What are common techniques used in machine learning?
Topic: Technology

Question: What was Alan Turing's contribution to computer science?
Topic: People

Question: What are some of the impacts of climate change?
Topic: Science

Question: What are some popular books published after 2000?
Topic: Literature

Question: What is a good cheese to pair with red wine?
Topic: Other

Now classify this question:
Question: {query}
Topic:
"""
)

## Lazy loading of a singleton Ollama model
_ollama_model = None # Module-level private variable to hold the model instance

def get_llm():
    global _ollama_model
    if _ollama_model is None:
        print("Loading Gemma3 Ollama model...")
        _ollama_model = OllamaLLM(model="gemma3") # Instantiate the model only once
    return _ollama_model

## Now chains using the LLM will be created dynamically in rag_pipeline.py
