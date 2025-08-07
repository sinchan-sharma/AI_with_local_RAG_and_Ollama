import os

## Function to assign each file type to a particular topic, based on the documents used
def assign_topic(path: str) -> str:
    """
    Function to assign each file type to a particular topic, based on the documents used
    """
    if path.endswith(".pdf"):
        return "Technology"
    elif path.endswith(".txt"):
        return "People"
    elif path.endswith(".html"):
        return "Science"
    elif path.endswith(".json"):
        return "Literature"
    return "Other"

def get_vector_store_by_filename(filename, pdf_store, non_pdf_store):
    """
    Returns the appropriate vector store for chunk retrieval assuming the user
    explicitally provides a filename.
    """
    ext = os.path.splitext(filename)[1].lower()
    if ext == ".pdf":
        return pdf_store        # Note: Only PDF files are stored in the vector store using Google embeddings.
    else:                       # All other file types are stored in a separate vector store using Hugging
        return non_pdf_store    # Face embeddings for simplicity.

def get_vector_store_by_topic(topic, pdf_store, non_pdf_store):
    """
    Returns the appropriate vector store for chunk retrieval assuming the user
    didn't directly provide a filename, either through a topic being directly 
    provided by the user or inferred by the LLM.
    """
    topic = topic.lower()
    if topic == "technology":   # Note: all PDF files are designated as "Technology" as far as content topic is concerned.
        return pdf_store        # This is for simplicity when using multiple embedding models and storing them in different
    else:                       # vector stores. All other file types fall under other categories besides "Technology".
        return non_pdf_store

