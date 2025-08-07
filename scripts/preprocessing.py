import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader, TextLoader, JSONLoader, BSHTMLLoader
from .config import DOCUMENTS_FOLDER, VECTOR_DB_DIR, CHUNK_SIZE, CHUNK_OVERLAP, google_embeddings, hf_embeddings
from .utils import assign_topic

## Create a DocumentLoader class for processing documents/files
class DocumentPreprocessor:

    def __init__(self, folder_path=DOCUMENTS_FOLDER, persist_dir=VECTOR_DB_DIR,
                 chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP,
                 google_embeddings=google_embeddings, hf_embeddings=hf_embeddings):
        
        self.folder_path = folder_path
        self.persist_dir = persist_dir
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.google_embeddings = google_embeddings
        self.hf_embeddings = hf_embeddings
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=self.chunk_size,
                                                            chunk_overlap=self.chunk_overlap)

    # Function to load/split documents using the appropriate loader for each file type
    def load_and_split_file(self, path):
        """
        Load and split documents with appropriate loader based on file type.
        Returns a list of Document chunks with metadata.
        """
        should_split = True  # JSON files are pre-chunked, all other file types
                             # are split/chunked using a RecursiveCharacterSplitter

        if path.endswith(".pdf"):
            loader = PyPDFLoader(path)
        elif path.endswith(".txt"):
            loader = TextLoader(path, encoding="utf-8")
        elif path.endswith(".html"):
            loader = BSHTMLLoader(path, open_encoding="utf-8")
        elif path.endswith(".json"):
        # Extract all book fields into a single formatted text using a JQ query expression.
        # This is so that specific data from a JSON file can be effectively parsed and extracted,
        # resulting in a clean, readable text representation that is already "chunked".
            jq_schema = (
                '.books[] | "Title: " + .title + '
                '"\\nAuthor: " + .author + '
                '"\\nGenre: " + .genre + '
                '"\\nPublication Year: " + (.publication_year | tostring) + '
                '"\\nDescription: " + .description + '
                '"\\nPositive Review: " + .reviews.positive + '
                '"\\nNegative Review: " + .reviews.negative'
            )
            loader = JSONLoader(
                file_path=path,
                jq_schema=jq_schema,
                text_content=False
            )
            should_split = False
        else:
            raise ValueError(f"Skipping unsupported file type: {path}")

        docs = loader.load()
        # Assign a topic label for filtering or classification, based on filename
        topic = assign_topic(os.path.basename(path))

        # Add specific document metadata related to source and content topic
        for doc in docs:
            doc.metadata["source"] = os.path.basename(path)
            doc.metadata["topic"] = topic

        # Split non-JSON files using a RecursiveCharacterSplitter
        if should_split:
            return self.text_splitter.split_documents(docs)
        else:
            return docs

    def load_or_build_vectorstores(self, force_rebuild=False):
        """
        Load all documents from folder, split, chunk, and build two vector stores within a Chroma DB:
        - PDF documents use Google embeddings (API-based)
        - Non-PDF documents use Hugging Face embeddings

        If vector stores already exist and force_rebuild is False, load from disk instead.
        Otherwise, new vector stores will be built from scratch

        Returns tuple: (vector_store_pdf, vector_store_non_pdf)
        """

        # Set up paths to save both created vector stores to
        pdf_dir = os.path.join(self.persist_dir, "pdf")
        nonpdf_dir = os.path.join(self.persist_dir, "nonpdf")

        # Helper function to check if a Chroma DB directory already exists and is populated
        def vector_store_exists(path):
            """
            Helper function to check if a Chroma DB directory already exists and is populated
            """
            return os.path.exists(path) and os.path.isdir(path) and len(os.listdir(path)) > 0

        # Load from disk if vector stores exist and not forcing a rebuild
        if not force_rebuild and vector_store_exists(pdf_dir) and vector_store_exists(nonpdf_dir):
            print("Vector stores found on disk. Loading...")
            vector_store_pdf = Chroma(
                persist_directory=pdf_dir,
                collection_name="pdf-collection",
                embedding_function=self.google_embeddings
            )
            vector_store_non_pdf = Chroma(
                persist_directory=nonpdf_dir,
                collection_name="nonpdf-collection",
                embedding_function=self.hf_embeddings
            )
            return vector_store_pdf, vector_store_non_pdf

        # Otherwise, build vector stores again from scratch
        print("Rebuilding vector stores from scratch...")

        # Store chunks to be later stored in respective vector stores
        pdf_chunks = []
        non_pdf_chunks = []

        # Split documents into chunks according to the chunk size and overlap specified
        for filename in os.listdir(self.folder_path):
            path = os.path.join(self.folder_path, filename)
            chunks = self.load_and_split_file(path)

            # Add chunks to the appropriate list based on file extension
            if filename.lower().endswith(".pdf"):
                pdf_chunks.extend(chunks)
            else:
                non_pdf_chunks.extend(chunks)

        # Store and embed chunks from PDF files in one vector store
        vector_store_pdf = Chroma.from_documents(
            documents=pdf_chunks,
            embedding=self.google_embeddings,
            collection_name="pdf-collection",
            persist_directory=pdf_dir
        )

        # Store and embed chunks from NON-PDF files in another vector store
        vector_store_non_pdf = Chroma.from_documents(
            documents=non_pdf_chunks,
            embedding=self.hf_embeddings,
            collection_name="nonpdf-collection",
            persist_directory=nonpdf_dir
        )

        print("Vector stores were successfully built and saved to disk.")
        return vector_store_pdf, vector_store_non_pdf
