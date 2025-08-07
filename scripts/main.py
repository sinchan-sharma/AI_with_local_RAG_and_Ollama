## Import all necessary packages
from .config import (
    DOCUMENTS_FOLDER, VECTOR_DB_DIR, CHUNK_SIZE, CHUNK_OVERLAP,
    google_embeddings, hf_embeddings,
    question_classifier_prompt, topic_classifier_prompt,
    base_prompt_template, instructions, get_llm
)
from .preprocessing import DocumentPreprocessor
from .rag_pipeline import RAGPipeline
from langchain_core.output_parsers import StrOutputParser


## Function to collect differnt inputs form the user interactively in the terminal
def collect_user_inputs():
    print("\nEnter your query and optional filters. Type 'exit' or 'quit' to quit.")
    query = input("Question: ").strip()
    if query.lower() in {"exit", "quit"}:
        return None

    ## User options to pass in a specific file, topic, or question type (can all be skipped by pressing 'enter' when prompted)
    filename = input("Restrict to specific filename? (Leave blank to search all): ").strip() or None
    topic = input("Specify a topic? (Leave blank to infer): ").strip() or None
    question_type = input("Specify question type (Factual / Interpretive)? (Leave blank to auto-detect): ").strip() or None

    return {
        "query": query,
        "filename": filename,
        "topic": topic,
        "question_type": question_type
    }

## Main pipeline
def main():
    print("Initializing model, vector stores, and document processing pipeline...")
    # Set up a DocumentPreprocessor class instance
    preprocessor = DocumentPreprocessor(
        folder_path=DOCUMENTS_FOLDER,
        persist_dir=VECTOR_DB_DIR,
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        google_embeddings=google_embeddings,
        hf_embeddings=hf_embeddings
    )

    while True:
        # Ask user whether to rebuild vector store from scratch or load an existing one
        user_input = input("Would you like to force rebuilding the vector stores? (type only 'y' or 'n'): ").strip().lower()
        if user_input in ("y", "n"):
            break
        print("Please enter 'y' or 'n'.")

    # Check whether user_input == "y", determining whether force_rebuild should be set to True or False
    force_rebuild = (user_input == "y")

    # Either load existing vector stores or build them from scratch
    vector_store_pdf, vector_store_non_pdf = preprocessor.load_or_build_vectorstores(force_rebuild=force_rebuild)

    # Initialize the LLM
    llm = get_llm()

    # Create different chains for question type and topic detection, using StrOutputParser to parse simple strings
    question_classifier_chain = question_classifier_prompt | llm | StrOutputParser()
    topic_classifier_chain = topic_classifier_prompt | llm | StrOutputParser()

    # Set up an instance of the RAGPipeline class
    rag = RAGPipeline(
        llm=llm,
        pdf_store=vector_store_pdf,
        nonpdf_store=vector_store_non_pdf,
        instruction_templates=instructions,
        prompt_template=base_prompt_template,
        question_classifier_chain=question_classifier_chain,
        topic_classifier_chain=topic_classifier_chain
    )

    print("Ready! Ask your questions (type 'exit' to quit):")

    # Let the user keep asking questions as long as they don't exit the application 
    while True:
        user_input = collect_user_inputs()
        if not user_input:
            print("Goodbye!")
            break

        # Use the LLM with RAG to answer a user question
        result = rag.answer_question(**user_input)
        print("\nAnswer:\n", result["result"])
        print("-" * 100)

## Run main pipeline above
if __name__ == "__main__":
    main()

