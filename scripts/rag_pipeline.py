from langchain.chains import RetrievalQA
from langchain_core.output_parsers import StrOutputParser
from .utils import get_vector_store_by_filename, get_vector_store_by_topic

## Create a class called RAGPipeline, that can take an LLM, prompt template/instructions, 
## question and topic classifiers, vector stores, and answer user questions
class RAGPipeline:

    def __init__(self, llm, pdf_store, nonpdf_store, instruction_templates,
                 prompt_template, question_classifier_chain, topic_classifier_chain):
        
        self.llm = llm
        self.pdf_store = pdf_store
        self.nonpdf_store = nonpdf_store
        self.instructions = instruction_templates
        self.prompt_template = prompt_template
        self.question_classifier = question_classifier_chain
        self.topic_classifier = topic_classifier_chain

    def answer_question(self, query, filename=None, topic=None, question_type=None, k=3):
        """
        Core method to answer questions using:
          - question type classification
          - topic classification
          - vector store retrieval
          - answer generation via LLM + RetrievalQA chain
        """
        # Have the LLM infer the question type if not explicitally provided by the user
        # Options are 'Factual' or 'Interpretive'
        if not question_type:
            try:
                question_type = self.question_classifier.invoke({"query": query}).strip()
                print(f"Question type detected: {question_type}")
            except Exception as e:
                print(f"Question type classification error: {e}, defaulting to Interpretive.")
                question_type = "Interpretive"

        # Have the LLM infer the query topic if neither a specific filename nor topic is provided
        # by the user
        if not filename and not topic:
            try:
                topic = self.topic_classifier.invoke({"query": query}).strip()
                print(f"Topic detected: {topic}")
            except Exception as e:
                print(f"Topic classification error: {e}, skipping topic filtering.")
                topic = None

        # Early exit if topic is 'Other', as none of the documents used to build the vector stores fall
        # under this category
        if topic and topic.lower() == "other":
            return {"result": "Your question falls under the topic category 'Other', which is outside of the covered document topics. \
                               Try asking a question whose topic falls under one of ['Technology', 'Science', 'People', 'Literature']."}

        # Choose vector store based on filename or topic, assuming at least 1 of these is explicatlly provide by the user.
        # If both are provided by the user, the specified filename extension will take precedence over the selected topic.
        # If neither are provided, the inferred query topic from earlier is used instead and get_vector_store_by_topic() is used.
        if filename:
            vector_store = get_vector_store_by_filename(filename, self.pdf_store, self.nonpdf_store)
            filter_by = {"source": filename}
        else:
            vector_store = get_vector_store_by_topic(topic, self.pdf_store, self.nonpdf_store)
            filter_by = {"topic": topic} if topic else None

        # Retrieve documents based on cosine distance (here cosine distance = 1 - cosine similarity)
        # Since cosine similarity ranges between -1 and 1, cosine distance can range between 0 and 2.
        try:
            scored_docs = vector_store.similarity_search_with_score(query, k=k, filter=filter_by if filter_by else None)
        except Exception as e:
            print(f"Error during retrieval: {e}")
            return {"result": "There was an error retrieving documents."}

        # Filter out irrelevant results using a distance threshold of 0.5. This was selected as a good middle ground to filter out 
        # obviously irrelevant chunks but still be inclusive enough to retain somewhat relevant chunks.
        similarity_threshold = 0.5
        docs = [doc for doc, score in scored_docs if score <= similarity_threshold]

        # Check if any relevant chunks were retrieved. If none were, return a message to the user.
        if not docs:
            return {
                "result": "Sorry, I couldn't find any documents related to your question. "
                        "Please try asking something else or check the document collection "
                        "to see which topics are likely covered."
    }

        # Prepare the prompt using the 'instructions' template from config.py
        instruction_text = self.instructions.get(question_type, self.instructions["Interpretive"])
        prompt = self.prompt_template.partial(instruction=instruction_text)

        # Build the RetrievalQA chain and manually inject documents
        qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=None,
            return_source_documents=False,
            chain_type_kwargs={"prompt": prompt}
        )

        # Run the QA chain and return the result
        try:
            # Manually pass the filtered documents to the LLM chain and get an answer
            answer = qa_chain.combine_documents_chain.run(docs=docs, question=query)
            return {"result": answer}
        except Exception as e:
            print(f"Error during QA chain invoke: {e}")
            return {"result": "Sorry, I was unable to process your request."}

