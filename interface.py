import gradio as gr
from document_processing import DocumentProcessor
from faiss_index import FaissIndex
from response_generation import ResponseGenerator
from transformers import AutoTokenizer, AutoModel, GPT2LMHeadModel, GPT2Tokenizer
import os
import numpy as np


def gradio_interface(query, faiss_index, documents, doc_tokenizer, doc_model, gpt_tokenizer, gpt_model):
    # Embed the query
    query_embedding = DocumentProcessor(data_directory).embed_query(doc_tokenizer, doc_model, query)

    # Retrieve the top documents
    top_indices = faiss_index.retrieve_top_documents(query_embedding)
    retrieved_docs = [documents[i] for i in top_indices[0]]

    answer = ResponseGenerator(gpt_tokenizer, gpt_model).generate_response(query, retrieved_docs)
    print("answer: ", answer)
    return retrieved_docs, answer


if __name__ == "__main__":
    data_directory = 'data'

    # Initialize document processor
    doc_processor = DocumentProcessor(data_directory)

    # Load or process documents
    if os.path.exists("document_texts.npy") and os.path.exists("document_labels.npy"):
        documents = np.load("document_texts.npy", allow_pickle=True).tolist()
        labels = np.load("document_labels.npy", allow_pickle=True).tolist()
    else:
        doc_processor.load_documents_and_labels()
        documents = doc_processor.documents
        labels = doc_processor.labels

    # Initialize tokenizers and models
    doc_tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/paraphrase-MiniLM-L6-v2")
    doc_model = AutoModel.from_pretrained("sentence-transformers/paraphrase-MiniLM-L6-v2")
    gpt_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    gpt_model = GPT2LMHeadModel.from_pretrained("gpt2")

    # Initialize FAISS index
    faiss_index = FaissIndex("sanad_index.faiss")
    if os.path.exists("sanad_index.faiss"):
        faiss_index.load_faiss_index()
    else:
        document_embeddings = doc_processor.embed_documents(doc_tokenizer, doc_model)
        faiss_index.create_faiss_index(document_embeddings)

    # Launch Gradio interface
    interface = gr.Interface(
        fn=lambda query: gradio_interface(query, faiss_index, documents, doc_tokenizer, doc_model, gpt_tokenizer,
                                          gpt_model),
        inputs=gr.Textbox(label="Query"),
        outputs=[
            gr.Textbox(label="Most Similar Articles"),
            gr.Textbox(label="Answer to the Question")
        ],
        title="Document Retrieval and Question Answering",
        description="Enter a question to retrieve the most similar articles and get an answer."
    )
    interface.launch()
