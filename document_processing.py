import os
import glob
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel


class DocumentProcessor:
    def __init__(self, data_directory):
        self.data_directory = data_directory
        self.documents = []
        self.labels = []

    def load_documents_and_labels(self):
        category_folders = os.listdir(self.data_directory)

        for category in tqdm(category_folders, desc="Processing Category Folders"):
            category_path = os.path.join(self.data_directory, category)
            if os.path.isdir(category_path):
                text_files = glob.glob(f"{category_path}/*.txt")
                for text_file in tqdm(text_files, desc=f"Processing Files in {category}", leave=False):
                    with open(text_file, 'r', encoding='utf-8') as file:
                        content = file.read().strip()
                        self.documents.append(content)
                        self.labels.append(category)

        print("length documents: ", len(self.documents))
        print("length labels: ", len(self.labels))

        # Save documents and labels
        np.save("document_labels.npy", self.labels)
        np.save("document_texts.npy", self.documents)

    def embed_documents(self, tokenizer, model):
        document_embeddings = []

        for document in tqdm(self.documents, desc="Embedding Documents"):
            tokenized_input = tokenizer(document, return_tensors='pt', padding=True, truncation=True, max_length=512)
            with torch.no_grad():
                embedding = model(**tokenized_input).last_hidden_state.mean(dim=1).squeeze().numpy()
            document_embeddings.append(embedding)

        return np.array(document_embeddings)

    def embed_query(self, tokenizer, model, query):
        tokenized_input = tokenizer(query, return_tensors='pt', padding=True, truncation=True, max_length=512)
        with torch.no_grad():
            embedding = model(**tokenized_input).last_hidden_state.mean(dim=1).squeeze().numpy()
        return embedding
