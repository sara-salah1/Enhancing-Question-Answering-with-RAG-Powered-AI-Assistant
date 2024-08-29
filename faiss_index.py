import faiss


class FaissIndex:
    def __init__(self, index_file_path):
        self.index_file_path = index_file_path
        self.index = None

    def create_faiss_index(self, embeddings):
        self.index = faiss.IndexFlatL2(embeddings.shape[1])
        self.index.add(embeddings)
        faiss.write_index(self.index, self.index_file_path)
        return self.index

    def load_faiss_index(self):
        self.index = faiss.read_index(self.index_file_path)
        return self.index

    def retrieve_top_documents(self, query_embedding, top_k=3):
        _, top_indices = self.index.search(query_embedding.reshape(1, -1), k=top_k)
        return top_indices
