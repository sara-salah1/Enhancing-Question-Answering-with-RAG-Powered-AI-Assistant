from sentence_transformers import SentenceTransformer


class EmbeddingModel:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')

    def embed_documents(self, docs):
        if not docs:
            print("Warning: No documents to embed.")
            return []
        print("docs: ", len(docs))
        embeddings = self.model.encode(docs)
        print('embed: ', len(embeddings))
        return embeddings.tolist()

    def embed_query(self, query):
        return self.model.encode(query).tolist()
