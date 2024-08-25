from sklearn.metrics.pairwise import cosine_similarity


class Evaluator:
    def __init__(self, embedding_model, similarity_threshold=0.8):
        self.embedding_model = embedding_model
        self.similarity_threshold = similarity_threshold

    def compute_similarity(self, answer1, answer2):
        embeddings1 = self.embedding_model.embed_query(answer1)
        embeddings2 = self.embedding_model.embed_query(answer2)
        similarity = cosine_similarity([embeddings1], [embeddings2])[0][0]
        return similarity

    def evaluate(self, evaluation_data, rag_chain):
        print('Enter Evaluate')
        print("evaluation_data: ", evaluation_data)
        generated_answers = [rag_chain.generate_answer(item["question"]) for item in evaluation_data]
        print('generated: ', generated_answers)
        correct_count = 0
        for item, generated_answer in zip(evaluation_data, generated_answers):
            similarity = self.compute_similarity(item["correct_answer"], generated_answer)
            if similarity >= self.similarity_threshold:
                correct_count += 1
        accuracy = (correct_count / len(evaluation_data)) * 100
        return accuracy
