from transformers import GPT2LMHeadModel, GPT2Tokenizer
from bert_score import score


class ResponseGenerator:
    def __init__(self, gpt_tokenizer, gpt_model):
        self.gpt_tokenizer = gpt_tokenizer
        self.gpt_model = gpt_model

    def generate_response(self, query, retrieved_documents):
        input_text = query + "\n\n" + "\n".join(retrieved_documents)
        max_length = self.gpt_model.config.n_positions
        if len(input_text) > max_length:
            input_text = input_text[:max_length]

        tokenized_input = self.gpt_tokenizer(input_text, return_tensors='pt', max_length=max_length, truncation=True, clean_up_tokenization_spaces=False)
        try:
            generated_outputs = self.gpt_model.generate(tokenized_input['input_ids'], max_length=512, num_return_sequences=1)
            response = self.gpt_tokenizer.decode(generated_outputs[0], skip_special_tokens=True)
        except Exception as e:
            response = "Error generating response."

        return response

    def evaluate_with_bertscore(self, queries):
        generated_responses = []
        expected_responses = []

        for q in queries:
            generated_response = self.generate_response(q['query'], q['relevant_docs'])
            generated_responses.append(generated_response)
            expected_responses.append(q['expected_response'])

        P, R, F1 = score(generated_responses, expected_responses, lang='ar')
        print(f"Average Precision: {P.mean():.4f}")
        print(f"Average Recall: {R.mean():.4f}")
        print(f"Average F1 Score: {F1.mean():.4f}")
