from flask import Flask, request, jsonify, render_template
from generation import RAGChain
from data_loader import DataLoader
from langchain.schema import Document
app = Flask(__name__)

base_dir = 'data'
data_loader = DataLoader(base_dir)
data = data_loader.load_data()
documents = [Document(page_content=text["content"]) for text in data[:166]]
google_api_key = ""
rag_chain = RAGChain(documents, google_api_key)


@app.route('/ask', methods=['POST'])
def ask():
    user_input = request.json.get('question')
    print(user_input)
    if not user_input:
        return jsonify({'error': 'No question provided'}), 400

    try:
        response = rag_chain.generate_answer(user_input)
        return jsonify({'answer': response})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/')
def index():
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)
