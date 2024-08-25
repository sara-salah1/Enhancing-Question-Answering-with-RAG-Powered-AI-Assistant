from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_google_genai import ChatGoogleGenerativeAI

from embedding import EmbeddingModel


class RAGChain:
    def __init__(self, documents, google_api_key):
        print("Enter RAG chain")
        self.vector_database = Chroma.from_documents(documents=documents, embedding=EmbeddingModel())
        print("vector database: ", self.vector_database)
        self.retriever = self.vector_database.as_retriever(search_type="similarity", search_kwargs={'k': 3})
        self.template = """
        You are an AI-powered QA Assistant designed your role is to provide precise and contextually 
        appropriate answers to customer questions.
        At the end of the answer thank the user.
        The answer must be detailed and no less than 100 words.
        if you are asked about yourself, tell the user you are AI-powered QA Assistant.

        what to do if the answer is not included in the prompt or the context:
            1. apologies to the user
            2. tell the user that you don't know the answer for the asked question.
            3. ask the user if he has more questions to ask.
            4. do not mention anything about the context.

        for the answer:
            1. the output must be the answer only without any additional thoughts.

        knowledge you know"
        {context}

        Question: {question}

        answer: 
        """
        self.custom_rag_prompt = PromptTemplate.from_template(self.template)
        self.llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=google_api_key, temperature=0)
        self.rag_chain = (
            {"context": self.retriever | self.format_docs, 'question': RunnablePassthrough()}
            | self.custom_rag_prompt
            | self.llm
            | StrOutputParser()
        )

    def format_docs(self, docs):
        return "\n\n".join(doc.page_content for doc in docs)

    def generate_answer(self, question):
        print("question: ", question)
        res = self.rag_chain.invoke(question)
        print("result: ", res)
        return res
