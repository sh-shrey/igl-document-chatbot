import os
import streamlit as st
import pickle
import numpy as np
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from langchain_community.document_loaders import Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

import ollama


class AdvancedEmbeddingManager:
    def __init__(self, document_path='Rnr-backend/IGL FAQs (1).docx', model_name='all-MiniLM-L6-v2'):
        self.embedding_model = SentenceTransformer(model_name)
        self.document_path = document_path

        self.cache_path = 'advanced_document_embedding_cache.pkl'
        self.documents = []
        self.embeddings = []

        self.load_or_create_embedding()

    def load_or_create_embedding(self):
        if os.path.exists(self.cache_path):
            try:
                with open(self.cache_path, 'rb') as f:
                    cached_data = pickle.load(f)
                    self.documents = cached_data['documents']
                    self.embeddings = cached_data['embeddings']
                print("Loaded cached embeddings")
                return
            except Exception as e:
                print(f"Error loading cache: {e}")

        loader = Docx2txtLoader(self.document_path)
        raw_documents = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=250,
            chunk_overlap=50,
            length_function=len
        )
        split_docs = text_splitter.split_documents(raw_documents)

        self.documents = [doc.page_content for doc in split_docs]
        self.embeddings = self.embedding_model.encode(self.documents)

        # Cache the results
        with open(self.cache_path, 'wb') as f:
            pickle.dump({
                'documents': self.documents,
                'embeddings': self.embeddings
            }, f)

        print("Created new embeddings")

    def retrieve_context(self, query: str, k: int = 3) -> List[str]:
        query_embedding = self.embedding_model.encode([query])[0]

        similarities = cosine_similarity([query_embedding], self.embeddings)[0]

        top_k_indices = np.argsort(similarities)[-k:][::-1]

        return [self.documents[idx] for idx in top_k_indices]


class OptimizedOllamaChatbot:
    def __init__(self,
                 embedding_manager: AdvancedEmbeddingManager,
                 ollama_model: str = 'mistral:7b-instruct-v0.2-q4_K_M',
                 temperature: float = 0.7):
        self.embedding_manager = embedding_manager
        self.ollama_model = ollama_model
        self.temperature = temperature

    def generate_response(self, query: str):
        contexts = self.embedding_manager.retrieve_context(query)

        # Advanced prompt engineering
        system_prompt = """
        You are an expert assistant for Indraprastha Gas Limited (IGL). 
        Provide precise, concise answers based strictly on the given context.
        If the information is not available, clearly state that.
        """

        full_prompt = f"""
{system_prompt}

Context:
{' '.join(contexts)}

Question: {query}

Answer:"""

        try:
            full_response = ""
            for chunk in ollama.chat(
                    model=self.ollama_model,
                    messages=[
                        {'role': 'system', 'content': system_prompt},
                        {'role': 'user', 'content': full_prompt}
                    ],
                    stream=True,
                    options={
                        'temperature': self.temperature,
                        'num_ctx': 4096,
                    }
            ):
                if chunk['done']:
                    break

                if 'message' in chunk and 'content' in chunk['message']:
                    chunk_content = chunk['message']['content']
                    full_response += chunk_content
                    yield full_response

        except Exception as e:
            yield f"An error occurred: {str(e)}"


def main():
    st.set_page_config(page_title="IGL Smart Assistant", page_icon="🔧")

    embedding_manager = AdvancedEmbeddingManager()
    chatbot = OptimizedOllamaChatbot(embedding_manager)

    st.title("🔧 IGL Document Intelligence Assistant")

    if 'messages' not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask me anything about IGL services"):
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            full_response = ""

            for chunk in chatbot.generate_response(prompt):
                full_response = chunk
                response_placeholder.markdown(full_response + "▌")

            response_placeholder.markdown(full_response)

        st.session_state.messages.append({"role": "assistant", "content": full_response})


if __name__ == "__main__":
    main()
