import json
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.sentiment import SentimentIntensityAnalyzer
import numpy as np
import ollama
from nltk.stem import WordNetLemmatizer

class ConversationMemory:
    def __init__(self, memory_file="conversation_memory.json", max_memory_size=10):
        self.memory_file = memory_file
        self.history = self._load_memory()
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
        self.embedding_model = EmbeddingModel()
        self.max_memory_size = max_memory_size

    def save_context(self, role, content):
        self.history.append({"role": role, "content": content})
        self._save_memory()

    def get_history(self):
        return self.history

    def _load_memory(self):
        try:
            with open(self.memory_file, "r") as file:
                memory = json.load(file)
                return memory
        except FileNotFoundError:
            return []

    def _save_memory(self):
        with open(self.memory_file, "w") as file:
            json.dump(self.history, file, indent=2)

    def retrieve_relevant_messages(self, user_prompt, memory):
        embeddings = self.embedding_model.get_embeddings()
        user_prompt_embedding = ollama.embeddings(model='snowflake-arctic-embed', prompt=preprocess_text(user_prompt))['embedding']
        
        relevant_messages = sorted(
            [(i, msg, cosine_similarity(np.array(embeddings[i]), np.array(user_prompt_embedding))) for i, msg in enumerate(memory.get_history())],
            key=lambda x: x[2],
            reverse=True
        )[:self.max_memory_size]

        return [msg for _, msg, _ in relevant_messages]

class EmbeddingModel:
    def __init__(self, json_file_path: str = 'conversation_memory.json', model_name: str = 'snowflake-arctic-embed'):
        self.json_file_path = json_file_path
        self.model_name = model_name

    def _load_documents(self) -> list:
        try:
            with open(self.json_file_path, 'r') as file:
                return json.load(file)
        except FileNotFoundError:
            return []

    def get_embeddings(self) -> list:
        documents = self._load_documents()
        embeddings = [ollama.embeddings(model=self.model_name, prompt=doc['content'])['embedding'] for doc in documents]
        return embeddings

def preprocess_text(text: str) -> str:
    lemmatizer = WordNetLemmatizer()
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    lemmatized_tokens = [lemmatizer.lemmatize(token.lower()) for token in tokens if token.lower() not in stop_words]
    return ' '.join(lemmatized_tokens)

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))