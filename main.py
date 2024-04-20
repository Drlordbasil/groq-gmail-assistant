import imaplib
import smtplib
from email.mime.text import MIMEText
from email import policy
from email.parser import BytesParser
import json
import os
import time
import logging
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import numpy as np

from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_groq import ChatGroq
import ollama

# Download NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# User configuration
USER = '' # email address here
APP_PASSWORD = '' # add 2FA app password here from google account
IMAP_URL = 'imap.gmail.com' # IMAP URL for Gmail
SMTP_URL = 'smtp.gmail.com' # SMTP URL for Gmail
SMTP_PORT = 587 # SMTP port for Gmail

SYSTEM_PROMPT = '''

'''
## Add your system prompt here ^
# Load GROQ API key from environment
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if GROQ_API_KEY is None:
    raise ValueError("GROQ_API_KEY environment variable is not set.")
else:
    print("GROQ_API_KEY loaded successfully.")

# Set up logging
logging.basicConfig(filename='gmail_assistant.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

class OllamaEmbedding:
    def __init__(self, json_file_path: str = 'conversation_memory.json', model_name: str = 'mxbai-embed-large'):
        self.json_file_path = json_file_path
        self.model_name = model_name

    def _load_documents(self) -> list:
        with open(self.json_file_path, 'r') as file:
            return json.load(file)

    def get_embeddings(self) -> str:
        documents = self._load_documents()
        embeddings = []
        for doc in documents:
            response = ollama.embeddings(model=self.model_name, prompt=doc['content'])
            if 'embedding' in response:
                embeddings.append(response['embedding'])
            else:
                embeddings.append({'error': 'Failed to get embedding'})
        return json.dumps(embeddings)

class ConversationMemory:
    def __init__(self, memory_file="conversation_memory.json"):
        self.memory_file = memory_file
        self.history = self.load_memory()

    def save_context(self, role, content):
        self.history.append({"role": role, "content": content})
        self.save_memory()

    def get_history(self):
        return self.history

    def load_memory(self):
        try:
            with open(self.memory_file, "r") as file:
                return json.load(file)
        except FileNotFoundError:
            return []

    def save_memory(self):
        with open(self.memory_file, "w") as file:
            json.dump(self.history, file, indent=2)

def create_chat_groq(temperature=0.7, model_name="llama3-70b-8192"):
    return ChatGroq(groq_api_key=GROQ_API_KEY, temperature=temperature, model_name=model_name)

def preprocess_text(text):
    # Tokenize the text
    tokens = word_tokenize(text)
    
    # Remove stop words
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [token.lower() for token in tokens if token.lower() not in stop_words]
    
    return ' '.join(filtered_tokens)

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def retrieve_relevant_messages(user_prompt, memory):
    ollama_embedding = OllamaEmbedding()
    embeddings = json.loads(ollama_embedding.get_embeddings())

    relevant_messages = []
    user_prompt_processed = preprocess_text(user_prompt)
    user_prompt_embedding = ollama.embeddings(model='mxbai-embed-large', prompt=user_prompt_processed)['embedding']
    
    for i, msg in enumerate(memory.get_history()):
        if 'error' not in embeddings[i]:
            msg_processed = preprocess_text(msg['content'])
            similarity = cosine_similarity(np.array(embeddings[i]), np.array(user_prompt_embedding))
            if similarity > 0.8:
                relevant_messages.append(msg)

    return relevant_messages

def chat_with_groq(system_prompt, user_prompt, chat_instance=None, memory=None):
    if chat_instance is None:
        chat_instance = create_chat_groq()
    if memory is None:
        memory = ConversationMemory()

    memory.save_context("user", user_prompt)

    relevant_messages = retrieve_relevant_messages(user_prompt, memory)
    history = relevant_messages + [{"role": "user", "content": user_prompt}]
    messages = [SystemMessagePromptTemplate.from_template(system_prompt)] + \
               [HumanMessagePromptTemplate.from_template(msg["content"]) if msg["role"] == "user" else 
                SystemMessagePromptTemplate.from_template(msg["content"]) for msg in history]

    prompt = ChatPromptTemplate.from_messages(messages)
    response = chat_instance.invoke(prompt.format_prompt())
    memory.save_context("assistant", response.content)

    return response

def get_email_body(msg):
    if msg.is_multipart():
        for part in msg.walk():
            ctype = part.get_content_type()
            cdispo = str(part.get('Content-Disposition'))

            if ctype == 'text/plain' and 'attachment' not in cdispo:
                return part.get_payload(decode=True).decode('utf-8')
    else:
        return msg.get_payload(decode=True).decode('utf-8')

def process_email(mail_id, imap_server, parser):
    _, data = imap_server.fetch(mail_id, '(RFC822)')
    for response_part in data:
        if isinstance(response_part, tuple):
            msg = parser.parsebytes(response_part[1])
            email_from = msg['from']
            email_subject = msg['subject'] or "No Subject"
            email_body = get_email_body(msg)

            # Check if the email is from a no-reply address
            if "no-reply" in email_from.lower() or "do-not-reply" in email_from.lower():
                logging.info(f"Skipping no-reply email from {email_from} with subject '{email_subject}'.")
                print(f"Skipping no-reply email from {email_from} with subject '{email_subject}'.")
                return

            logging.info(f"Processing new email from {email_from} with subject '{email_subject}'.")
            print(f"Processing new email from {email_from} with subject '{email_subject}'.")

            user_prompt = f"Respond to this email: sender:{email_from}\n\n{email_body}\n\n As Anthony's assistant named Chaos."
            response = chat_with_groq(SYSTEM_PROMPT, user_prompt)
            print("Response to email:", response.content)

            send_response_email(email_from, email_subject, response.content)

def send_response_email(email_from, email_subject, response_content):
    smtp_server = smtplib.SMTP(SMTP_URL, SMTP_PORT)
    smtp_server.starttls()
    smtp_server.login(USER, APP_PASSWORD)

    message = MIMEText(response_content)
    message['From'] = USER
    message['To'] = email_from
    message['Subject'] = "Re: " + email_subject

    smtp_server.sendmail(USER, [email_from], message.as_string())
    smtp_server.quit()

    logging.info(f"Auto-reply sent to: {email_from}")

def handle_emails():
    parser = BytesParser(policy=policy.default)

    while True:
        try:
            imap_server = imaplib.IMAP4_SSL(IMAP_URL)
            imap_server.login(USER, APP_PASSWORD)
            imap_server.select('inbox')

            _, data = imap_server.search(None, 'UNSEEN')
            mail_ids = data[0].split()

            if not mail_ids:
                imap_server.logout()
                time.sleep(15)
                continue

            for mail_id in mail_ids:
                process_email(mail_id, imap_server, parser)

            imap_server.logout()
        except Exception as e:
            logging.error(f"An error occurred: {e}")
        finally:
            time.sleep(15)

if __name__ == "__main__":
    handle_emails()
