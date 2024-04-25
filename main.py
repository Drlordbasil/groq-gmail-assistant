import datetime
import imaplib
import smtplib
from email.mime.text import MIMEText
from email import policy
from email.parser import BytesParser
import json
import os
import time
import logging
from bs4 import BeautifulSoup
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.sentiment import SentimentIntensityAnalyzer
import numpy as np

from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_groq import ChatGroq
import ollama
from config import USER, APP_PASSWORD, IMAP_URL, SMTP_URL, SMTP_PORT, SYSTEM_PROMPT
from groqtools import run_conversation
from getnow import get_current_time_formatted

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('vader_lexicon')

now = get_current_time_formatted()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if GROQ_API_KEY is None:
    raise ValueError("GROQ_API_KEY environment variable is not set.")

logging.basicConfig(filename='gmail_assistant.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

class EmbeddingModel:
    def __init__(self, json_file_path: str = 'conversation_memory.json', model_name: str = 'mxbai-embed-large'):
        self.json_file_path = json_file_path
        self.model_name = model_name

    def _load_documents(self) -> list:
        with open(self.json_file_path, 'r') as file:
            return json.load(file)

    def get_embeddings(self) -> list:
        documents = self._load_documents()
        return [ollama.embeddings(model=self.model_name, prompt=doc['content'])['embedding'] for doc in documents]

class ConversationMemory:
    def __init__(self, memory_file="conversation_memory.json"):
        self.memory_file = memory_file
        self.history = self._load_memory()

    def save_context(self, role, content):
        self.history.append({"role": role, "content": content})
        self._save_memory()

    def get_history(self):
        return self.history

    def _load_memory(self):
        try:
            with open(self.memory_file, "r") as file:
                return json.load(file)
        except FileNotFoundError:
            return []

    def _save_memory(self):
        with open(self.memory_file, "w") as file:
            json.dump(self.history, file, indent=2)

class ChatGroqFactory:
    @staticmethod
    def create_chat_groq(temperature=0.7, model_name="llama3-70b-8192"):
        return ChatGroq(groq_api_key=GROQ_API_KEY, temperature=temperature, model_name=model_name)

def preprocess_text(text: str) -> str:
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [token.lower() for token in tokens if token.lower() not in stop_words]
    return ' '.join(filtered_tokens)

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

class EmailHandler:
    def __init__(self, embedding_model: EmbeddingModel):
        self.embedding_model = embedding_model
        self.sentiment_analyzer = SentimentIntensityAnalyzer()

    def retrieve_relevant_messages(self, user_prompt, memory):
        embeddings = self.embedding_model.get_embeddings()
        user_prompt_embedding = ollama.embeddings(model='mxbai-embed-large', prompt=preprocess_text(user_prompt))['embedding']

        return [
            msg for i, msg in enumerate(memory.get_history())
            if cosine_similarity(np.array(embeddings[i]), np.array(user_prompt_embedding)) > 0.8
        ]

    def chat_with_groq(self, system_prompt, user_prompt, chat_instance=None, memory=None):
        chat_instance = chat_instance or ChatGroqFactory.create_chat_groq()
        memory = memory or ConversationMemory()
        memory.save_context("user", user_prompt)

        relevant_messages = self.retrieve_relevant_messages(user_prompt, memory)
        history = relevant_messages + [{"role": "user", "content": user_prompt}]
        messages = [SystemMessagePromptTemplate.from_template(system_prompt)] + [
            HumanMessagePromptTemplate.from_template(msg["content"]) if msg["role"] == "user" else
            SystemMessagePromptTemplate.from_template(msg["content"]) for msg in history
        ]

        prompt = ChatPromptTemplate.from_messages(messages)
        response = chat_instance.invoke(prompt.format_prompt())
        memory.save_context("assistant", response.content)

        return response

    def get_email_body(self, msg):
        if msg.is_multipart():
            for part in msg.walk():
                content_type = part.get_content_type()
                if content_type == 'text/plain' and 'attachment' not in str(part.get('Content-Disposition')):
                    return part.get_payload(decode=True).decode('utf-8')
                elif content_type == 'text/html' and 'attachment' not in str(part.get('Content-Disposition')):
                    return self._parse_html_content(part.get_payload(decode=True).decode('utf-8'))
        else:
            content_type = msg.get_content_type()
            if content_type == 'text/plain':
                return msg.get_payload(decode=True).decode('utf-8')
            elif content_type == 'text/html':
                return self._parse_html_content(msg.get_payload(decode=True).decode('utf-8'))

        return ""

    def _parse_html_content(self, html_content):
        soup = BeautifulSoup(html_content, 'html.parser')
        for element in soup(["script", "style"]):
            element.decompose()
        for link in soup.find_all('a'):
            link.decompose()
        text = soup.get_text()
        return re.sub(r'\s+', ' ', text).strip()

    def chunk_text(self, text, chunk_size=4096):
        sentences = sent_tokenize(text)
        chunks = []
        current_chunk = ""

        for sentence in sentences:
            if len(current_chunk) + len(sentence) <= chunk_size:
                current_chunk += sentence + " "
            else:
                chunks.append(current_chunk.strip())
                current_chunk = sentence + " "

        if current_chunk:
            chunks.append(current_chunk.strip())

        return chunks

    def filter_chunks(self, chunks):
        return [
            chunk for chunk in chunks
            if len(chunk) <= 1500 or self.sentiment_analyzer.polarity_scores(chunk)['compound'] >= 0.05
        ]

    def clean_email_body(self, email_body):
        lines = email_body.split("\n")
        cleaned_lines = []
        reply_found = False

        for line in lines:
            if line.startswith("From:") or line.startswith("Sent:") or line.startswith("To:") or line.startswith("Subject:"):
                reply_found = True
                continue
            
            if reply_found and line.strip() == "":
                break

            if not reply_found:
                cleaned_lines.append(line)

        return "\n".join(cleaned_lines).strip()

    def get_client_name(self, email_from):
        if '<' in email_from:
            client_name = email_from.split('<')[0].strip()
        else:
            client_name = email_from

        return client_name

    def process_email(self, mail_id, imap_server, parser):
        _, data = imap_server.fetch(mail_id, '(RFC822)')
        for response_part in data:
            if isinstance(response_part, tuple):
                msg = parser.parsebytes(response_part[1])
                email_from = msg['from']
                email_subject = msg['subject'] or "No Subject"
                email_body = self.get_email_body(msg)

                print(f"Received email from: {email_from}")
                print(f"Email subject: {email_subject}")
                print(f"Email body: {email_body}")

                skip_keywords = [
                    "no-reply", "do-not-reply", "noreply", "donotreply", "no_reply", "do_not_reply",
                    "newsletter", "news letter", "notifications", "notification", "account", "accounts"
                ]
                if any(keyword in email_from.lower() or keyword in email_subject.lower() for keyword in skip_keywords):
                    logging.info(f"Skipping email from {email_from} with subject '{email_subject}'.")
                    print(f"Skipping email from {email_from} with subject '{email_subject}'.")
                    return

                logging.info(f"Processing new email from {email_from} with subject '{email_subject}'.")
                print(f"Processing new email from {email_from} with subject '{email_subject}'.")

                cleaned_email_body = self.clean_email_body(email_body)
                print(f"Cleaned email body: {cleaned_email_body}")

                filtered_chunks = self.filter_chunks(self.chunk_text(cleaned_email_body))

                client_name = self.get_client_name(email_from)

                for chunk in filtered_chunks:
                    user_prompt = f" {now}\n\nAs Chaos, analyze this email chunk and determine if it requires a response.\n\nSender: {email_from}\n\nClient Name: {client_name}\n\nEmail Content:\n{chunk}\n\nDoes this email require a response? Respond with only YES or NO."

                    try:
                        response = self.chat_with_groq(SYSTEM_PROMPT, user_prompt)
                        print(f"Analysis result: {response.content}")

                        if "YES" in response.content.strip().upper():
                            user_prompt = f" {now}\n\nAs Chaos, respond to this email chunk. Use any tools necessary to formulate your response.\n\nSender: {email_from}\n\nClient Name: {client_name}\n\nEmail Content:\n{chunk}\n\nOnly include in your response what you are sending to the client."
                            print(f"Using tools to formulate a response...")
                            tool_response = run_conversation(user_prompt)
                            print(f"Tool response: {tool_response}")
                            response = self.chat_with_groq(SYSTEM_PROMPT, user_prompt + "\n" + tool_response)
                            print(f"Final response: {response.content}")

                            self.send_response_email(email_from, email_subject, response.content)
                        else:
                            logging.info(f"No response required for email chunk from {email_from} with subject '{email_subject}'.")
                            print(f"No response required for email chunk from {email_from} with subject '{email_subject}'.")

                    except Exception as e:
                        logging.error(f"An error occurred while processing the email chunk: {str(e)}")
                        print(f"An error occurred while processing the email chunk: {str(e)}")

    def send_response_email(self, email_from, email_subject, response_content):
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
        print(f"Auto-reply sent to: {email_from}")

    def handle_emails(self):
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
                    self.process_email(mail_id, imap_server, parser)

                imap_server.logout()
            except Exception as e:
                logging.error(f"An error occurred: {e}")
                print(f"An error occurred: {e}")
            finally:
                time.sleep(15)


if __name__ == "__main__":
    embedding_model = EmbeddingModel()
    email_handler = EmailHandler(embedding_model)
    email_handler.handle_emails()
