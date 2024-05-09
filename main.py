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
import email

from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_groq import ChatGroq
import ollama
from config import USER, APP_PASSWORD, IMAP_URL, SMTP_URL, SMTP_PORT, SYSTEM_PROMPT
from groqtools import run_conversation
from getnow import get_current_time_formatted
from nltk.stem import WordNetLemmatizer

nltk.download('wordnet')
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
    lemmatizer = WordNetLemmatizer()
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    lemmatized_tokens = [lemmatizer.lemmatize(token.lower()) for token in tokens if token.lower() not in stop_words]
    return ' '.join(lemmatized_tokens)


def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


class EmailHandler:
    def __init__(self, embedding_model: EmbeddingModel):
        self.embedding_model = embedding_model
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
        self.imap_server = None

    def retrieve_relevant_messages(self, user_prompt, memory):
        embeddings = self.embedding_model.get_embeddings()
        user_prompt_embedding = ollama.embeddings(model='snowflake-arctic-embed',
                                                   prompt=preprocess_text(user_prompt))['embedding']

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
                    return part.get_payload(decode=True).decode('utf-8', errors='ignore')
                elif content_type == 'text/html' and 'attachment' not in str(part.get('Content-Disposition')):
                    return self._parse_html_content(part.get_payload(decode=True).decode('utf-8', errors='ignore'))
        else:
            content_type = msg.get_content_type()
            if content_type == 'text/plain':
                return msg.get_payload(decode=True).decode('utf-8', errors='ignore')
            elif content_type == 'text/html':
                return self._parse_html_content(msg.get_payload(decode=True).decode('utf-8', errors='ignore'))

        return ""

    def _parse_html_content(self, html_content):
        soup = BeautifulSoup(html_content, 'html.parser')
        for element in soup(["script", "style"]):
            element.decompose()
        for link in soup.find_all('a'):
            link.decompose()
        text = soup.get_text()
        return re.sub(r'\s+', ' ', text).strip()

    def extract_sentences(self, text):
        return sent_tokenize(text)

    def clean_email_body(self, email_body):
        lines = email_body.split("\n")
        cleaned_lines = []

        for line in lines:
            if line.startswith("From:") or line.startswith("Sent:") or line.startswith("To:") or line.startswith(
                    "Subject:"):
                break
            cleaned_lines.append(line)

        return "\n".join(cleaned_lines).strip()

    def get_client_name(self, email_from):
        if '<' in email_from:
            client_name = email_from.split('<')[0].strip()
        else:
            client_name = email_from

        return client_name

    def process_email(self, mail_id, parser):
        _, data = self.imap_server.fetch(mail_id, '(RFC822)')
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

                sentences = self.extract_sentences(cleaned_email_body)

                client_name = self.get_client_name(email_from)

                for sentence in sentences:
                    user_prompt = f" {now}\n\nAs Chaos, analyze this email sentence and determine if it requires a response. You will always respond to emails asking for help from you. You have RAG using embedding with NLP to find relevant info that's also attached.\n\nSender: {email_from}\n\nClient Name: {client_name}\n\nEmail Content:\n{sentence}\n\nDoes this email require a response? Respond with only YES or NO."

                    try:
                        response = self.chat_with_groq(SYSTEM_PROMPT, user_prompt)
                        print(f"Analysis result: {response.content}")

                        if "YES" in response.content.strip().upper():
                            user_prompt = f" {now}\n\nAs Chaos, respond to this email sentence. Use any tools necessary to formulate your response.\n\nSender: {email_from}\n\nClient Name: {client_name}\n\nEmail Content:\n{sentence}\n\nOnly include in your response what you are sending to the client."
                            print(f"Using tools to formulate a response...")
                            tool_response = run_conversation(user_prompt)
                            print(f"Tool response: {tool_response}")
                            response = self.chat_with_groq(SYSTEM_PROMPT, user_prompt + "\n" + tool_response)
                            print(f"Final response: {response.content}")

                            self.send_response_email(email_from, email_subject, response.content)
                        else:
                            user_prompt = f" {now}\n\nAs Chaos, analyze this email sentence and figure out what tools may be needed, if none, just say none or leave a note with the note tool.\n\nSender: {email_from}\n\nClient Name: {client_name}\n\nEmail Content:\n{sentence}\n\n"
                            tool_response = run_conversation(user_prompt)

                            logging.info(
                                f"No response required for email sentence from {email_from} with subject '{email_subject}'. Tool response: {tool_response}")
                            print(
                                f"No response required for email sentence from {email_from} with subject '{email_subject}'. Tool response: {tool_response}")

                    except Exception as e:
                        logging.error(f"An error occurred while processing the email sentence: {str(e)}")
                        print(f"An error occurred while processing the email sentence: {str(e)}")
                        continue

    def save_draft_email(self, email_to, email_subject, email_body):
        try:
            message = MIMEText(email_body)
            message['From'] = USER
            message['To'] = email_to
            message['Subject'] = email_subject

            draft_folder = 'Drafts'  # Specify the name of the draft folder in your email account
            self.imap_server.append(draft_folder, '', imaplib.Time2Internaldate(time.time()), message.as_bytes())
            logging.info(f"Draft email saved for: {email_to}")
            print(f"Draft email saved for: {email_to}")
        except Exception as e:
            logging.error(f"An error occurred while saving the draft email: {str(e)}")
            print(f"An error occurred while saving the draft email: {str(e)}")

    def send_response_email(self, email_to, email_subject, response_content, send_now=False):
        if send_now:
            try:
                smtp_server = smtplib.SMTP(SMTP_URL, SMTP_PORT)
                smtp_server.starttls()
                smtp_server.login(USER, APP_PASSWORD)

                message = MIMEText(response_content)
                message['From'] = USER
                message['To'] = email_to
                message['Subject'] = email_subject

                smtp_server.sendmail(USER, [email_to], message.as_string())
                logging.info(f"Email sent to: {email_to}")
                print(f"Email sent to: {email_to}")
            except Exception as e:
                logging.error(f"An error occurred while sending the email: {str(e)}")
                print(f"An error occurred while sending the email: {str(e)}")
            finally:
                smtp_server.quit()
        else:
            self.save_draft_email(email_to, email_subject, response_content)

    def process_draft_emails(self):
        try:
            self.imap_server.select('Drafts')  # Select the draft folder
            _, data = self.imap_server.search(None, 'ALL')
            mail_ids = data[0].split()

            for mail_id in mail_ids:
                _, data = self.imap_server.fetch(mail_id, '(RFC822)')
                for response_part in data:
                    if isinstance(response_part, tuple):
                        msg = email.message_from_bytes(response_part[1])
                        email_to = msg['To']
                        email_subject = msg['Subject']
                        email_body = msg.get_payload()

                        self.send_response_email(email_to, email_subject, email_body, send_now=True)
                        self.imap_server.store(mail_id, '+FLAGS', '\\Deleted')  # Mark the draft email as deleted

            self.imap_server.expunge()  # Permanently remove deleted draft emails
        except Exception as e:
            logging.error(f"An error occurred while processing draft emails: {str(e)}")
            print(f"An error occurred while processing draft emails: {str(e)}")

    def handle_emails(self):
        parser = BytesParser(policy=policy.default)

        while True:
            try:
                self.imap_server = imaplib.IMAP4_SSL(IMAP_URL)
                self.imap_server.login(USER, APP_PASSWORD)

                # Process incoming emails
                self.imap_server.select('inbox')
                _, data = self.imap_server.search(None, 'UNSEEN')
                mail_ids = data[0].split()

                if mail_ids:
                    for mail_id in mail_ids:
                        self.process_email(mail_id, parser)
                else:
                    print("No new emails found.")

                # Process draft emails
                self.process_draft_emails()

            except Exception as e:
                logging.error(f"An error occurred while handling emails: {str(e)}")
                print(f"An error occurred while handling emails: {str(e)}")
            finally:
                try:
                 self.imap_server.close()
                 self.imap_server.logout()
                except Exception:
                   pass

            print("Waiting for 10 seconds before checking for new emails...")
            time.sleep(10)  # Wait for 10 seconds before checking for new emails again


if __name__ == "__main__":
   embedding_model = EmbeddingModel()
   email_handler = EmailHandler(embedding_model)
   email_handler.handle_emails()