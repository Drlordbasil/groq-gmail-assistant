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
from tool_stuff.groqtools import run_conversation
from utilities.getnow import get_current_time_formatted
from nltk.stem import WordNetLemmatizer

# Download necessary NLTK data
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('vader_lexicon')

# Get current time formatted
now = get_current_time_formatted()

# Set Groq API key from environment variable
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if GROQ_API_KEY is None:
    raise ValueError("GROQ_API_KEY environment variable is not set.")

# Set up logging
logging.basicConfig(filename='gmail_assistant.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


class EmbeddingModel:
    """
    EmbeddingModel handles loading conversation documents and retrieving their embeddings.
    """
    def __init__(self, json_file_path: str = 'conversation_memory.json', model_name: str = 'snowflake-arctic-embed'):
        self.json_file_path = json_file_path
        self.model_name = model_name

    def _load_documents(self) -> list:
        """Load documents from a JSON file."""
        try:
            with open(self.json_file_path, 'r') as file:
                return json.load(file)
        except FileNotFoundError:
            return []

    def get_embeddings(self) -> list:
        """Retrieve embeddings for the loaded documents."""
        documents = self._load_documents()
        return [ollama.embeddings(model=self.model_name, prompt=doc['content'])['embedding'] for doc in documents]


class ConversationMemory:
    """
    ConversationMemory manages saving and loading conversation history.
    """
    def __init__(self, memory_file="conversation_memory.json"):
        self.memory_file = memory_file
        self.history = self._load_memory()

    def save_context(self, role: str, content: str):
        """Save context to memory."""
        self.history.append({"role": role, "content": content})
        self._save_memory()

    def get_history(self) -> list:
        """Get the conversation history."""
        return self.history

    def _load_memory(self) -> list:
        """Load conversation memory from a file."""
        try:
            with open(self.memory_file, "r") as file:
                return json.load(file)
        except FileNotFoundError:
            return []

    def _save_memory(self):
        """Save conversation memory to a file."""
        with open(self.memory_file, "w") as file:
            json.dump(self.history, file, indent=2)


class ChatGroqFactory:
    """
    ChatGroqFactory creates ChatGroq instances with specified parameters.
    """
    @staticmethod
    def create_chat_groq(temperature=0.7, model_name="llama3-70b-8192") -> ChatGroq:
        """Create a ChatGroq instance."""
        return ChatGroq(groq_api_key=GROQ_API_KEY, temperature=temperature, model_name=model_name)


def preprocess_text(text: str) -> str:
    """
    Preprocess text by tokenizing, removing stop words, and lemmatizing.

    Args:
        text (str): Input text to preprocess.

    Returns:
        str: Preprocessed text.
    """
    lemmatizer = WordNetLemmatizer()
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    lemmatized_tokens = [lemmatizer.lemmatize(token.lower()) for token in tokens if token.lower() not in stop_words]
    return ' '.join(lemmatized_tokens)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Calculate the cosine similarity between two vectors.

    Args:
        a (np.ndarray): First vector.
        b (np.ndarray): Second vector.

    Returns:
        float: Cosine similarity between vectors a and b.
    """
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


class EmailHandler:
    """
    EmailHandler processes emails, retrieves relevant messages, and interacts with Groq.
    """
    def __init__(self, embedding_model: EmbeddingModel):
        self.embedding_model = embedding_model
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
        self.imap_server = None
        self.response_to_client = ""
        self.email_from_client = ""
        self.drafted_email = ""

    def retrieve_relevant_messages(self, user_prompt: str, memory: ConversationMemory) -> list:
        """
        Retrieve relevant messages based on user prompt and memory.

        Args:
            user_prompt (str): User's prompt.
            memory (ConversationMemory): Conversation memory.

        Returns:
            list: List of relevant messages.
        """
        embeddings = self.embedding_model.get_embeddings()
        user_prompt_embedding = ollama.embeddings(model='snowflake-arctic-embed',
                                                  prompt=preprocess_text(user_prompt))['embedding']

        return [
            msg for i, msg in enumerate(memory.get_history())
            if cosine_similarity(np.array(embeddings[i]), np.array(user_prompt_embedding)) > 0.8
        ]

    def chat_with_groq(self, system_prompt: str, user_prompt: str, chat_instance: ChatGroq = None, memory: ConversationMemory = None) -> str:
        """
        Interact with Groq based on system and user prompts.

        Args:
            system_prompt (str): System prompt.
            user_prompt (str): User prompt.
            chat_instance (ChatGroq, optional): ChatGroq instance. Defaults to None.
            memory (ConversationMemory, optional): Conversation memory. Defaults to None.

        Returns:
            str: Response from Groq.
        """
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

    def get_email_body(self, msg) -> str:
        """
        Extract the email body from an email message.

        Args:
            msg (email.message.Message): Email message.

        Returns:
            str: Email body.
        """
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

    def _parse_html_content(self, html_content: str) -> str:
        """
        Parse and clean HTML content to extract text.

        Args:
            html_content (str): HTML content.

        Returns:
            str: Cleaned text from HTML.
        """
        soup = BeautifulSoup(html_content, 'html.parser')
        for element in soup(["script", "style"]):
            element.decompose()
        for link in soup.find_all('a'):
            link.decompose()
        text = soup.get_text()
        return re.sub(r'\s+', ' ', text).strip()

    def extract_sentences(self, text: str) -> list:
        """
        Extract sentences from text.

        Args:
            text (str): Input text.

        Returns:
            list: List of sentences.
        """
        return sent_tokenize(text)

    def clean_email_body(self, email_body: str) -> str:
        """
        Clean the email body by removing unnecessary headers.

        Args:
            email_body (str): Raw email body.

        Returns:
            str: Cleaned email body.
        """
        lines = email_body.split("\n")
        cleaned_lines = []

        for line in lines:
            if line.startswith("From:") or line.startswith("Sent:") or line.startswith("To:") or line.startswith(
                    "Subject:"):
                break
            cleaned_lines.append(line)

        return "\n".join(cleaned_lines).strip()

    def get_client_name(self, email_from: str) -> str:
        """
        Extract client name from email address.

        Args:
            email_from (str): Email address.

        Returns:
            str: Client name.
        """
        if '<' in email_from:
            client_name = email_from.split('<')[0].strip()
        else:
            client_name = email_from

        return client_name

    def process_email(self, mail_id: bytes, parser: BytesParser):
        """
        Process an email based on its mail ID.

        Args:
            mail_id (bytes): Mail ID.
            parser (BytesParser): BytesParser instance to parse the email.
        """
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
                    user_prompt = f"{now}\n\nAs Chaos, analyze this email sentence and determine if it requires a response. You will always respond to emails asking for help from you. You have RAG using embedding with NLP to find relevant info that's also attached.\n\nSender: {email_from}\n\nClient Name: {client_name}\n\nEmail Content:\n{sentence}\n\nDoes this email require a response? Respond with only YES or NO."

                    try:
                        response = self.chat_with_groq(SYSTEM_PROMPT, user_prompt)
                        print(f"Analysis result: {response.content}")

                        if "YES" in response.content.strip().upper():
                            user_prompt = f"{now}\n\nAs Chaos, respond to this email sentence. Use any tools necessary to formulate your response. Provide only the response text, without any additional information.\n\nSender: {email_from}\n\nClient Name: {client_name}\n\nEmail Content:\n{sentence}"
                            print(f"Using tools to formulate a response...")
                            tool_response = run_conversation(user_prompt)
                            print(f"Tool response: {tool_response}")

                            # Combine tool_response in the final response API request
                            memories = self.retrieve_relevant_messages(sentence, ConversationMemory())
                            response = self.chat_with_groq(SYSTEM_PROMPT, user_prompt + "\n" + tool_response + "\n" + memories[-1]["content"])
                            print(f"Final response: {response.content}")

                            self.drafted_email = response.content.strip()
                            self.response_to_client = response.content.strip()
                            self.email_from_client = email_from
                        else:
                            print("Email does not require a response.")
                            pass

                    except Exception as e:
                        logging.error(f"An error occurred while processing the email sentence: {str(e)}")
                        print(f"An error occurred while processing the email sentence: {str(e)}")
                        continue

    def send_response_email(self, email_to: str, email_subject: str, response_content: str):
        """
        Send a response email.

        Args:
            email_to (str): Recipient email address.
            email_subject (str): Email subject.
            response_content (str): Email body content.
        """
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

    def handle_emails(self):
        """
        Handle incoming emails.
        """
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
                        if self.response_to_client and self.email_from_client:
                            self.send_response_email(self.email_from_client, "Re: Your Inquiry", self.response_to_client)
                else:
                    print("No new emails found.")

            except imaplib.IMAP4.error as e:
                logging.error(f"IMAP error occurred: {str(e)}")
                print(f"IMAP error occurred: {str(e)}")
            except Exception as e:
                logging.error(f"An error occurred while handling emails: {str(e)}")
                print(f"An error occurred while handling emails: {str(e)}")
            finally:
                try:
                    if self.imap_server.state == 'SELECTED':
                        self.imap_server.close()
                    self.imap_server.logout()
                except imaplib.IMAP4.error as e:
                    logging.error(f"IMAP error during logout: {str(e)}")
                    print(f"IMAP error during logout: {str(e)}")
                except Exception as e:
                    logging.error(f"An error occurred during IMAP logout: {str(e)}")
                    print(f"An error occurred during IMAP logout: {str(e)}")

            print("Waiting for 30 seconds before checking for new emails...")
            time.sleep(30)  # Wait for 30 seconds before checking for new emails again


if __name__ == "__main__":
    embedding_model = EmbeddingModel()
    email_handler = EmailHandler(embedding_model)
    email_handler.handle_emails()
