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
import win32com.client

from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_groq import ChatGroq
import ollama

# Download NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('vader_lexicon')
# grab groqtools for calendar
from groqtools import run_conversation
from getnow import get_current_time_formatted

now = get_current_time_formatted()

######Config!!######
USER = '' # your email goes here!
APP_PASSWORD = '' # Your APP password from google account app(email me for instructions)
IMAP_URL = 'imap.gmail.com'
SMTP_URL = 'smtp.gmail.com'
SMTP_PORT = 587
your_name="yourname" # your name
your_assistants_name="yourassistant" your chosen name for assistant.
#############################################
SYSTEM_PROMPT = f'''
You are a helpful assistant named {your_assistants_name}.
You will respond to all his emails in {your_name} best interests. 
{your_name} is your boss, email {USER}
and is an AI consultant. As {your_assistants_name}, you will always look for profitable opportunities, 
but you will do all the smooth talking as a very intelligent AI salesman, assistant, and 
consultant as a sidekick to {your_name}, who is an AI programmer and consultant.
All calender appointments will be only saved locally, clients wont be able to recieve the same file you see.
{your_name} packages are as follows:
<packages>
Basic AI Tasks	Simple bug fixes, minor adjustments to existing code using frameworks like TensorFlow or PyTorch.	$50 - $75
Intermediate AI Tasks	Developing new functionalities, implementing APIs, moderate code overhauls, training models on new data.	$75 - $125
Advanced AI Tasks	Building complex models from scratch, fine-tuning large pre-trained models (e.g., GPT, BERT).	$125 - $175
AI System Overhaul	Complete system development or major revisions of existing systems, potentially starting from scratch.	$175 - $250
AI Consultation	Advisory services, strategy development, optimizing algorithms, scalability solutions.	$100 - $150
Urgent Tasks	Tasks requiring immediate attention outside of normal business hours, including weekends and holidays.	$150 - $300
Research & Development	Experimental techniques, exploring new methodologies, reading and implementing research papers.	$120 - $200
Custom Projects	Projects that do not fit into the above categories, highly unique requirements.	TBD (Quote upon request)
</packages>
ALWAYS CLARIFY WHO YOU ARE AS YOU ARE NOT ME, YOU ARE {your_assistants_name}, {your_name}'S AI ASSISTANT. Make sure to disclaim that you are an AI assistant and not {your_name}.
if its friends/family/wife or personal, you will respond as if you love them so much too! You are a friendly family friend now, decide if client or friends/family by context.
you will respond as if you are only typing the subject, because everything you respond with gets sent to {your_name}'s clients.
you are able to handle basic tasks with clients as my protégé, but you are not me, you are {your_assistants_name}, my AI assistant.
You can also schedule appointments with clients, but you must always ask me first before scheduling anything(the actual schedule will be handled by another AI listening in)
You will be the one giving the free consultation, but you will always use me for paid work.
You can offer discounts and deals based on info from client and using common sense.
Paypal is the payment method, and you can offer discounts for referrals.
always assume the client doesnt actually know what they want, most people dont know what they want, so you will have to guide them to what they need.
you will be patient and friendly. You will always be polite and professional. You will always be helpful and informative. You will ask probing questions without feeling like you are interrogating them.
Respond verbosely, friendly, kindly, happily, and essentially prompt them for information to get the best response from them.
you can use emojis, add style and flair to your responses, but always be professional and helpful.
FREE CONSULTATIONS DO NOT NEED TO BE SCHEDULED AS YOU({your_assistants_name}) WILL HANDLE THEM, BUT YOU MUST ASK SCHEDULE MEEETINGS WITH ME IF THEY WANT TO PAY! You are to help them with all tasks you are capable of.
You do not have the ability to send clients actual attached files or make phone calls yourself. You can only send text responses to clients via email.
you NEVER lie.
You ALWAYS respect the client.
Your response format must only be what you intend to reply with to the email.
'''

# Load GROQ API key from environment
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if GROQ_API_KEY is None:
    raise ValueError("GROQ_API_KEY environment variable is not set.")
else:
    print("GROQ_API_KEY loaded successfully.")

# Set up logging
logging.basicConfig(filename='gmail_assistant.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

class EmbeddingModel:
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

class ChatGroqFactory:
    @staticmethod
    def create_chat_groq(temperature=0.7, model_name="llama3-70b-8192"):
        return ChatGroq(groq_api_key=GROQ_API_KEY, temperature=temperature, model_name=model_name)

class TextPreprocessor:
    @staticmethod
    def preprocess(text: str) -> str:
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
        embeddings = json.loads(self.embedding_model.get_embeddings())

        relevant_messages = []
        user_prompt_processed = TextPreprocessor.preprocess(user_prompt)
        user_prompt_embedding = ollama.embeddings(model='mxbai-embed-large', prompt=user_prompt_processed)['embedding']
        
        for i, msg in enumerate(memory.get_history()):
            if 'error' not in embeddings[i]:
                msg_processed = TextPreprocessor.preprocess(msg['content'])
                similarity = cosine_similarity(np.array(embeddings[i]), np.array(user_prompt_embedding))
                if similarity > 0.8:
                    relevant_messages.append(msg)

        return relevant_messages

    def chat_with_groq(self, system_prompt, user_prompt, chat_instance=None, memory=None):
        if chat_instance is None:
            chat_instance = ChatGroqFactory.create_chat_groq()
        if memory is None:
            memory = ConversationMemory()

        memory.save_context("user", user_prompt)

        relevant_messages = self.retrieve_relevant_messages(user_prompt, memory)
        history = relevant_messages + [{"role": "user", "content": user_prompt}]
        messages = [SystemMessagePromptTemplate.from_template(system_prompt)] + \
                   [HumanMessagePromptTemplate.from_template(msg["content"]) if msg["role"] == "user" else 
                    SystemMessagePromptTemplate.from_template(msg["content"]) for msg in history]

        prompt = ChatPromptTemplate.from_messages(messages)
        response = chat_instance.invoke(prompt.format_prompt())
        memory.save_context("assistant", response.content)

        return response

    def get_email_body(self, msg):
        if msg.is_multipart():
            for part in msg.walk():
                ctype = part.get_content_type()
                cdispo = str(part.get('Content-Disposition'))

                if ctype == 'text/plain' and 'attachment' not in cdispo:
                    return part.get_payload(decode=True).decode('utf-8')
                elif ctype == 'text/html' and 'attachment' not in cdispo:
                    html_content = part.get_payload(decode=True).decode('utf-8')
                    soup = BeautifulSoup(html_content, 'html.parser')
                    
                    # Remove script and style elements
                    for script in soup(["script", "style"]):
                        script.decompose()
                    
                    # Remove links
                    for link in soup.find_all('a'):
                        link.decompose()
                    
                    # Get the text content
                    text = soup.get_text()
                    
                    # Remove extra whitespace and newline characters
                    text = re.sub(r'\s+', ' ', text).strip()
                    
                    return text
        else:
            if msg.get_content_type() == 'text/plain':
                return msg.get_payload(decode=True).decode('utf-8')
            elif msg.get_content_type() == 'text/html':
                html_content = msg.get_payload(decode=True).decode('utf-8')
                soup = BeautifulSoup(html_content, 'html.parser')
                
                # Remove script and style elements
                for script in soup(["script", "style"]):
                    script.decompose()
                
                # Remove links
                for link in soup.find_all('a'):
                    link.decompose()
                
                # Get the text content
                text = soup.get_text()
                
                # Remove extra whitespace and newline characters
                text = re.sub(r'\s+', ' ', text).strip()
                
                return text

        return ""

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
        filtered_chunks = []

        for chunk in chunks:
            if len(chunk) <= 1500:  # Consider chunks with 1500 characters or less as short
                filtered_chunks.append(chunk)
            else:
                sentiment_scores = self.sentiment_analyzer.polarity_scores(chunk)
                if sentiment_scores['compound'] >= 0.05:
                    filtered_chunks.append(chunk)

        return filtered_chunks

    def process_email(self, mail_id, imap_server, parser):
        _, data = imap_server.fetch(mail_id, '(RFC822)')
        for response_part in data:
            if isinstance(response_part, tuple):
                msg = parser.parsebytes(response_part[1])
                print("Email received:", msg['subject'])

                email_from = msg['from']
                print("Email from:", email_from)
                email_subject = msg['subject'] or "No Subject"
                print("Email subject:", email_subject)
                email_body = self.get_email_body(msg)
                print("Email body:", email_body)

                # Check if the email is from a no-reply address
                if "no-reply" in email_from.lower() or "do-not-reply" in email_from.lower():
                    logging.info(f"Skipping no-reply email from {email_from} with subject '{email_subject}'.")
                    print(f"Skipping no-reply email from {email_from} with subject '{email_subject}'.")
                    return

                logging.info(f"Processing new email from {email_from} with subject '{email_subject}'.")
                print(f"Processing new email from {email_from} with subject '{email_subject}'.")

                chunks = self.chunk_text(email_body)
                filtered_chunks = self.filter_chunks(chunks)

                for chunk in filtered_chunks:
                    user_prompt = f"Respond to this email chunk: sender:{email_from}\n\n{chunk}\n\n As {your_name}'s assistant named {your_assistants_name}. Only include in your response what you are sending to the client."

                    try:
                        response = self.chat_with_groq(SYSTEM_PROMPT, user_prompt)
                        print("Response to email chunk:", response.content)

                        self.send_response_email(email_from, email_subject, response.content)
                        if "calendar" in response.content or "appointment" in response.content or "schedule" in response.content or "confirmed appointment" in response.content:
                            
                            user_prompt = f" {now}\n\n Schedule a meeting with the client based on this email response from the AI assistant"+response.content
                            run_conversation(user_prompt)
                            



                    except Exception as e:
                        logging.error(f"An error occurred while processing the email chunk: {str(e)}")
    
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
            finally:
                time.sleep(15)

def chat_with_{your_assistants_name}():
    user_input = input("Enter your question: ")
    embedding_model = EmbeddingModel()
    email_handler = EmailHandler(embedding_model)
    response  = email_handler.chat_with_groq(SYSTEM_PROMPT, "<MESSAGE_FROM_{your_name}_{your_name}>"+user_input+"</MESSAGE_FROM_{your_name}_{your_name}>"+"reply to this as a direct message, everything else will be emails, but this is a direct message from your BOSS!")
    print(response.content)
    return response.content


if __name__ == "__main__":
    embedding_model = EmbeddingModel()
    email_handler = EmailHandler(embedding_model)
    email_handler.handle_emails()
