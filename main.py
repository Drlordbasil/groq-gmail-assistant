import imaplib
import smtplib
from email.mime.text import MIMEText
import email
from email import policy
from email.parser import BytesParser
import json
import os
import time

from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_groq import ChatGroq

# User configuration
USER = ''
APP_PASSWORD = ''  # App password generated from Google
IMAP_URL = 'imap.gmail.com'
SMTP_URL = 'smtp.gmail.com'
SMTP_PORT = 587

# Load GROQ API key from environment
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if GROQ_API_KEY is None:
    raise ValueError("GROQ_API_KEY environment variable is not set.")
else:
    print("GROQ_API_KEY loaded successfully.")

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

def create_chat_groq(temperature=0, model_name="mixtral-8x7b-32768"):
    return ChatGroq(groq_api_key=GROQ_API_KEY, temperature=temperature, model_name=model_name)

def chat_with_groq(system_prompt, user_prompt, chat_instance=None, memory=None):
    if chat_instance is None:
        chat_instance = create_chat_groq()
    if memory is None:
        memory = ConversationMemory()

    memory.save_context("user", "Email you already responded to:<email_for_anthony_you_already_responded_to>"+user_prompt+"</email_for_anthony_you_already_responded_to>")

    history = memory.get_history()
    messages = [SystemMessagePromptTemplate.from_template(system_prompt)] + \
               [HumanMessagePromptTemplate.from_template(msg["content"]) if msg["role"] == "user" else 
                SystemMessagePromptTemplate.from_template(msg["content"]) for msg in history] + \
               [HumanMessagePromptTemplate.from_template(user_prompt)]

    prompt = ChatPromptTemplate.from_messages(messages)
    response = chat_instance.invoke(prompt.format_prompt())
    
    memory.save_context("assistant", "Your response to the email earlier that you already sent:<email_response_from_you>"+response.content+"</email_response_from_you>")

    return response

def get_email_body(msg):
    if msg.is_multipart():
        for part in msg.walk():
            ctype = part.get_content_type()
            cdispo = str(part.get('Content-Disposition'))

            # Skip any text/plain (txt) attachments
            if ctype == 'text/plain' and 'attachment' not in cdispo:
                return part.get_payload(decode=True).decode('utf-8')  # decode
    else:
        return msg.get_payload(decode=True).decode('utf-8')

def handle_emails():
    parser = BytesParser(policy=policy.default)
    
    while True:
        try:
            imap_server = imaplib.IMAP4_SSL(IMAP_URL)
            imap_server.login(USER, APP_PASSWORD)
            imap_server.select('inbox')

            # Search for all unread messages
            _, data = imap_server.search(None, 'UNSEEN')
            mail_ids = data[0].split()

            if not mail_ids or mail_ids[0] == b'':
                print("No new emails. Checking again in 60 seconds.")
                imap_server.logout()
                time.sleep(60)  # Wait 60 seconds before checking again
                continue

            for mail_id in mail_ids:
                _, data = imap_server.fetch(mail_id, '(RFC822)')
                for response_part in data:
                    if isinstance(response_part, tuple):
                        msg = parser.parsebytes(response_part[1])
                        email_from = msg['from']
                        email_subject = msg['subject'] or "No Subject"
                        email_body = get_email_body(msg)

                        print(f"Processing new email from {email_from} with subject '{email_subject}'.")

                        # Generate response using GROQ AI
                        system_prompt = "You are a helpful assistant named Chaos working for Anthony Snider. Respond informatively and courteously."
                        user_prompt = f"ONLY RESPOND TO THIS EMAIL FOR ANTHONY SNIDER AND YOU ARENT THEIR ASSISTANT!:\n\n Email to Anthony Snider: \n\n{email_body} your reply as Anthony Snider's assistant replying to his emails:"
                        response = chat_with_groq(system_prompt, user_prompt)

                        # Send the auto-reply
                        smtp_server = smtplib.SMTP(SMTP_URL, SMTP_PORT)
                        smtp_server.starttls()
                        smtp_server.login(USER, APP_PASSWORD)

                        message = MIMEText(response.content)
                        message['From'] = USER
                        message['To'] = email_from
                        message['Subject'] = "Re: " + email_subject

                        smtp_server.sendmail(USER, [email_from], message.as_string())
                        smtp_server.quit()

                        print("Auto-reply sent to:", email_from)

            imap_server.logout()
        except Exception as e:
            print(f"An error occurred: {e}")
        finally:
            time.sleep(60)  # Wait 60 seconds before checking again

handle_emails()
