import datetime
import imaplib
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email import policy
from email.parser import BytesParser
import json
import os
import time
import logging
import nltk
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer
import numpy as np

from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
import ollama
from config import USER, APP_PASSWORD, IMAP_URL, SMTP_URL, SMTP_PORT, SYSTEM_PROMPT
from groqtools import run_conversation
from tools_for_chaos import get_current_time_formatted
from nltk.stem import WordNetLemmatizer
from memory import ConversationMemory, EmbeddingModel, preprocess_text, cosine_similarity
from groqchat import ChatGroqFactory
from email_utils import get_email_body, chunk_text, clean_email_body, get_client_name, get_email_attachments, attach_files_to_email

now = get_current_time_formatted()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if GROQ_API_KEY is None:
    raise ValueError("GROQ_API_KEY environment variable is not set.")

logging.basicConfig(filename='gmail_assistant.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

class EmailHandler:
    def __init__(self, embedding_model: EmbeddingModel):
        self.embedding_model = embedding_model
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
        self.memory = ConversationMemory()

    def chat_with_groq(self, system_prompt, user_prompt, chat_instance=None, tools=None):
        chat_instance = chat_instance or ChatGroqFactory.create_chat_groq()
        self.memory.save_context("user", user_prompt)

        relevant_messages = self.memory.retrieve_relevant_messages(user_prompt, self.memory)
        history = relevant_messages + [{"role": "user", "content": user_prompt}]
        messages = [SystemMessagePromptTemplate.from_template(system_prompt)] + [
            HumanMessagePromptTemplate.from_template(msg["content"]) for msg in history
        ]

        prompt = ChatPromptTemplate.from_messages(messages)
        response = chat_instance.invoke(prompt.format_prompt(), tools=tools)
        self.memory.save_context("assistant", response.content)

        return response

    def filter_chunks(self, chunks):
        return [
            chunk for chunk in chunks
            if len(chunk) <= 1500 or self.sentiment_analyzer.polarity_scores(chunk)['compound'] >= 0.05
        ]

    def process_email(self, mail_id, imap_server, parser):
        _, data = imap_server.fetch(mail_id, '(RFC822)')
        for response_part in data:
            if isinstance(response_part, tuple):
                msg = parser.parsebytes(response_part[1])
                email_from = msg['from']
                email_subject = msg['subject'] or "No Subject"
                email_body = get_email_body(msg)
                email_attachments = get_email_attachments(msg)

                print(f"Received email from: {email_from}")
                print(f"Email subject: {email_subject}")
                print(f"Email body: {email_body}")
                print(f"Email attachments: {[attachment[0] for attachment in email_attachments]}")

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

                cleaned_email_body = clean_email_body(email_body)
                print(f"Cleaned email body: {cleaned_email_body}")

                filtered_chunks = self.filter_chunks(chunk_text(cleaned_email_body))

                client_name = get_client_name(email_from)

                for chunk in filtered_chunks:
                    relevant_messages = self.memory.retrieve_relevant_messages(chunk, self.memory)

                    user_prompt_for_tool_use =f"""
                    As Chaos, analyze this email and figure out what tools may be needed,
                      if none, just say none or leave a note with the note tool.
                      Cleaned email body = {cleaned_email_body}

                    Sender: {email_from}

                    Client Name: {client_name}

                    Email Content:
                        {relevant_messages if relevant_messages else chunk}

                    !!!IMPORTANT!!! YOU ARE ONLY THE TOOL STAGE AND WILL NOT BE REPLYING YET.
                        """

                    user_prompt_for_email_reply = f" {now}\n\nAs Chaos, respond to this email chunk. Use any tools necessary to formulate your response.\n\nSender: {email_from}\n\nClient Name: {client_name}\n\nEmail Content:\n{chunk}\n\nOnly include in your response what you are sending to the client. NOTE: you are replying to an email directly. DO NOT SEND ANYTHING THAT WILL LOOK LIKE YOU FAILED TO RESPOND! SEND YOUR RESPONSE TO THE CLIENT ONLY AS YOU ARE DIRECTLY REPLYING VIA EMAIL!"
                    relevant_messages = self.memory.retrieve_relevant_messages(chunk, self.memory)

                    relevant_content = [msg['content'] for msg in relevant_messages]

                    user_prompt_for_analysis = f"Analyze the email chunk and determine if a response is required. If a response is required, reply to the email chunk. If no response is required, leave a note with the note tool. Cleaned email body = {cleaned_email_body}\n\nSender: {email_from}\n\nClient Name: {client_name}\n\nEmail Content:\n{relevant_content if relevant_content else chunk} RESPOND SIMPLY WITH YES OR NO!!!"

                    try:
                        response = self.chat_with_groq(SYSTEM_PROMPT, user_prompt_for_analysis + "\n" + chunk)
                        print(f"Analysis result: {response.content}")

                        if "YES" in response.content.strip().upper():
                            user_prompt_for_email_reply = f" {now}\n\nAs Chaos, respond to this email chunk. Use any tools necessary to formulate your response.\n\nSender: {email_from}\n\nClient Name: {client_name}\n\nEmail Content:\n{chunk}\n\nOnly include in your response what you are sending to the client. NOTE: you are replying to an email directly. DO NOT SEND ANYTHING THAT WILL LOOK LIKE YOU FAILED TO RESPOND! SEND YOUR RESPONSE TO THE CLIENT ONLY AS YOU ARE DIRECTLY REPLYING VIA EMAIL!"

                            print(f"Using tools to formulate a response...")
                            tool_response = run_conversation(user_prompt_for_tool_use)

                            user_prompt_for_thinking =f"""
                                based on the email content and the tool response below, formulate a response to the email chunk and get ready to reply to an email.
                                Cleaned email body = {cleaned_email_body}
                                tool response: {tool_response}
                                Sender: {email_from}
                                Client Name: {client_name}
                                Email Content:  {relevant_messages if relevant_messages else chunk}

                                You arent replying yet, just preparing to reply as a think before you speak stage.
                            """

                            print(f"Tool response: {tool_response}")
                            print(f"Formulating response...")

                            thoughts = self.chat_with_groq(SYSTEM_PROMPT, user_prompt_for_thinking + "\nTool Usage Summary:\n" + tool_response)
                            thoughts = thoughts.content

                            print(f"Thinking stage: {response.content}")

                            # Construct the prompt with structured context
                            user_prompt_for_email_reply = f"""
                            You are about to reply to the email.

                            Your thoughts: {thoughts}

                            The email body: {cleaned_email_body}

                            The client's name and email: {client_name} {email_from}

                            Attachments: {', '.join([attachment[0] for attachment in email_attachments]) if email_attachments else 'No attachments'}

                            Relevant info from past emails from {client_name}: {' '.join([msg['content'] for msg in relevant_messages])}
                                ALWAYS INCLUDE OUTPUT FROM TOOLS USED IN YOUR RESPONSE TO THE CLIENT AS THEY WONT SEE IT OTHERWISE!
                            Please send your reply as your response ONLY. Include all needed info to the client. Keep in mind that your memory and chat history work differently, but send your response to the client only.
                            This means if you say code is in response, to actually include it to the response of the email.
                            """

                            response = self.chat_with_groq(SYSTEM_PROMPT, user_prompt_for_email_reply)

                            print(f"Final response: {response.content}")

                            self.send_response_email(email_from, email_subject, response.content, email_attachments)

                        else:
                            user_prompt_for_tool_use =f"""
                            As Chaos, analyze this email and figure out what tools may be needed,
                            if none, just say none or leave a note with the note tool.
                            Cleaned email body = {cleaned_email_body}

                            Sender: {email_from}

                            Client Name: {client_name}

                            Email Content:
                                {relevant_messages if relevant_messages else chunk}

                            !!!IMPORTANT!!! YOU ARE ONLY THE TOOL STAGE AND WILL NOT BE REPLYING YET.
                                """

                            tool_response = run_conversation(user_prompt_for_tool_use)

                            logging.info(f"No response required for email chunk from {email_from} with subject '{email_subject}'. tool response: {tool_response}")
                            print(f"No response required for email chunk from {email_from} with subject '{email_subject}'.  tool response: {tool_response}")

                    except Exception as e:
                        logging.error(f"An error occurred while processing the email chunk: {str(e)}")
                        print(f"An error occurred while processing the email chunk: {str(e)}")
                        continue

    def send_response_email(self, email_from, email_subject, response_content, attachments=None):
        smtp_server = smtplib.SMTP(SMTP_URL, SMTP_PORT)
        smtp_server.starttls()
        smtp_server.login(USER, APP_PASSWORD)

        message = MIMEMultipart()
        message['From'] = USER
        message['To'] = email_from
        message['Subject'] = "Re: " + email_subject
        message.attach(MIMEText(response_content))

        if attachments:
            message = attach_files_to_email(message, attachments)

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
