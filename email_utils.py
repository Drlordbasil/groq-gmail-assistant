from bs4 import BeautifulSoup
import re
from nltk.tokenize import sent_tokenize
import base64
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.application import MIMEApplication

def get_email_body(msg):
    if msg.is_multipart():
        for part in msg.walk():
            content_type = part.get_content_type()
            if content_type == 'text/plain' and 'attachment' not in str(part.get('Content-Disposition')):
                return part.get_payload(decode=True).decode('utf-8')
            elif content_type == 'text/html' and 'attachment' not in str(part.get('Content-Disposition')):
                return parse_html_content(part.get_payload(decode=True).decode('utf-8'))
    else:
        content_type = msg.get_content_type()
        if content_type == 'text/plain':
            return msg.get_payload(decode=True).decode('utf-8')
        elif content_type == 'text/html':
            return parse_html_content(msg.get_payload(decode=True).decode('utf-8'))

    return ""

def parse_html_content(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')
    for element in soup(["script", "style"]):
        element.decompose()
    for link in soup.find_all('a'):
        link.decompose()
    text = soup.get_text()
    return re.sub(r'\s+', ' ', text).strip()

def chunk_text(text, chunk_size=4096):
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

def clean_email_body(email_body):
    lines = email_body.split("\n")
    cleaned_lines = []

    for line in lines:
        if line.startswith("From:") or line.startswith("Sent:") or line.startswith("To:") or line.startswith("Subject:"):
            break
        cleaned_lines.append(line)

    return "\n".join(cleaned_lines).strip()

def get_client_name(email_from):
    if '<' in email_from:
        client_name = email_from.split('<')[0].strip()
    else:
        client_name = email_from

    return client_name

def get_email_attachments(msg):
    attachments = []
    for part in msg.walk():
        if part.get_content_maintype() == 'multipart':
            continue
        if part.get('Content-Disposition') is None:
            continue
        file_name = part.get_filename()
        if bool(file_name):
            file_data = part.get_payload(decode=True)
            attachments.append((file_name, file_data))
    return attachments

def attach_files_to_email(message, file_paths):
    for file_path in file_paths:
        with open(file_path, "rb") as attachment:
            part = MIMEApplication(attachment.read(), _subtype="txt")
            part.add_header('Content-Disposition', 'attachment', filename=file_path.split("/")[-1])
            message.attach(part)
    return message