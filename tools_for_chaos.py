import os
import json
import datetime
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
from datetime import datetime
import requests
import io
from PIL import Image

API_URL_IMAGE_GEN = "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-xl-base-1.0"
API_URL_IMAGE_CAPTION = "https://api-inference.huggingface.co/models/nlpconnect/vit-gpt2-image-captioning"
headers = {"Authorization": "Bearer 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"}

def get_current_time_formatted():
    now = datetime.now()
    time_string = now.strftime("%Y-%m-%d %H:%M:%S")
    return f"{time_string} is the current date and time"

def write_note(content):
    with open("notes.txt", "a") as file:
        file.write(content + "\n")
    return "Note written successfully."

def read_notes():
    try:
        with open("notes.txt", "r") as file:
            notes = file.read()
        return notes
    except FileNotFoundError:
        return "No notes found."

def create_calendar_appointment(subject, start_time, end_time, location, body):
    start_time = datetime.strptime(start_time, '%Y-%m-%d %H:%M:%S')
    end_time = datetime.strptime(end_time, '%Y-%m-%d %H:%M:%S')

    start_time_formatted = start_time.strftime('%Y%m%dT%H%M%S')
    end_time_formatted = end_time.strftime('%Y%m%dT%H%M%S')

    ics_content = (
        'BEGIN:VCALENDAR\n'
        'VERSION:2.0\n'
        'BEGIN:VEVENT\n'
        f'SUMMARY:{subject}\n'
        f'DTSTART;VALUE=DATE-TIME:{start_time_formatted}\n'
        f'DTEND;VALUE=DATE-TIME:{end_time_formatted}\n'
        f'LOCATION:{location}\n'
        f'DESCRIPTION:{body}\n'
        'END:VEVENT\n'
        'END:VCALENDAR'
    )

    ics_file_path = os.path.join(os.getenv('TEMP', '.'), f"{subject.replace(' ', '_')}.ics")

    with open(ics_file_path, 'w') as file:
        file.write(ics_content)

    os.startfile(ics_file_path)

    return f"Appointment created: {subject} from {start_time} to {end_time} at {location}. " \
           f"The .ics file has been opened with the default calendar application."

def web_browser(query):
    chrome_options = Options()
    chrome_options.add_argument("--headless")

    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=chrome_options)

    try:
        search_url = f"https://www.google.com/search?q={query}"
        driver.get(search_url)

        wait = WebDriverWait(driver, 10)
        search_results = wait.until(EC.presence_of_element_located((By.ID, "search")))

        result_snippets = search_results.find_elements(By.CSS_SELECTOR, ".g .s .st")
        relevant_info = "\n".join([snippet.text for snippet in result_snippets])

        return relevant_info

    finally:
        driver.quit()

def create_image(prompt, file_name):
    payload = {
        "inputs": prompt,
    }

    response = requests.post(API_URL_IMAGE_GEN, headers=headers, json=payload)
    image_bytes = response.content

    image = Image.open(io.BytesIO(image_bytes))
    file_path = os.path.join("workspace", file_name)
    image.save(file_path)

    return file_path

def image_to_text(file_path):
    with open(file_path, "rb") as f:
        data = f.read()

    response = requests.post(API_URL_IMAGE_CAPTION, headers=headers, data=data)
    result = response.json()

    if "error" in result:
        return f"Error: {result['error']}"
    else:
        return result[0]["generated_text"]