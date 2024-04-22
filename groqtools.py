from groq import Groq
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

client = Groq(api_key=os.getenv('GROQ_API_KEY'))
MODEL = 'llama3-70b-8192'

def write_note(content):
    """Write a note to the notes.txt file."""
    with open("notes.txt", "a") as file:
        file.write(content + "\n")
    return "Note written successfully."

def read_notes():
    """Read notes from the notes.txt file."""
    try:
        with open("notes.txt", "r") as file:
            notes = file.read()
        return notes
    except FileNotFoundError:
        return "No notes found."

def create_calendar_appointment(subject, start_time, end_time, location, body):
    """Create an .ics file for an appointment and open it with the default calendar application."""
    start_time = datetime.datetime.strptime(start_time, '%Y-%m-%d %H:%M:%S')
    end_time = datetime.datetime.strptime(end_time, '%Y-%m-%d %H:%M:%S')
    
    # Format start and end times for iCalendar
    start_time_formatted = start_time.strftime('%Y%m%dT%H%M%S')
    end_time_formatted = end_time.strftime('%Y%m%dT%H%M%S')
    
    # Create iCalendar content
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

    # Define file path for .ics file
    ics_file_path = os.path.join(os.getenv('TEMP', '.'), f"{subject.replace(' ', '_')}.ics")

    # Write the iCalendar file
    with open(ics_file_path, 'w') as file:
        file.write(ics_content)
    
    # Open the file with the default application
    os.startfile(ics_file_path)
    
    return f"Appointment created: {subject} from {start_time} to {end_time} at {location}. " \
           f"The .ics file has been opened with the default calendar application."

def web_browser(query):
    """Perform a web search using Selenium and return the relevant information."""
    # Set up Chrome options
    chrome_options = Options()
    chrome_options.add_argument("--headless")  # Run Chrome in headless mode

    # Auto-detect and install the appropriate ChromeDriver
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=chrome_options)

    try:
        # Perform the web search
        search_url = f"https://www.google.com/search?q={query}"
        driver.get(search_url)

        # Wait for the search results to load
        wait = WebDriverWait(driver, 10)
        search_results = wait.until(EC.presence_of_element_located((By.ID, "search")))

        # Extract the relevant information from the search results
        result_snippets = search_results.find_elements(By.CSS_SELECTOR, ".g .s .st")
        relevant_info = "\n".join([snippet.text for snippet in result_snippets])

        return relevant_info

    finally:
        # Close the browser
        driver.quit()

def run_conversation(user_prompt):
    # Step 1: send the conversation and available functions to the model
    messages = [
        {
            "role": "system",
            "content": f" \n\n\nYou are a function calling LLM(multimodel with optional tools) AI assistant.  "
        },
        {
            "role": "user",
            "content": user_prompt,
        }
    ]
    tools = [
        {
            "type": "function",
            "function": {
                "name": "create_calendar_appointment",
                "description": "create a calendar appointment in the local Windows 11 calendar . This must ONLY be used if requested.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "subject": {
                            "type": "string",
                            "description": "title of the appointment/subject section of the calendar event",
                        },
                        "start_time": {
                            "type": "string",
                            "description": "start time of the appointment in the format 'YYYY-MM-DD HH:MM:SS'",
                        },
                        "end_time": {
                            "type": "string",
                            "description": "end time of the appointment in the format 'YYYY-MM-DD HH:MM:SS'",
                        },
                        "location": {
                            "type": "string",
                            "description": "location of the appointment",
                        },
                        "body": {
                            "type": "string",
                            "description": "description of the appointment",
                        },
                    },
                    "required": ["subject", "start_time", "end_time", "location", "body"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "web_browser",
                "description": "Perform a web search using Selenium and return the relevant information/context from google.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The google search query to perform.",
                        },
                    },
                    "required": ["query"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "write_note",
                "description": "Write a note to the notes.txt file.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "content": {
                            "type": "string",
                            "description": "The content of the note to write.",
                        },
                    },
                    "required": ["content"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "read_notes",
                "description": "Read notes from the notes.txt file.",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": [],
                },
            },
        }
    ]

    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        tools=tools,
        tool_choice="auto",
        max_tokens=4096
    )

    response_message = response.choices[0].message
    print(response_message.content)

    tool_calls = response_message.tool_calls
    print(tool_calls)
    # Step 2: check if the model wanted to call a function
    if tool_calls:
        # Step 3: call the function
        available_functions = {
            "create_calendar_appointment": create_calendar_appointment,
            "web_browser": web_browser,
            "write_note": write_note,
            "read_notes": read_notes,
        }
        messages.append(response_message)  # extend conversation with assistant's reply
        # Step 4: send the info for each function call and function response to the model
        for tool_call in tool_calls:
            function_name = tool_call.function.name
            function_to_call = available_functions[function_name]
            function_args = json.loads(tool_call.function.arguments)
            if function_name == "create_calendar_appointment":
                function_response = function_to_call(
                    subject=function_args.get("subject"),
                    start_time=function_args.get("start_time"),
                    end_time=function_args.get("end_time"),
                    location=function_args.get("location"),
                    body=function_args.get("body")
                )
            elif function_name == "web_browser":
                function_response = function_to_call(
                    query=function_args.get("query")
                )
            elif function_name == "write_note":
                function_response = function_to_call(
                    content=function_args.get("content")
                )
            elif function_name == "read_notes":
                function_response = function_to_call()
            messages.append(
                {
                    "tool_call_id": tool_call.id,
                    "role": "tool",
                    "name": function_name,
                    "content": function_response,
                }
            )  # extend conversation with function response
        second_response = client.chat.completions.create(
            model=MODEL,
            messages=messages
        )  # get a new response from the model where it can see the function response
        print(second_response.choices[0].message.content)
        return second_response.choices[0].message.content
