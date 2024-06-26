import os
import json
import datetime

from datetime import datetime
from tool_stuff.web_browser import WebResearchTool

web_research_tool = WebResearchTool()

def get_current_time_formatted():
    # Get the current date and time
    now = datetime.now()

    # Format the date and time as a string
    time_string = now.strftime("%Y-%m-%d %H:%M:%S")

    return f"{time_string} is the current date and time"


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
    start_time = datetime.strptime(start_time, '%Y-%m-%d %H:%M:%S')
    end_time = datetime.strptime(end_time, '%Y-%m-%d %H:%M:%S')
    
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
    """Perform a web search using Selenium and return the relevant information/context from Google."""
    return web_research_tool.web_research(query)
