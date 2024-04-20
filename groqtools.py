from groq import Groq
import os
import json
import datetime

client = Groq(api_key=os.getenv('GROQ_API_KEY'))
MODEL = 'llama3-70b-8192'

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


def run_conversation(user_prompt):
    # Step 1: send the conversation and available functions to the model
    messages = [
        {
            "role": "system",
            "content": f" \n\n\nYou are a function calling LLM that uses the data extracted from the create_calendar_appointment function to set appointments for Anthony Snider Locally. Include all details in the appointment."
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
                "description": "create a calendar appointment in the local Windows 11 calendar for Anthony Snider.",
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
    tool_calls = response_message.tool_calls
    # Step 2: check if the model wanted to call a function
    if tool_calls:
        # Step 3: call the function
        available_functions = {
            "create_calendar_appointment": create_calendar_appointment,
        }
        messages.append(response_message)  # extend conversation with assistant's reply
        # Step 4: send the info for each function call and function response to the model
        for tool_call in tool_calls:
            function_name = tool_call.function.name
            function_to_call = available_functions[function_name]
            function_args = json.loads(tool_call.function.arguments)
            function_response = function_to_call(
                subject=function_args.get("subject"),
                start_time=function_args.get("start_time"),
                end_time=function_args.get("end_time"),
                location=function_args.get("location"),
                body=function_args.get("body")
            )
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
        return second_response.choices[0].message.content


