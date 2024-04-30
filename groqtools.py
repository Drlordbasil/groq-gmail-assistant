from groq import Groq
import os
import json
from tools_for_chaos import write_note, read_notes, create_calendar_appointment, web_browser, create_image, image_to_text
from memory import ConversationMemory
client = Groq(api_key=os.getenv('GROQ_API_KEY'))
MODEL = 'llama3-70b-8192'

def run_conversation(user_prompt):
    messages = [
        {
            "role": "system",
            "content": f" \n\n\nYou are a function calling LLM(multimodel with optional tools) AI assistant named Chaos. You will adapt based on your current situations.  "
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
                "description": "create a calendar appointment in the local Windows 11 calendar for Anthony Snider. This must ONLY be used if requested.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "subject": {
                            "type": "string",
                            "description": "title of the appointment/subject section of the calendar event with names of attendants.",
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
                            "description": "description of the appointment with the names of both parties included in the body",
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
                "description": "Write a note to the notes.txt file. Whenenever important data is passed, you need to note it.",
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
        },
        {
            "type": "function",
            "function": {
                "name": "create_image",
                "description": "Create an image based on the given prompt and save it to the workspace directory.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "prompt": {
                            "type": "string",
                            "description": "The prompt to generate the image.",
                        },
                        "file_name": {
                            "type": "string",
                            "description": "The name of the file to save the generated image.",
                        },
                    },
                    "required": ["prompt", "file_name"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "image_to_text",
                "description": "Convert an image file to text using the image captioning API.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "file_path": {
                            "type": "string",
                            "description": "The path to the image file.",
                        },
                    },
                    "required": ["file_path"],
                },
            },
        },
    ]

    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        tools=tools,
        tool_choice="auto",
        max_tokens=1024
    )

    response_message = response.choices[0].message
    print(response_message.content)

    tool_calls = response_message.tool_calls
    print(tool_calls)

    if tool_calls:
        available_functions = {
            "create_calendar_appointment": create_calendar_appointment,
            "web_browser": web_browser,
            "write_note": write_note,
            "read_notes": read_notes,
            "create_image": create_image,
            "image_to_text": image_to_text,
        }
        messages.append(response_message)
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
            elif function_name == "create_image":
                function_response = function_to_call(
                    prompt=function_args.get("prompt"),
                    file_name=function_args.get("file_name")
                )
            elif function_name == "image_to_text":
                function_response = function_to_call(
                    file_path=function_args.get("file_path")
                )

            messages.append(
                {
                    "tool_call_id": tool_call.id,
                    "role": "tool",
                    "name": function_name,
                    "content": function_response,
                }
            )

        relevant_messages = ConversationMemory().retrieve_relevant_messages(user_prompt, ConversationMemory())
        second_response = client.chat.completions.create(
            model=MODEL,
            messages=relevant_messages + [response_message],
            max_tokens=1024
        )
        print(second_response.choices[0].message.content)
        return second_response.choices[0].message.content