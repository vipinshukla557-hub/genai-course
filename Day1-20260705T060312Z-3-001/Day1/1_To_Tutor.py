from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv("OPEN_API_KEY")

client = OpenAI(api_key=api_key)

print("client created successfully")

messages = [
    {
        "role": "system",
        "content": "You are a helpful assistant. Keep answers short and simple."
    },
    {
        "role": "user",
        "content": "What is artificial intelligence? Explain in 2 lines."
    }
]

print("messages built let's go to the actual call")

print("This is now the sstart of the api call")

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages= messages
)

print("This is ai's response")

answer = response.choices[0].message.content
print("ai says")
print(answer)

print()

print("this is the raw response")
print(response)

print()

# JSON

import json

# json.dumps() #it converts output into nicely formatted JSON string

# json.loads() #converts a JSON string back into a python dictionary


student = {"name":"Sunidhi", "city":"Pune","course":"Genai"}

print(student)

print("Pretty printing")

print(json.dumps(student, indent=2))

print()

print("api response in better readble format")

response_dict = response.model_dump()
print(json.dumps(response_dict, indent=2))
