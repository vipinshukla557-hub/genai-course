from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

api_key = os.getenv("OPEN_API_KEY")

client = OpenAI(api_key=api_key)

print("Client created successfully")

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
print("messages built lets go to the actual call")
print("this is now startt of api call")

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=messages   
)
answer = response.choices[0].message.content
print("API call completed successfully")
print("Answer from the model:", answer)

print("this is raw response from the model")
print(response)
print()

# JSON
import json
# json.dumps() #it converts output into nicely formatted JSON string
 
# json.loads() #converts a JSON string back into a python dictionary

student={"name": "John", "age": 20, "courses": "Math"}
print(student)


response_dict=response.model_dump() #converts the response into a python dictionary
print(json.dumps(response_dict, indent=2))

#