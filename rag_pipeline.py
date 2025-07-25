from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()  # Load .env file into environment

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def generate_response(user_input: str, detected_emotion: str) -> str:
    messages = [
        {
            "role": "system",
            "content": f"You are a compassionate mental health assistant. The user's emotional state is: {detected_emotion}."
        },
        {
            "role": "user",
            "content": user_input
        }
    ]

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages
    )

    return response.choices[0].message.content.strip()
