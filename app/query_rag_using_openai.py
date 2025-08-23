import os
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables
load_dotenv()

openai_client = OpenAI(
    api_key=os.getenv('OPENAI_API_KEY'),
    base_url=os.getenv('OPENAI_BASE_URL', 'https://api.opentyphoon.ai/v1')
)

prompt = "สวัสดี"

model = "typhoon-v2.1-12b-instruct" # Specify the model you're using with vLLM

# Non-Streaming Response
def response(prompt):
    try:
        response = openai_client.chat.completions.create(
            model=model,
            messages=[
            {"role": "system", "content": "You are a helpful assistant that translates English to Thai."},
            {"role": "user", "content": f"{prompt}"}
            ],
            max_tokens=512,
            temperature=0.7,
            top_p=0.8,
            stop=["<|im_end|>"]
        )
        print("Generated Text:", response.choices[0].message.content)
    except Exception as e:
        print("Error:", str(e))
    
# Example usage of non-streaming version
print("Non-streaming response:")
response(prompt)

# Streaming version
def stream_response(prompt):
    try:
        response = openai_client.chat.completions.create(
            model=model,
            messages=[
            {"role": "system", "content": "You are a helpful assistant that translates English to Thai."},
            {"role": "user", "content": f"{prompt}"}
            ],
            max_tokens=512,
            temperature=0.7,
            top_p=0.8,
            stop=["<|im_end|>"],
            stream=True  # Enable streaming
        )
        
        for chunk in response:
            if chunk.choices[0].delta.content is not None:
                print(chunk.choices[0].delta.content, end="", flush=True)
                # print()  # Print a newline at the end
    except Exception as e:
        print("Error:", str(e))

# Example usage of streaming version
print("Streaming response:")
stream_response(prompt)

