import openai
from config import get_openai_client, MODEL_CONFIG

# Initialize OpenAI client
openai_client = get_openai_client()

prompt = "สวัสดี"

model = "typhoon-v2-70b-instruct" # Specify the model you're using with vLLM

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

