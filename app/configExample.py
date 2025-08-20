import os
from openai import OpenAI

# OpenAI/Typhoon API Configuration
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY', 'your-api-key')
OPENAI_BASE_URL = os.environ.get('OPENAI_BASE_URL', 'https://api.opentyphoon.ai/v1')
OPENAI_MODEL = os.environ.get('OPENAI_MODEL', 'typhoon-v2-70b-instruct')

# Initialize OpenAI client
def get_openai_client():
    return OpenAI(
        api_key=OPENAI_API_KEY,
        base_url=OPENAI_BASE_URL
    )

# Model configuration
MODEL_CONFIG = {
    'model': OPENAI_MODEL,
    'max_tokens': 512,
    'temperature': 0.7,
    'top_p': 0.8,
    'stop': ["<|im_end|>"]
}