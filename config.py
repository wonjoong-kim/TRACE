import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

FULL_PROMPT_FORMAT = '''<|begin_of_text|><|start_header_id|>system<|end_header_id|>{SYSTEM_PROMPT}<|eot_id|><|start_header_id|>user<|end_header_id|>{USER_PROMPT}<|eot_id|><|start_header_id|>assistant<|end_header_id|>'''

PREDEFINED_UNAVAILABLE_SET = {'FastOCR', 'FastCalculator', 'ImageDescriptor', 'WebSearch'}