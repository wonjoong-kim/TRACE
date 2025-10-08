import openai
from anthropic import Anthropic
from together import Together
from tenacity import retry, stop_after_attempt, wait_exponential
import os
import config


try:
    if config.OPENAI_API_KEY:
        openai.api_key = config.OPENAI_API_KEY
    if config.TOGETHER_API_KEY:
        os.environ['TOGETHER_API_KEY'] = config.TOGETHER_API_KEY
        together_client = Together()
    else:
        together_client = None
    if config.ANTHROPIC_API_KEY:
        claude_client = Anthropic(api_key=config.ANTHROPIC_API_KEY)
    else:
        claude_client = None
except Exception as e:
    print(f"API Client initialization error: {e}")

@retry(
    wait=wait_exponential(multiplier=1, min=2, max=60), 
    stop=stop_after_attempt(5), 
    retry_error_callback=lambda retry_state: None 
)
def generate_response(llm_model: str, system_prompt: str, user_prompt: str) -> str:
    try:
        if llm_model == 'gpt-4.1' or llm_model == 'o3-mini':
            if not openai.api_key:
                raise ValueError("OpenAI API key is not set.")
            response = openai.chat.completions.create(
                model=llm_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ]
            )
            return response.choices[0].message.content.strip()

        elif llm_model == 'claude':
            resp = claude_client.messages.create(
                model="claude-sonnet-4-20250514", 
                max_tokens=4096,                 
                temperature=0.3,                 
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}]
            )
            return resp.content[0].text

        else: 
            if not together_client:
                 raise ValueError("Together client is not initialized.")
            full_prompt = config.FULL_PROMPT_FORMAT.replace('{SYSTEM_PROMPT}', system_prompt).replace('{USER_PROMPT}', user_prompt)
            response = together_client.chat.completions.create(
                model=llm_model,  
                messages=[{"role": "user", "content": full_prompt}],
                max_tokens=4096, 
                temperature=0,
            )
            return response.choices[0].message.content.strip()

    except Exception as e:
        print(f"Error calling LLM API for model {llm_model}: {e}")
        return f"ERROR: {e}"