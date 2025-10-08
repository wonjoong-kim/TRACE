from together import Together
import json, os, openai, argparse
import openai
import csv
from anthropic import Anthropic
from tenacity import retry, stop_after_attempt, wait_exponential
import re

openai.api_key = "YOUR_OPENAI_API_KEY"
os.environ['TOGETHER_API_KEY'] = 'YOUR_TOGETHER_API_KEY'
client = Together()
claude_client = Anthropic(api_key="YOUR_ANTHROPIC_API_KEY")
full_prompt_format = '''<|begin_of_text|><|start_header_id|>system<|end_header_id|>{SYSTEM_PROMPT}<|eot_id|><|start_header_id|>user<|end_header_id|>{USER_PROMPT}<|eot_id|><|start_header_id|>assistant<|end_header_id|>'''

parser = argparse.ArgumentParser()
parser.add_argument('--llm', default = 'gpt-4.1', choices= [
    'gpt-4.1',
    'o3-mini',
    'claude',
    'meta-llama/Llama-3.3-70B-Instruct-Turbo',
    'meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo'])
args, _ = parser.parse_known_args()


@retry(
    wait=wait_exponential(multiplier=1, min=2, max=60), 
    stop=stop_after_attempt(5), 
    retry_error_callback=lambda retry_state: None 
)
def gen_answer(system_prompt: str, user_prompt: str) -> str:
    try:
        if args.llm =='gpt-4.1':
            if not openai.api_key:
                raise ValueError("OpenAI API key is not set.")
            response = openai.chat.completions.create(
                model= args.llm,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=10000,
                temperature=0.1,
            )
            return response.choices[0].message.content.strip()
        
        if args.llm =='o3-mini':
            if not openai.api_key:
                raise ValueError("OpenAI API key is not set.")
            response = openai.chat.completions.create(
                model= args.llm,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
            )
            return response.choices[0].message.content.strip()
        elif args.llm == 'claude':
            resp = claude_client.messages.create(
            model="claude-sonnet-4-20250514", 
            max_tokens=4096,                 
            temperature=0.3,                 
            system = system_prompt,
            messages=[{"role": "user", "content": user_prompt}])
            return resp.content[0].text
        else:
            if not client:
                 raise ValueError("LLM client is not initialized for the specified model.")
            response = client.chat.completions.create(
                model=args.llm,  
                messages=[{"role": "user", "content": full_prompt_format.replace('{SYSTEM_PROMPT}', system_prompt).replace('{USER_PROMPT}', user_prompt)}],
                max_tokens=4096, 
                temperature=0,
            )
            return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error calling LLM API: {e}")
        return f"ERROR: {e}"


def create_inefficiency_prompt(query, plan):
    system_prompt = """
    You are an expert AI assistant specializing in evaluating the plans of other AI agents. 
    Your task is to identify and flag any inefficient steps within a given plan designed to address a user's request.

    A step is considered **inefficient** if:
    1.  It is redundant or unnecessary to fulfill the user's request.
    2.  There is a more direct or appropriate tool available that could achieve the same outcome more effectively.

    Please review the user's request and the proposed plan, then identify the IDs of all inefficient steps.
    """

    plan_str = json.dumps(plan, indent=4)

    user_prompt = f"""
    Here is the user's request and the execution plan. Please identify all inefficient steps in the plan.

    **User Request:**
    "{query}"

    **Plan:**
    {plan_str}

    Provide your response ONLY in the form of a JSON object. The object should have a single key, "inefficient_steps", which contains a list of integer IDs for the steps you have identified as inefficient. If you find no inefficient steps, return an empty list.

    Example of a valid response:
    {{
        "inefficient_steps": [0, 2]
    }}
    
    Another example (if none are inefficient):
    {{
        "inefficient_steps": []
    }}
    """
    return system_prompt, user_prompt


def trace(gt_filepath, eval_filepath):

    with open(gt_filepath, 'r', encoding='utf-8') as f:
        gt_data = json.load(f)
    with open(eval_filepath, 'r', encoding='utf-8') as f:
        eval_data = json.load(f)

    metrics = {
        'inefficiency': {'correct': 0, 'total': 0},
    }

    key = 0
    for eval_item in eval_data:

        gt_item = gt_data[key]
        query = eval_item['user_request']

        s_prompt, u_prompt = create_inefficiency_prompt(query, eval_item['plan'])
        response = gen_answer(s_prompt, u_prompt)
        key = key+1
        try:
            match = re.search(r'\{[\s\S]*?\}', response)
            if not match:
                print(f"Warning (key: {key-1}): {response}")
                continue
            json_string = match.group(0)
            fixed_string = re.sub(r",\s*([}\]])", r"\1", json_string)
            json_string = fixed_string
            predicted_indices = json.loads(json_string)
            pred_inefficiency = set(predicted_indices.get('inefficiency', []))
        except json.JSONDecodeError:
            print(f"Warning (key: {key-1}): {response}")
            continue

        for i, gt_step in enumerate(gt_item['plan']):
        
            if 'is_inefficient' in gt_step:
                gt_label = gt_step['is_inefficient']
                pred_label = 1 if i in pred_inefficiency else 0
                if gt_label == pred_label:
                    metrics['inefficiency']['correct'] += 1
                metrics['inefficiency']['total'] += 1
        print(f"key {key-1}의 inefficiency: {metrics['inefficiency']['correct']}/{metrics['inefficiency']['total']}")

    results = {}
    for metric_name, values in metrics.items():
        if values['total'] > 0:
            accuracy = (values['correct'] / values['total']) * 100
            results[metric_name] = f"{accuracy:.2f}% ({values['correct']} / {values['total']})"
        else:
            results[metric_name] = "N/A"
            
    return results


ground_truth_file = "meta_m&m's.json"
evaluation_file = "meta_m&m's_wo_label.json"

accuracies = trace(ground_truth_file, evaluation_file)

if accuracies:
    print(f"Inefficiency Accuracy : {accuracies['inefficiency']}")

