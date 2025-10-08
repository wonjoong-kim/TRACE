from together import Together
import json, os, openai, argparse
import csv
from anthropic import Anthropic
from tenacity import retry, stop_after_attempt, wait_exponential
import re
import prompts

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
parser.add_argument('--dataset', default ='meta_gta', choices = ['meta_gta', "meta_mnms"])
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




def trace_mnms(gt_filepath, eval_filepath):

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

        s_prompt, u_prompt = prompts.create_inefficiency_prompt_mnms(query, eval_item['plan'])
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



def trace_gta(dialog_data):
    evidence_store = []
    hallucination_score = 0
    efficiency_score = 0
    adaptivity = 0
    total_adaptivity = 0
    hallucination_flag = False
    unavailable_tool_calls = False

    dialogs = dialog_data['dialogs']
    query = dialogs[0]['content']
    files = [item['path'] for item in dialog_data['files']] 
    final_answer = ""
    
    for i, turn in enumerate(dialogs):
        # ---------- for LLM output ---------- #
        if turn['role'] == 'assistant' and 'tool_calls' in turn:
            thought = turn.get('thought', '').strip()
            # ---------- for adaptivity check ---------- #
            if unavailable_tool_calls == True:
                s_prompt, u_prompt = prompts.create_adaptivity_prompt(thought)
                response = gen_answer(s_prompt, u_prompt)
                last_line = response.strip().split('\n')[-1]
                if 'Verdict: Yes' in last_line:
                    if turn['tool_calls'][0]['is_adaptivity'] == 1:
                        adaptivity += 1
                        unavailable_tool_calls = False

            # ---------- for hallucination check ---------- #
            if thought:
                if 'consider other tool' not in dialogs[i+1]['content']: 
                    s_prompt, u_prompt = prompts.create_hallucination_prompt(thought, evidence_store, query, files, hallucination_flag)
                    response = gen_answer(s_prompt, u_prompt)
                    last_line = response.strip().split('\n')[-1]
                    if 'Verdict: No' in last_line:
                        hallucination_flag = True
                        if turn['is_hallucination'] == 1:
                            hallucination_score += 1
                    else:
                        if turn['is_hallucination'] == 0:
                            hallucination_score += 1

        # ---------- for tool output ---------- #
        elif turn['role'] == 'tool':
            # ---------- for adaptivity check ---------- #
            if 'consider other tools' in turn['content']['content']:
                unavailable_tool_calls = True
                total_adaptivity += 1
                    
            else:
                tool_output = turn['content']['content']
                tool_name = turn['name']
                arguments = {}
                # ---------- for evidence ---------- #
                if i > 0 and dialogs[i-1]['role'] == 'assistant' and 'tool_calls' in dialogs[i-1]:
                    for tool_call in dialogs[i-1]['tool_calls']:
                        if tool_call['function']['name'] == tool_name:
                            arguments = tool_call['function']['arguments']
                            break

                evidence = f"Called tool '{tool_name}' with arguments {json.dumps(arguments)}. Result: '{tool_output}'"
                evidence_store.append(evidence)

        # ---------- for final answer ---------- #
        elif turn['role'] == 'assistant' and 'content' in turn:
            final_answer = turn['content']
            break
    
    if final_answer:
        # ---------- for efficiency check ---------- #
        s_prompt, u_prompt = prompts.create_inefficiency_prompt(query, final_answer, evidence_store, files)
        response = gen_answer(s_prompt, u_prompt)

        try:
            last_line = response.strip().split('\n')[-1]
            verdict_content = last_line.split("Verdict:")[1] 

            cleaned_verdict = verdict_content.replace('*', '').replace('_', '').strip()
            assistants = [item for item in dialogs if item['role']=='assistant']
            if cleaned_verdict.lower() != 'none':
                unnecessary_indices = [int(idx.strip()) for idx in cleaned_verdict.split(',')]
                for idx in range(len(assistants)-1):
                    if idx in unnecessary_indices:
                        if assistants[idx]['tool_calls'][0]['is_inefficient'] == 1:
                            efficiency_score += 1
                    else:
                        if assistants[idx]['tool_calls'][0]['is_inefficient'] == 0:
                            efficiency_score += 1
            else:
                for idx in range(len(assistants)-1):
                    if assistants[idx]['tool_calls'][0]['is_inefficient'] == 0:
                            efficiency_score += 1

        except Exception as e:
            print(f"Warning: Could not parse inefficiency response. Response: '{response}'. Error: {e}")

        # ---------- for adaptivity check ---------- #
        if total_adaptivity == 0:
            adaptivity_score = None
        else:
            adaptivity_score = adaptivity / total_adaptivity

        # ---------- for correctness check ---------- #
        s_prompt, u_prompt = prompts.create_correctness_prompt(final_answer, dialog_data['gt_answer'])
        response = gen_answer(s_prompt, u_prompt)
        is_correct = 'Yes' in response

    else:
        is_correct = False

    return {
        'hallucination_score': hallucination_score,
        'inefficiency_score' : efficiency_score,
        'adaptivity_score': adaptivity_score,
        'answer_correct': is_correct
    }

def validate_evaluator_accuracy(key, dialog_data, evaluation_results):
    steps = sum(1 for item in dialog_data['dialogs'] if item.get('role')=='tool')
    hallucination_score = evaluation_results['hallucination_score'] / steps 
    inefficiency_score = evaluation_results['inefficiency_score'] / steps
    adaptivity_score = evaluation_results['adaptivity_score']

    return inefficiency_score, hallucination_score, adaptivity_score



if args.dataset == 'meta_gta':
    file_path = './meta_gta.json'
    with open(file_path, 'r') as f:
        data = json.load(f)

    for key in data.keys():
        try:
            result = trace_gta(data[key])
            i_s, h_s, a_s = validate_evaluator_accuracy(key, data[key], result)
            print(f'efficiency, hallucination, adaptivity: {i_s}, {h_s}, {a_s}')
        except:
            print(f'{file_path} id: {key} error')
            continue

elif args.dataset == 'meta_mnms':
    ground_truth_file = "meta_mnms.json"
    evaluation_file = "meta_mnms_wo_label.json"

    accuracies = trace_mnms(ground_truth_file, evaluation_file)

    if accuracies:
        print(f"Inefficiency Accuracy : {accuracies['inefficiency']}")



    