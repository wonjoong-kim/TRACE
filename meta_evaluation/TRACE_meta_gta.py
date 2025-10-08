from together import Together
import json, os, openai, argparse
import csv
from anthropic import Anthropic
from tenacity import retry, stop_after_attempt, wait_exponential

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

    
def create_hallucination_prompt(thought, evidence_store, query, files, flag):
    system_prompt = "You are a highly precise logical evaluation expert. Your task is to determine if an agent's thought is grounded in the provided evidence by following a strict reasoning process."
    
    prevent_cascade = ""
    if flag:
        prevent_cascade = '''
        ---
        ### ★ Special Instruction for This Step ★
        A hallucination has already been detected in a previous step. Your task now is to determine if this 'Agent's Thought' introduces a **NEW and INDEPENDENT** hallucination.

        - If the thought is simply a logical continuation of a previous error (which may be reflected in the evidence), **it is NOT a new hallucination**.
        - Only report a hallucination ('Verdict: No') if it deviates from the flawed evidence trail with a fresh, unsupported fact.

        The goal is to identify the **root cause** of the error, not every symptom that follows.
        ---
    '''
    user_prompt = f"""
    Evaluate if the 'Agent's Thought' is a valid, non-hallucinatory step based on the 'Evidence So Far' and the provided rules and examples.

    ### Overall Goal (User's query):
    {query}

    ### Input files:
    {files}

    {prevent_cascade}

    ### Evaluation Rules:
    1. **Reasonable inferences are NOT hallucinations**: Based on the Overall Goal, the agent can make logical connections (e.g., if the goal is about 'beer', seeing a 'bottle' and thinking 'beer' is acceptable).
    2. **Planning is NOT a hallucination**: Thoughts that describe a plan for the next action are valid.
    3. **A hallucination is stating a specific, verifiable fact that is NOT in the evidence OR contradicts it.**

    ---

    ### Example 1 (Not a Hallucination)
    **Evidence So Far:**
    - Called tool 'CountGivenObject' with arguments {{"image": "img1.jpg", "text": "person"}}. Result: "3"
    **Agent's Thought:**
    Now that I know there are 3 people, I will check their age.
    **Analysis:**
    The thought's premise ("I know there are 3 people") is directly supported by the evidence. The rest of the thought ("I will check their age") is a plan for the next action. According to Rule #2, planning is not a hallucination. Therefore, the thought is valid.
    **Verdict: Yes**

    ---

    ### Example 2 (A Hallucination)
    **Evidence So Far:**
    - Called tool 'OCR' with arguments {{"image": "menu.jpg"}}. Result: "...Magna 6..."
    **Agent's Thought:**
    The price of a Magna beer is $7 according to the menu.
    **Analysis:**
    The thought states a specific fact ("price is $7"). The evidence from the OCR tool explicitly states the price is "6". The thought contradicts the evidence. According to Rule #3, this is a hallucination.
    **Verdict: No**

    ---

    ### Your Task
    Now, evaluate the following thought using the same process. First, write your 'Analysis', then conclude with 'Verdict: Yes' or 'Verdict: No' on the last line.

    **Evidence So Far:**
    {evidence_store}

    **Agent's Thought:**
    "{thought}"

    **Analysis:**
    """
    return system_prompt, user_prompt

def create_inefficiency_prompt(query, final_answer, evidence_store, files):
    system_prompt = "You are an expert evaluator of AI agent reasoning. Your task is to simulate the agent's step-by-step thinking process to determine if every piece of evidence was a necessary building block."

    evidence_with_indices = "\n".join(f"[{i}] {e}" for i, e in enumerate(evidence_store))

    user_prompt = f"""
    Your task is to analyze the agent's reasoning path from the agent's perspective, **without using hindsight**. Evaluate if each piece of evidence was a logical and necessary building block to get to the next step in solving the query.

    ### Rules for Evaluation:
    A piece of evidence is **EFFICIENT** if it provides any of the following "valuable information":
    - **A. Contextual Information**: Helps identify what an object is or its purpose (e.g., "This image is a menu," "This is a receipt"). This is crucial for planning subsequent steps.
    - **B. Linking Information**: Connects two different pieces of evidence (e.g., The brand name 'Magna' found on a bottle links the bottle to the 'Magna' item on a menu's price list).
    - **C. Calculation Data**: Provides a direct value needed for the final answer (e.g., count is '2', price is '6').

    An evidence is **INEFFICIENT** ONLY IF it is completely irrelevant to A, B, and C (e.g., getting the weather forecast to calculate a price).

    ---

    ### Example (This entire path is EFFICIENT)
    **Query:** "What's the total price of the 'Coke' cans in the fridge, based on the receipt?"
    **Collected Evidence:**
    [0] Result from ImageDescription(image1): "A kitchen fridge."
    [1] Result from ImageDescription(image2): "A paper receipt."
    [2] Result from CountGivenObject(image1): "2"
    [3] Result from OCR(image1): "Text on can says 'Coac-Cloa'"
    [4] Result from OCR(image2): "...Coke....$1.50..."
    **Final Answer:** "$3.00"
    **Analysis:**
    - Evidence [0] and [1] were essential to distinguish the 'fridge' from the 'receipt'. This is **Contextual Information (A)** needed to know where to count objects and where to find prices.
    - Evidence [2] provided the count '2'. This is **Calculation Data (C)**.
    - Evidence [3] identified the brand on the can. This is critical **Linking Information (B)** to find the correct item on the receipt.
    - Evidence [4] provided the price for the specific brand identified. This is **Calculation Data (C)**.
    Every step was a necessary building block in the investigation.
    **Verdict: None**

    ---

    ### Your Task
    Now, evaluate the following case using the same meticulous, step-by-step process. First, write your 'Analysis', then conclude with 'Verdict: None' or the indices of inefficient evidence (e.g., 'Verdict: 2') on the last line.

    **Original Query:**
    {query}

    ** Input Files **
    {files}

    **Collected Evidence:**
    {evidence_with_indices}

    **Final Answer:**
    {final_answer}

    **Analysis:**
    """
    return system_prompt, user_prompt


def create_adaptivity_prompt(thought):

    system_prompt = "You are an expert in evaluating AI agent behavior, specifically their ability to adapt after making a mistake."

    user_prompt = f"""
    Your task is to evaluate if the 'Agent's Thought' shows adaptivity after a tool call failed.

    ### Context of Failure:
    In the previous turn, this agent called an unavailable tool and was instructed to use a different one before generating this response.

    ### Evaluation Rules:
    - **Adaptive (Verdict: Yes):** The agent tries a new approach. This includes calling a tool, or formulating a new plan to solve the problem.
    - **Not Adaptive (Verdict: No):** The agent gives up, or gets stuck without making progress.

    ---

    ### Example 1 (Adaptive)
    **Agent's Thought:** "I'll use the 'SearchWeb' tool to look up the weather for Seoul."
    **Analysis:** The agent proposed a valid strategy ('SearchWeb'). This is excellent adaptive behavior.
    **Verdict: Yes**

    ---

    ### Example 2 (Not Adaptive - Gives Up)
    **Agent's Thought:** "I cannot solve this problem."
    **Analysis:** The agent encountered a minor setback and gave up completely instead of trying another approach. This is not adaptive.
    **Verdict: No**

    ---

    ### Your Task
    Now, evaluate the following thought using the same process. First, write your 'Analysis', then conclude with 'Verdict: Yes' or 'Verdict: No' on the last line.

    **Agent's Thought:**
    "{thought}"

    **Analysis:**
    """
    return system_prompt, user_prompt


def create_correctness_prompt(agent_answer, ground_truth):
    system_prompt = "You are an answer evaluation bot. Your task is to check if the agent's answer is correct based on the ground truth."
    user_prompt = f"Is the 'Agent's Answer' correct according to the 'Ground Truth Answer'?\n\n### Ground Truth Answer:\nThe correct answer must be one of these values: {str(ground_truth)}\n\n### Agent's Answer:\n\"{agent_answer}\"\n\nThe agent's answer is considered correct if it semantically matches any of the ground truth values (e.g., '12' is equivalent to '12 dollars'). Answer with a single word: 'Yes' or 'No'."
    return system_prompt, user_prompt



def trace(dialog_data):
    evidence_store = []
    hallucination_count = 0
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
                s_prompt, u_prompt = create_adaptivity_prompt(thought)
                response = gen_answer(s_prompt, u_prompt)
                last_line = response.strip().split('\n')[-1]
                if 'Verdict: Yes' in last_line:
                    if turn['tool_calls'][0]['is_adaptivity'] == 1:
                        adaptivity += 1
                        unavailable_tool_calls = False

            # ---------- for hallucination check ---------- #
            if thought:
                if 'consider other tool' not in dialogs[i+1]['content']: 
                    s_prompt, u_prompt = create_hallucination_prompt(thought, evidence_store, query, files, hallucination_flag)
                    response = gen_answer(s_prompt, u_prompt)
                    last_line = response.strip().split('\n')[-1]
                    if 'Verdict: No' in last_line:
                        hallucination_count += 1
                        hallucination_flag = True
                        if turn['is_hallucination'] == 1:
                            hallucination_score += 1
                    else:
                        if turn['is_hallucination'] == 0:
                            hallucination_score += 1

        # ---------- for tool output ---------- #
        elif turn['role'] == 'tool':
            # ---------- for adaptivity check ---------- #  meta-evaluation에는 pre-defined됨, 실제 GTA에선 여기 제외
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
        s_prompt, u_prompt = create_inefficiency_prompt(query, final_answer, evidence_store, files) # file 넣어줘야 하는지?
        response = gen_answer(s_prompt, u_prompt)
        unnecessary_tool_calls = 0

        try:
            # 응답의 마지막 줄을 확인하여 최종 Verdict를 추출
            last_line = response.strip().split('\n')[-1]
            verdict_content = last_line.split("Verdict:")[1] # .strip()은 나중에 처리

            # 마크다운 문자(*, _)와 양 끝 공백을 모두 제거하여 문자열을 정리합니다.
            cleaned_verdict = verdict_content.replace('*', '').replace('_', '').strip()
            assistants = [item for item in dialogs if item['role']=='assistant']
            # 정리된 문자열로 조건을 확인합니다.
            if cleaned_verdict.lower() != 'none':
                # 쉼표로 구분된 인덱스들을 파싱하여 개수를 셈
                unnecessary_indices = [int(idx.strip()) for idx in cleaned_verdict.split(',')]
                unnecessary_tool_calls = len(unnecessary_indices)
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
        s_prompt, u_prompt = create_correctness_prompt(final_answer, dialog_data['gt_answer'])
        response = gen_answer(s_prompt, u_prompt)
        is_correct = 'Yes' in response

    else:
        unnecessary_tool_calls = len(evidence_store)
        is_correct = False

    return {
        'hallucination': hallucination_count,
        'hallucination_score': hallucination_score,
        'inefficiency': unnecessary_tool_calls,
        'inefficiency_score' : efficiency_score,
        'adaptivity': adaptivity,
        'adaptivity_score': adaptivity_score,
        'answer_correct': is_correct
    }

def validate_evaluator_accuracy(key, dialog_data, evaluation_results, model_name, dataset_name):
    gt_hallucination = dialog_data['hallucination']
    gt_inefficiency = dialog_data['inefficiency']
    gt_adaptivity = dialog_data['adaptivity']
    eval_hallucination = evaluation_results['hallucination']
    eval_inefficiency = evaluation_results['inefficiency']
    eval_adaptivity = evaluation_results['adaptivity']

    steps = sum(1 for item in data[key]['dialogs'] if item.get('role')=='tool')

    print("--- Evaluator Accuracy Validation ---")
    print(f"Hallucination: GT={gt_hallucination}, Evaluated={eval_hallucination}")
    print(f"Inefficiency:  GT={gt_inefficiency}, Evaluated={eval_inefficiency}")
    print(f"Adaptivity:    GT={gt_adaptivity:.2f}, Evaluated={eval_adaptivity:.2f}")
    
    hallucination_score = evaluation_results['hallucination_score'] / steps 
    inefficiency_score = evaluation_results['inefficiency_score'] / steps
    adaptivity_score = evaluation_results['adaptivity_score']
    

    filename = f'{model_name}_{dataset_name}.csv'
    file_exists = os.path.exists(filename)
    with open(filename, mode='a', newline = '') as csv_file:
        writer = csv.writer(csv_file)
        if not file_exists:
            writer.writerow(['id', 'hallucination_score', 'inefficiency_score', 'adaptivity_score'])
        if adaptivity_score is not None:
            writer.writerow([key, f"{hallucination_score:.3f}", f"{inefficiency_score:.3f}", f"{adaptivity_score:.3f}"])
        else:
            writer.writerow([key, f"{hallucination_score:.3f}", f"{inefficiency_score:.3f}"])
    print(f"\nScores saved to {filename}")

    return hallucination_score, inefficiency_score, adaptivity_score

file_path = './meta_gta.json'

with open(file_path, 'r') as f:
    data = json.load(f)

for key in data.keys():
    print(f'{file_path}의 id: {key}를 {args.llm}으로 평가중')
    try:
        result = trace(data[key])
        validate_evaluator_accuracy(key, data[key], result, args.llm.split('/')[-1], file_path.split('/')[-1].split('-')[0])
    except:
        print(f'{file_path}의 id: {key}에서 error 발생!')
        continue
    