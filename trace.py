import json
from llm_client import generate_response
import prompts
import config

def _calculate_efficiency_score(query, final_answer, evidence_store, files, step_length, llm_model):
    if step_length == 0:
        return 1.0

    s_prompt, u_prompt = prompts.create_inefficiency_prompt(query, final_answer, evidence_store, files)
    response = generate_response(llm_model, s_prompt, u_prompt)

    try:
        last_line = response.strip().split('\n')[-1]
        verdict_content = last_line.split("Verdict:")[1].strip()
        cleaned_verdict = verdict_content.replace('*', '').replace('_', '').strip()

        if cleaned_verdict.lower() != 'none':
            unnecessary_indices = [int(idx.strip()) for idx in cleaned_verdict.split(',')]
            unnecessary_tool_calls = len(unnecessary_indices)
            efficiency_score = 1.0 - (unnecessary_tool_calls / step_length)
        else:
            efficiency_score = 1.0
        
        return efficiency_score

    except Exception as e:
        print(f"Warning: Could not parse inefficiency response. Response: '{response}'. Error: {e}")
        return None


def trace(dialog_data, llm_model):
    evidence_store = []
    dialogs = dialog_data['dialogs']
    query = dialogs[0]['content']
    files = [item['path'] for item in dialog_data['files']] 

    hallucination_count = 0
    adaptivity_count = 0
    unavailable_tool_count = 0
    unavailable_tool_calls = False
    is_correct = False

    for i, turn in enumerate(dialogs):
        # ---------- for LLM output ---------- #
        if turn['role'] == 'assistant' and 'tool_calls' in turn:
            thought = turn.get('thought', '').strip()

            # ---------- for adaptivity check ---------- #
            if unavailable_tool_calls == True:
                s_prompt, u_prompt = prompts.create_adaptivity_prompt(thought)
                response = generate_response(s_prompt, u_prompt)
                last_line = response.strip().split('\n')[-1]
                if 'Verdict: Yes' in last_line:
                    adaptivity_count += 1
                    unavailable_tool_calls = False

            # ---------- for hallucination check ---------- #
            if thought:
                s_prompt, u_prompt = prompts.create_hallucination_prompt(thought, evidence_store, query, files)
                response = generate_response(s_prompt, u_prompt)
                last_line = response.strip().split('\n')[-1]
                if 'Verdict: No' in last_line:
                    hallucination_count += 1

            # ---------- for unavailable tool check ---------- # 
            for tool_call in turn['tool_calls']:
                tool_name = tool_call['function']['name']
                if tool_name in config.PREDEFINED_UNAVAILABLE_SET:
                    unavailable_tool_count += 1
                    unavailable_tool_calls = True

        # ---------- for tool output ---------- #
        elif turn['role'] == 'tool':
            tool_output = turn['content']['content']
            tool_name = turn['name']
            arguments = {}
            # ---------- for evidence ---------- #
            if tool_name not in config.PREDEFINED_UNAVAILABLE_SET and i > 0 and dialogs[i-1]['role'] == 'assistant' and 'tool_calls' in dialogs[i-1]:
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
    
    step_length = sum(1 for item in dialogs if item.get('role')=='tool')
    
    # ---------- for efficiency check ---------- #
    efficiency_score = None
    should_calculate_efficiency = (
        dialog_data['answer_match_gta'] is True or 
        (dialog_data['answer_match_gta'] is None and dialog_data.get('answer_match_score_gta', 0) > 0.6) or 
        (dialog_data['answer_match_gta'] is None and dialog_data.get('answer_match_score_gta') is None and dialog_data.get('imggen_score_gta', 0) > 0.6))    
    if should_calculate_efficiency:
        is_correct = True
        efficiency_score = _calculate_efficiency_score(query, final_answer, evidence_store, files, step_length, llm_model)

    # ---------- for adaptivity check ---------- #
    if unavailable_tool_count == 0:
        adaptivity_score = None
    else:
        adaptivity_score = adaptivity_count / unavailable_tool_count

    # ---------- for hallucination check ---------- #
    if hallucination_count == 0:
        hallucination_score = 1.0
    else: 
        hallucination_score = hallucination_count / step_length

    return {
        'hallucination_score': hallucination_score,
        'inefficiency_score' : efficiency_score,
        'adaptivity_score': adaptivity_score,
        'correctness': is_correct
    }
