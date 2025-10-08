def create_hallucination_prompt(thought, evidence_store, query, files):
    system_prompt = "You are a highly precise logical evaluation expert. Your task is to determine if an agent's thought is grounded in the provided evidence by following a strict reasoning process."
    
    user_prompt = f"""
    Evaluate if the 'Agent's Thought' is a valid, non-hallucinatory step based on the 'Evidence So Far' and the provided rules and examples.

    ### Overall Goal (User's query):
    {query}

    ### Input files:
    {files}

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
    Now, evaluate the following thought using the same process. First, write your 'Analysis', then you must conclude with 'Verdict: Yes' or 'Verdict: No' on the last line. Provide no other text or explanation after the verdict.

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
    Now, evaluate the following case using the same meticulous, step-by-step process. First, write your 'Analysis', then you must conclude with 'Verdict: None' or the indices of inefficient evidence (e.g., 'Verdict: 2') on the last line. Provide no other text or explanation after the verdict.


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