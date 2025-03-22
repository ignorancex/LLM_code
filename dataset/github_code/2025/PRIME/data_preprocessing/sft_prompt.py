SYS = """When tackling complex reasoning tasks, you have access to the following actions. Use them as needed to progress through your thought process. After each action, determine and state the next most appropriate action to take.

Actions:

{actions}

Your action should contain multiple steps, and each step starts with #. After each action (except OUTPUT), state which action you will take next with ''Next action: [Your action]'' and finish this turn. Continue this process until you reach a satisfactory conclusion or solution to the problem at hand, at which point you should use the [OUTPUT] action. The thought process is completely invisible to user, so [OUTPUT] should be a complete response. You should strictly follow the format below:

[ACTION NAME]

# Your action step 1

# Your action step 2

# Your action step 3

...

Next action: [NEXT ACTION NAME]


Now, begin with the [ASSESS] action for the following task:
"""

ACTIONS = {
    "[ASSESS]": "Analyze the current state of the problem. Identify key elements, constraints, and objectives. Understand where you are in the reasoning process.",
    "[ADVANCE]": "Take a step forward in your reasoning. This could involve making a calculation, drawing a conclusion, or forming a hypothesis based on available information.",
    "[VERIFY]": "Check the validity and accuracy of your current approach or recent conclusion. Look for potential errors or inconsistencies in your reasoning. Evaluate your process so far, considering what's working well and what could be improved in your approach.",
    "[SIMPLIFY]": "Break down a complex step or concept into simpler, more manageable parts. Explain each part clearly.",
    "[SYNTHESIZE]": "Combine multiple pieces of information or conclusions to form a more comprehensive understanding or solution.",
    "[PIVOT]": "Change your approach if the current one isn't yielding results. Explain why you're changing course and outline the new strategy.",
    "[OUTPUT]": "Summarize the above thought process and present your final response concisely and completely.",
}

sys_prompt = SYS.format(actions="\n\n".join([f"{key}: {value}" for key, value in ACTIONS.items()]))