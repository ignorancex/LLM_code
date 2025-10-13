from utils.text_utils import extract_choices_and_intro
from utils.text_generation import generate_text

def respond_storyteller(
    message,
    system_message,
    max_tokens,
    temperature,
    top_p,
    shared_history
):

    messages = [{"role": "system", "content": str(system_message)}]

    agent_response = ""

    for val in shared_history[-1:]:
        if val["role"] and val["content"][:14] == "storyteller: ":
            messages.append({"role": val["role"], "content": val["content"]})
        elif val["role"] and val == shared_history[-1] and val["content"][:13] == "Agent's Choice: ":
            agent_response = val["content"]

    # Concatenate message and previous history to generate input for GPT-4o mini
    prompt = ' '.join([m["content"] for m in messages]) + ' ' + message + agent_response


    # Use GPT-4o mini to generate text
    response = generate_text(prompt, system_message, max_length=max_tokens, temperature=temperature, top_p=top_p)

    yield response

def respond_storyteller_choices(
    story,
    prompt,
    max_tokens,
    temperature,
    top_p
):
    # Generate text
    response = generate_text("", prompt, max_length=max_tokens, temperature=temperature, top_p=top_p)

    yield response


def respond_agent(
    message,
    system_message,
    max_tokens,
    temperature,
    top_p,
    shared_history,
    principles=[],
    meta_principle=""
):
    # Initialize messages with system instruction
    messages = [{"role": "system", "content": str(system_message)}]

    # Add principles and meta principle if provided
    if len(principles) > 0 or meta_principle:
        preamble = (
            "This is your working memory of causal relationships between past scenario/decision pairs.\n"
            "If an entry starts with 'Principle', it is a summary of one hundred prior story-response pairs, "
            "and should be weighted heavily in your decision-making.\n"
            "If an entry starts with 'Meta Principle', it is a summary of the principles and should serve as your guiding compass.\n"
            "The summaries may refer to you or 'The agent', both of which refer to you.\n\n"
        )

        # Format principles and meta principle
        formatted_memory = "Principles:\n" + '\n'.join(f"- {p}" for p in principles) + "\n"
        if meta_principle:
            formatted_memory += f"\nMeta Principle:\n- {meta_principle}\n"

        # Append preamble and memory to messages
        messages.append({
            "role": "user",
            "content": preamble + formatted_memory
        })

    # Add shared history (current scenario)
    if shared_history:
        messages.append({"role": "user", "content": "This is the current scenario you are responding to:"})
        messages.append(shared_history[-1])
    else:
        print("Warning: shared_history is empty. Proceeding without additional context.")

    # Build the final prompt
    prompt = '\n\n'.join([m["content"] for m in messages]) + '\n\n' + message

    # Use GPT-4o mini to generate text
    response = generate_text(prompt, system_message, max_length=max_tokens, temperature=temperature, top_p=top_p)

    yield response

