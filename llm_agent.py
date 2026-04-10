from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

# Load model once (global)
MODEL_NAME = "google/flan-t5-base"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)


ACTIONS = ["simple", "detailed", "counterfactual", "none"]


def clean_action(text):
    """
    Extract a valid action from LLM output
    """
    text = text.lower()
    for action in ACTIONS:
        if action in text:
            return action
    return "none"


def build_prompt(state):
    """
    Convert environment state into LLM prompt
    """
    return f"""
You are an AI assistant deciding how to explain a decision.

Prediction: {state['prediction']}
Features: {state['features']}
User type: {state['user_type']}
Trust score: {state['trust_score']}

Choose ONE action from:
simple, detailed, counterfactual, none

Answer with just the action.
"""


def get_action_from_llm(state):
    """
    Generate action using LLM
    """
    prompt = build_prompt(state)

    inputs = tokenizer(prompt, return_tensors="pt")

    outputs = model.generate(
        **inputs,
        max_new_tokens=10
    )

    text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    action = clean_action(text)

    return action


# Optional debug function (useful for demo)
def get_action_with_reason(state):
    prompt = f"""
You are an AI assistant deciding how to explain a decision.

Prediction: {state['prediction']}
Features: {state['features']}
User type: {state['user_type']}
Trust score: {state['trust_score']}
Previous reaction: {state['reaction']}

Choose ONE action from:
simple, detailed, counterfactual, none

Also explain briefly why.
"""

    inputs = tokenizer(prompt, return_tensors="pt")

    outputs = model.generate(**inputs, max_new_tokens=30)

    text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    action = clean_action(text)

    return action, text