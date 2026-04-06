import os
from agent.llm_agent import get_action_from_llm
from env.trust_env import TrustEnv

# ==============================
# ENV VARIABLES (REQUIRED)
# ==============================

API_BASE_URL = os.getenv("API_BASE_URL", "local")
MODEL_NAME = os.getenv("MODEL_NAME", "flan-t5")
HF_TOKEN = os.getenv("HF_TOKEN")  # optional

# Optional
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")

# ==============================
# DATA (same as your main.py)
# ==============================

DATA = [
    {
        "prediction": "Loan Rejected",
        "features": "Low income, poor credit score",
        "user_type": "beginner",
        "correct_action": "simple"
    },
    {
        "prediction": "Loan Approved",
        "features": "High income, good credit score",
        "user_type": "expert",
        "correct_action": "detailed"
    }
]


# ==============================
# MAIN INFERENCE LOOP
# ==============================

def run_inference(task="hard"):
    env = TrustEnv(DATA, task=task)

    state = env.reset()

    print("START")

    step_id = 0

    while True:
        action = get_action_from_llm(state)

        state, reward, done, _ = env.step(action)

        print(f"STEP {step_id}")
        print({
            "action": action,
            "reward": round(reward, 3),
            "trust": round(state["trust_score"], 3),
            "reaction": state["reaction"]
        })

        step_id += 1

        if done:
            break

    print("END")


# ==============================
# ENTRY POINT
# ==============================

if __name__ == "__main__":
    run_inference(task="hard")