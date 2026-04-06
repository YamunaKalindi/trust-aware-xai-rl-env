import os
from agent.llm_agent import get_action_from_llm
from env.trust_env import TrustEnv

TASK_NAME = "trust-xai"
BENCHMARK = "custom-env"
MODEL_NAME = os.getenv("MODEL_NAME", "flan-t5")

MAX_STEPS = 5


def log_start():
    print(f"[START] task={TASK_NAME} env={BENCHMARK} model={MODEL_NAME}", flush=True)


def log_step(step, action, reward, done):
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error=null",
        flush=True,
    )


def log_end(success, steps, score, rewards):
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.2f} rewards={rewards_str}",
        flush=True,
    )


def run():
    DATA = [
        {
            "prediction": "Loan Rejected",
            "features": "Low income, poor credit score",
            "user_type": "beginner",
            "correct_action": "simple",
        },
        {
            "prediction": "Loan Approved",
            "features": "High income, good credit score",
            "user_type": "expert",
            "correct_action": "detailed",
        },
    ]

    env = TrustEnv(DATA, task="hard")

    state = env.reset()

    rewards = []
    total_reward = 0

    log_start()

    for step in range(1, MAX_STEPS + 1):
        action = get_action_from_llm(state)

        state, reward, done, _ = env.step(action)

        rewards.append(reward)
        total_reward += reward

        log_step(step, action, reward, done)

        if done:
            break

    # Normalize score to [0,1]
    score = max(0, min(1, (total_reward + 5) / 10))
    success = score > 0.5

    log_end(success, step, score, rewards)


if __name__ == "__main__":
    run()