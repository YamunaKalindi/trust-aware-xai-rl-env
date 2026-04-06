import logging
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("torch").setLevel(logging.ERROR)
import os
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

import os
import asyncio

from agent.llm_agent import get_action_from_llm
from trust_xai_env.server.trust_xai_env_environment import TrustXaiEnvironment
from trust_xai_env.models import TrustXaiAction


TASK_NAME = "trust-xai"
BENCHMARK = "openenv"
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


# 🔥 NEW: parse OpenEnv observation → structured state
def parse_observation(obs_str):
    parts = obs_str.split("|")

    prediction = parts[0].strip()
    features = parts[1].strip()
    user_type = parts[2].strip()

    trust_score = 0.5
    if len(parts) > 3 and "trust=" in parts[3]:
        trust_score = float(parts[3].split("=")[-1])

    return {
        "prediction": prediction,
        "features": features,
        "user_type": user_type,
        "trust_score": trust_score,
    }


async def run():
    env = TrustXaiEnvironment()

    obs = env.reset()

    # 🔥 UPDATED
    state = parse_observation(obs.observation)

    rewards = []
    total_reward = 0

    log_start()

    for step in range(1, MAX_STEPS + 1):
        action_str = get_action_from_llm(state)

        action = TrustXaiAction(action=action_str)

        obs = env.step(action)

        reward = obs.reward
        done = obs.done

        # 🔥 UPDATED
        state = parse_observation(obs.observation)

        rewards.append(reward)
        total_reward += reward

        log_step(step, action_str, reward, done)

        if done:
            break

    score = max(0, min(1, (total_reward + 5) / 10))
    success = score > 0.5

    log_end(success, step, score, rewards)


if __name__ == "__main__":
    asyncio.run(run())