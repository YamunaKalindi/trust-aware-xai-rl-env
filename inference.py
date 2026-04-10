import logging
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("torch").setLevel(logging.ERROR)

import os
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

import asyncio

from llm_agent import get_action_from_llm
from server.trust_xai_env_environment import TrustXaiEnvironment


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


async def run():
    try:
        env = TrustXaiEnvironment()

        obs = env.reset()

        if not isinstance(obs, dict) or "observation" not in obs:
            print("[ERROR] Invalid reset response")
            return

        state = obs["observation"]

        rewards = []
        total_reward = 0

        log_start()

        for step in range(1, MAX_STEPS + 1):
            try:
                action_str = get_action_from_llm(state)

                action = {"action": action_str}

                obs = env.step(action)

                if not isinstance(obs, dict):
                    print("[ERROR] Invalid step response")
                    break

                reward = obs.get("reward", 0.0)
                done = obs.get("done", True)
                state = obs.get("observation", {})

                rewards.append(reward)
                total_reward += reward

                log_step(step, action_str, reward, done)

                if done:
                    break

            except Exception as step_error:
                print(f"[STEP ERROR] {step_error}")
                break

        score = max(0, min(1, (total_reward + 5) / 10))
        success = score > 0.5

        log_end(success, step, score, rewards)

    except Exception as e:
        print(f"[FATAL ERROR] {e}")


if __name__ == "__main__":
    asyncio.run(run())