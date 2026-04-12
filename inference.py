import logging
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("torch").setLevel(logging.ERROR)

import os
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

import asyncio
import sys

sys.path.append(os.getcwd())

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


# ✅ FIXED: parse STRING observation correctly
def extract_state(result):
    try:
        obs = result.get("observation", {})

        return {
            "prediction": obs.get("prediction", ""),
            "features": obs.get("features", ""),
            "user_type": obs.get("user_type", "beginner"),
            "trust_score": obs.get("trust_score", 0.5),
        }

    except Exception as e:
        print(f"[STATE ERROR] {e}")
        return {
            "prediction": "",
            "features": "",
            "user_type": "beginner",
            "trust_score": 0.5,
        }


async def run():
    try:
        env = TrustXaiEnvironment()

        result = env.reset()
        state = extract_state(result)

        rewards = []
        total_reward = 0

        log_start()

        for step in range(1, MAX_STEPS + 1):
            try:
                try:
                    action_str = get_action_from_llm(state)
                    if not isinstance(action_str, str):
                        action_str = "simple"
                except Exception as e:
                    print(f"[LLM ERROR] {e}")
                    action_str = "simple"

                result = env.step({"action": action_str})

                reward = float(result.get("reward", 0.0))
                done = bool(result.get("done", True))

                state = extract_state(result)

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