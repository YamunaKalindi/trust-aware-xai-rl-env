from env.trust_env import TrustEnv
from evaluation.evaluate import evaluate_rule_based, evaluate_llm, debug_run

# Your dataset (keep it here for simplicity)
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
    },
    {
        "prediction": "Loan Rejected",
        "features": "High debt-to-income ratio",
        "user_type": "expert",
        "correct_action": "detailed"
    },
    {
        "prediction": "Loan Approved",
        "features": "Stable job, moderate income",
        "user_type": "beginner",
        "correct_action": "simple"
    }
]


def run_all_tasks():
    print("\n" + "=" * 50)
    print("TRUST-AWARE RL ENVIRONMENT EVALUATION")
    print("=" * 50)

    results = []

    for task in ["easy", "medium", "hard"]:
        print("\n" + "=" * 40)
        print(f"TASK: {task.upper()}")
        print("=" * 40)

        env = TrustEnv(DATA, task=task)

        rb_reward, rb_trust = evaluate_rule_based(env)
        llm_reward, llm_trust = evaluate_llm(env)

        print("\nRule-Based Agent:")
        print(f"  Avg Reward: {round(rb_reward, 3)}")
        print(f"  Avg Trust : {round(rb_trust, 3)}")

        print("\nLLM Agent:")
        print(f"  Avg Reward: {round(llm_reward, 3)}")
        print(f"  Avg Trust : {round(llm_trust, 3)}")

        results.append({
            "task": task,
            "rb_reward": rb_reward,
            "rb_trust": rb_trust,
            "llm_reward": llm_reward,
            "llm_trust": llm_trust
        })

    return results


def run_demo():
    print("\n" + "=" * 50)
    print("DEMO: LLM INTERACTION (HARD TASK)")
    print("=" * 50)

    env = TrustEnv(DATA, task="hard")
    debug_run(env)


if __name__ == "__main__":
    # Run evaluation
    results = run_all_tasks()

    # Run demo interaction
    run_demo()