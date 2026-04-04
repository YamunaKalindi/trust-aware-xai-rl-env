from env.trust_env import TrustEnv
from agent.llm_agent import get_action_from_llm


def evaluate_rule_based(env, episodes=20):
    """
    Baseline agent using simple rules
    """
    episode_rewards = []
    trust_progress = []

    for _ in range(episodes):
        state = env.reset()
        total = 0

        while True:
            user = state["user_type"]

            # Rule-based policy
            if user == "beginner":
                action = "simple"
            else:
                action = "detailed"

            state, reward, done, _ = env.step(action)
            total += reward

            if done:
                episode_rewards.append(total)
                trust_progress.append(env.trust_score)
                break

    avg_reward = sum(episode_rewards) / len(episode_rewards)
    avg_trust = sum(trust_progress) / len(trust_progress)

    return avg_reward, avg_trust, episode_rewards, trust_progress


def evaluate_llm(env, episodes=20):
    """
    Evaluate LLM agent inside environment
    """
    episode_rewards = []
    trust_progress = []

    for _ in range(episodes):
        state = env.reset()
        total = 0

        while True:
            action = get_action_from_llm(state)

            state, reward, done, _ = env.step(action)
            total += reward

            if done:
                episode_rewards.append(total)
                trust_progress.append(env.trust_score)
                break

    avg_reward = sum(episode_rewards) / len(episode_rewards)
    avg_trust = sum(trust_progress) / len(trust_progress)

    return avg_reward, avg_trust, episode_rewards, trust_progress


def debug_run(env):
    """
    Run one episode step-by-step (for demo)
    """
    state = env.reset()
    print("\nInitial State:", state)

    while True:
        action = get_action_from_llm(state)

        print("\nAction:", action)

        state, reward, done, _ = env.step(action)

        print("Reward:", round(reward, 3))
        print("Trust:", round(state["trust_score"], 3))
        print("Reaction:", state["reaction"])

        if done:
            print("\nEpisode finished")
            break


if __name__ == "__main__":
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

    for task in ["easy", "medium", "hard"]:
        print("\n" + "=" * 40)
        print(f"TASK: {task.upper()}")
        print("=" * 40)

        env = TrustEnv(DATA, task=task)

        rb_reward, rb_trust, rb_rewards, rb_trusts = evaluate_rule_based(env)
        llm_reward, llm_trust, llm_rewards, llm_trusts = evaluate_llm(env)

        print("\nRule-Based Agent:")
        print("Avg Reward:", round(rb_reward, 3))
        print("Avg Trust :", round(rb_trust, 3))
        print("Sample Rewards:", [round(r, 2) for r in rb_rewards[:5]])

        print("\nLLM Agent:")
        print("Avg Reward:", round(llm_reward, 3))
        print("Avg Trust :", round(llm_trust, 3))
        print("Sample Rewards:", [round(r, 2) for r in llm_rewards[:5]])

    # Debug demo
    print("\n" + "=" * 40)
    print("DEBUG RUN (HARD TASK)")
    print("=" * 40)

    env = TrustEnv(DATA, task="hard")
    debug_run(env)