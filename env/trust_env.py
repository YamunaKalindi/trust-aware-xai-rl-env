import random

ACTIONS = ["simple", "detailed", "counterfactual", "none"]


class TrustEnv:
    def __init__(self, data, task="easy"):
        self.data = data
        self.task = task
        self.current = None
        self.trust_score = 0.5
        self.step_count = 0

        if task == "easy":
            self.max_steps = 1
        elif task == "medium":
            self.max_steps = 2   # sequential now
        else:
            self.max_steps = 3   # multi-step RL

    def reset(self):
        self.current = random.choice(self.data)
        self.trust_score = 0.5
        self.step_count = 0

        # reset reaction memory
        self.current["reaction"] = "neutral"

        return self._get_state()

    def _get_state(self):
        return {
            "prediction": self.current["prediction"],
            "features": self.current["features"],
            "user_type": self.current["user_type"],
            "trust_score": self.trust_score,
            "task": self.task,
            "reaction": self.current.get("reaction", "neutral"),
            "step_count": self.step_count
        }

    def step(self, action):
        correct_action = self.current["correct_action"]
        user = self.current["user_type"]

        # --- Correctness ---
        correctness_score = 1 if action == correct_action else -1

        # --- Personalization ---
        if user == "beginner" and action == "simple":
            personalization_score = 1
        elif user == "expert" and action == "detailed":
            personalization_score = 1
        else:
            personalization_score = 0

        # --- Trust Update ---
        trust_change = 0.1 if correctness_score > 0 else -0.1
        self.trust_score += trust_change
        self.trust_score = max(0, min(1, self.trust_score))

        # --- User Reaction (state evolves) ---
        if correctness_score > 0:
            reaction = "satisfied"
        else:
            reaction = "confused"

        self.current["reaction"] = reaction

        # --- Reward Calculation ---
        if self.task == "easy":
            reward = correctness_score

        elif self.task == "medium":
            reward = correctness_score + personalization_score

        else:  # HARD → true RL
            reward = (
                0.7 * correctness_score +
                0.3 * personalization_score +
                1.0 * trust_change
            )

        # --- Optional penalty for ignoring ---
        if action == "none" and correctness_score < 0:
            reward -= 0.5

        # --- Transition (VERY IMPORTANT for RL) ---
        if self.task != "easy":
            if self.trust_score < 0.4:
                # low trust → harder cases (expert users)
                hard_cases = [d for d in self.data if d["user_type"] == "expert"]
                if hard_cases:
                    self.current = random.choice(hard_cases)
                else:
                    self.current = random.choice(self.data)
            else:
                # normal transition
                self.current = random.choice(self.data)

            # reset reaction for new state
            self.current["reaction"] = "neutral"

        # --- Step tracking ---
        self.step_count += 1
        done = self.step_count >= self.max_steps

        return self._get_state(), reward, done, {}