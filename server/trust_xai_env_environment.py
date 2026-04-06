import random


class TrustXaiEnvironment:
    def __init__(self):
        self.trust_score = 0.5
        self.step_count = 0

        self.data = [
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

        self.current = None
        self.max_steps = 3

    def reset(self):
        self.step_count = 0
        self.trust_score = 0.5
        self.current = random.choice(self.data)

        return {
            "observation": {
                "prediction": self.current["prediction"],
                "features": self.current["features"],
                "user_type": self.current["user_type"],
                "trust_score": self.trust_score,
            },
            "reward": 0.0,
            "done": False,
            "info": {},
        }

    def step(self, action):
        self.step_count += 1

        user = self.current["user_type"]
        correct_action = self.current["correct_action"]

        # handle both dict and object input safely
        if isinstance(action, dict):
            chosen = action.get("action", "simple")
        else:
            chosen = getattr(action, "action", "simple")

        # --- Correctness ---
        correctness = 1 if chosen == correct_action else -1

        # --- Personalization ---
        if user == "beginner" and chosen == "simple":
            personalization = 1
        elif user == "expert" and chosen == "detailed":
            personalization = 1
        else:
            personalization = 0

        # --- Trust ---
        trust_change = 0.1 if correctness > 0 else -0.1
        self.trust_score = max(0, min(1, self.trust_score + trust_change))

        reward = correctness + personalization + (0.5 * trust_change)

        done = self.step_count >= self.max_steps

        # Move to next state
        self.current = random.choice(self.data)

        return {
            "observation": {
                "prediction": self.current["prediction"],
                "features": self.current["features"],
                "user_type": self.current["user_type"],
                "trust_score": self.trust_score,
            },
            "reward": float(reward),
            "done": done,
            "info": {},
        }