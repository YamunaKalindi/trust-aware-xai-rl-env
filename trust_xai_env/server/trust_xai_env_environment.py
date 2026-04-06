from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..models import TrustXaiAction, TrustXaiObservation
except ImportError:
    from models import TrustXaiAction, TrustXaiObservation


class TrustXaiEnvironment(Environment):
    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self):
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self.trust_score = 0.5

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

    def reset(self) -> TrustXaiObservation:
        import random

        self._state = State(episode_id=str(uuid4()), step_count=0)
        self.trust_score = 0.5
        self.current = random.choice(self.data)

        return TrustXaiObservation(
            observation=f"{self.current['prediction']} | {self.current['features']} | {self.current['user_type']}",
            done=False,
            reward=0.0,
        )

    def step(self, action: TrustXaiAction) -> TrustXaiObservation:
        import random

        self._state.step_count += 1

        user = self.current["user_type"]
        correct_action = self.current["correct_action"]

        chosen = action.action

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

        done = self._state.step_count >= self.max_steps

        # Next state transition
        self.current = random.choice(self.data)

        return TrustXaiObservation(
            observation=f"{self.current['prediction']} | {self.current['features']} | {self.current['user_type']} | trust={self.trust_score:.2f}",
            reward=float(reward),
            done=done,
        )

    @property
    def state(self) -> State:
        return self._state