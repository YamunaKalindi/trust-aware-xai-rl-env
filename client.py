from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from models import TrustXaiAction, TrustXaiObservation


class TrustXaiEnv(
    EnvClient[TrustXaiAction, TrustXaiObservation, State]
):

    def _step_payload(self, action: TrustXaiAction) -> Dict:
        # ✅ FIX: correct field
        return {
            "action": action.action,
        }

    def _parse_result(self, payload: Dict) -> StepResult[TrustXaiObservation]:
        # ✅ FIX: correct parsing
        observation = TrustXaiObservation(
            observation=payload.get("observation", ""),
            reward=payload.get("reward", 0.0),
            done=payload.get("done", False),
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward", 0.0),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> State:
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )