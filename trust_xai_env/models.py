from openenv.core.env_server.types import Action, Observation
from pydantic import Field


class TrustXaiAction(Action):
    """
    Action = explanation strategy chosen by agent
    """

    action: str = Field(
        ...,
        description="Explanation strategy: simple, detailed, counterfactual, none",
    )


class TrustXaiObservation(Observation):
    """
    Observation = current state of environment
    """

    observation: str = Field(
        ...,
        description="Current scenario including prediction, features, user type, trust",
    )

    reward: float = Field(
        default=0.0,
        description="Reward from last action",
    )

    done: bool = Field(
        default=False,
        description="Whether episode is finished",
    )