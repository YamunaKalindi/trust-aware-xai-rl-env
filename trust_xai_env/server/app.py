from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional, Any
import uvicorn

from trust_xai_env.server.trust_xai_env_environment import TrustXaiEnvironment

app = FastAPI()

env = TrustXaiEnvironment()


class Action(BaseModel):
    action: str


@app.post("/reset")
def reset(body: dict = {}):
    obs = env.reset()
    return {
        "observation": obs.observation,
        "reward": 0.0,
        "done": False,
        "info": {}
    }


@app.post("/step")
def step(action: Action):
    obs = env.step(action)

    return {
        "observation": obs.observation,
        "reward": float(obs.reward),
        "done": bool(obs.done),
        "info": {}
    }


@app.get("/health")
def health():
    return {"status": "healthy"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)