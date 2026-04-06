from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

from trust_xai_env.server.trust_xai_env_environment import TrustXaiEnvironment

app = FastAPI()

env = TrustXaiEnvironment()


class Action(BaseModel):
    action: str


@app.post("/reset")
def reset(body: dict = {}):
    obs = env.reset()
    return obs   # ✅ directly return dict


@app.post("/step")
def step(action: Action):
    obs = env.step(action.dict())   # ✅ pass dict
    return obs   # ✅ directly return dict


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.get("/")
def root():
    return {"message": "API is running"}  # optional but helpful


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)