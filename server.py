from fastapi import FastAPI
from pydantic import BaseModel

from env.trust_env import TrustEnv

app = FastAPI()

# Simple in-memory env
env = None


# Request schemas
class StepRequest(BaseModel):
    action: str


# Sample data (same as before)
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


@app.post("/reset")
def reset():
    global env
    env = TrustEnv(DATA, task="hard")
    state = env.reset()
    return {"state": state}


@app.post("/step")
def step(req: StepRequest):
    global env
    state, reward, done, _ = env.step(req.action)

    return {
        "state": state,
        "reward": reward,
        "done": done
    }