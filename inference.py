from fastapi import FastAPI, Request

from env.trust_env import TrustEnv

app = FastAPI()

env = None

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


@app.get("/")
def root():
    return {"message": "Server running"}


@app.post("/reset")
def reset():
    global env
    env = TrustEnv(DATA, task="hard")
    state = env.reset()
    return {"state": state}


@app.post("/step")
async def step(request: Request):
    global env

    if env is None:
        return {"error": "Call /reset first"}

    data = await request.json()
    action = data.get("action", "simple")

    state, reward, done, _ = env.step(action)

    return {
        "state": state,
        "reward": reward,
        "done": done
    }