from fastapi import FastAPI, Request
from env.trust_env import TrustEnv

app = FastAPI()

# Global environment instance
env = None

# Sample data
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


# Root endpoint (health check)
@app.get("/")
def root():
    return {"status": "ok"}


# RESET endpoint (VERY IMPORTANT)
@app.post("/reset")
@app.post("/reset/")
async def reset(request: Request):
    global env

    # consume request body (important for OpenEnv)
    try:
        await request.json()
    except:
        pass

    env = TrustEnv(DATA, task="hard")
    state = env.reset()

    return {
        "state": state
    }


# STEP endpoint
@app.post("/step")
@app.post("/step/")
async def step(request: Request):
    global env

    if env is None:
        return {"error": "Environment not initialized. Call /reset first."}

    try:
        data = await request.json()
    except:
        data = {}

    action = data.get("action", "simple")

    state, reward, done, _ = env.step(action)

    return {
        "state": state,
        "reward": float(reward),
        "done": bool(done)
    }