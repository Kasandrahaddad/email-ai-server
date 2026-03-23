from fastapi import FastAPI, Request
import os
import json
from google import genai

app = FastAPI()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

SYSTEM_PROMPT = """
You are an AI email agent.

Decide the best action for the email.

Available actions:
- send_reply
- mark_important
- schedule_meeting
- extract_task
- ignore

Return ONLY JSON:

{
  "action": "...",
  "response": "...",
  "reason": "..."
}
"""

def parse_json(text):
    text = text.strip()

    if text.startswith("```"):
        text = text.split("```")[1]

    try:
        return json.loads(text)
    except:
        return {
            "action": "send_reply",
            "response": text,
            "reason": "fallback"
        }

@app.get("/")
def home():
    return {"status": "ok"}

@app.post("/analyze")
async def analyze(request: Request):
    try:
        data = await request.json()
        email = data.get("email") or ""

        if not email.strip():
            return {
                "action": "ignore",
                "response": "",
                "reason": "empty"
            }

        prompt = f"{SYSTEM_PROMPT}\n\nEmail:\n{email}"

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )

        return parse_json(response.text)

    except Exception as e:
        return {
            "action": "error",
            "response": "",
            "reason": str(e)
        }

@app.post("/reply")
async def reply(request: Request):
    data = await request.json()
    email = data.get("email") or ""

    prompt = f"Write a professional reply:\n\n{email}"

    res = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )

    return {"reply": res.text}

@app.post("/extract-task")
async def extract_task(request: Request):
    data = await request.json()
    email = data.get("email") or ""

    prompt = f"Extract tasks from this email:\n\n{email}"

    res = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )

    return {"tasks": res.text}

@app.post("/schedule")
async def schedule(request: Request):
    data = await request.json()
    email = data.get("email") or ""

    prompt = f"Does this email request a meeting?\n\n{email}"

    res = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )

    return {"schedule": res.text}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ["PORT"])
    uvicorn.run(app, host="0.0.0.0", port=port)