from fastapi import FastAPI, Request
import os
import json
from google import genai

app = FastAPI()

# Gemini Setup
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# SYSTEM PROMPT (Open Agent)
SYSTEM_PROMPT = """
You are an advanced AI email agent.

Understand ANY email deeply.

IMPORTANT RULES:
- Return ONLY raw JSON
- Do NOT wrap in text
- Do NOT use `json
- Do NOT explain anything
- Output must start with {

FORMAT:

{
  "intent": "",
  "priority": "low | medium | high",
  "should_reply": true,
  "suggested_action": "",
  "response": "",
  "summary": ""
}
"""
# JSON PARSER
def parse_json(text):
    text = text.strip()

    if "```" in text:
        text = text.replace("```json", "").replace("```", "").strip()

    if "{" in text:
        text = text[text.find("{"):]

    try:
        return json.loads(text)
    except Exception as e:
        return {
            "intent": "unknown",
            "priority": "low",
            "should_reply": False,
            "suggested_action": "manual review",
            "response": text,
            "summary": f"fallback parsing: {str(e)}"
        }

# ROUTES=
@app.get("/")
def home():
    return {"message": "AI Email Agent Running"}

@app.post("/analyze")
async def analyze(request: Request):
    try:
        data = await request.json()
        email = data.get("email") or ""

        if not email.strip():
            return {
                "intent": "empty",
                "priority": "low",
                "should_reply": False,
                "suggested_action": "ignore",
                "response": "",
                "summary": "empty email"
            }

        prompt = f"{SYSTEM_PROMPT}\n\nEMAIL:\n{email}"

        res = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )

        clean = res.text.strip()
        return parse_json(clean)

    except Exception as e:
        return {
            "intent": "error",
            "priority": "low",
            "should_reply": False,
            "suggested_action": "check system",
            "response": "",
            "summary": str(e)
        }

# OPTIONAL: Reply endpoint
@app.post("/reply")
async def reply(request: Request):
    data = await request.json()
    email = data.get("email") or ""

    prompt = f"""
Write a professional and helpful email reply.

EMAIL:
{email}
"""

    res = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )

    return {"reply": res.text}

# RUN SERVER
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ["PORT"])
    uvicorn.run(app, host="0.0.0.0", port=port)