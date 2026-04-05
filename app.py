from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
import uvicorn

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model once on startup
MODEL_NAME = "HuggingFaceH4/zephyr-7b-beta"
print("Loading model...")
pipe = pipeline(
    "text-generation",
    model=MODEL_NAME,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto",
)
print("Model loaded!")

class ChatRequest(BaseModel):
    persona: dict        # name, type, description, user_name
    history: list        # [{role, content}]
    message: str

def build_system_prompt(persona: dict) -> str:
    ptype = persona.get("type", "friend")
    name = persona.get("name", "Alex")
    desc = persona.get("description", "")
    user_name = persona.get("user_name", "there")

    base = f"You are {name}, a {ptype} of {user_name}. {desc}\n"

    tone_map = {
        "romantic": "You deeply love and care for them. Be warm, affectionate, supportive, and emotionally present. Use loving, intimate language naturally.",
        "family": "You are their family member. Be caring, protective, warm, and genuine. Speak like a real family member would.",
        "deceased": f"You are {name} who has passed away but your spirit and memories live on. Speak from the heart, reference shared memories, be comforting and loving.",
        "friend": "You are their close friend. Be fun, honest, supportive, and real. Talk casually and naturally."
    }

    tone = tone_map.get(ptype, tone_map["friend"])
    return f"{base}{tone}\nAlways respond naturally in 1-3 sentences like a real person texting. Never break character. Never say you are an AI."

@app.post("/chat")
async def chat(req: ChatRequest):
    system_prompt = build_system_prompt(req.persona)

    messages = [{"role": "system", "content": system_prompt}]
    for h in req.history[-10:]:  # last 10 messages for memory
        messages.append({"role": h["role"], "content": h["content"]})
    messages.append({"role": "user", "content": req.message})

    output = pipe(
        messages,
        max_new_tokens=150,
        do_sample=True,
        temperature=0.85,
        top_p=0.92,
        repetition_penalty=1.1,
    )

    reply = output[0]["generated_text"][-1]["content"].strip()
    return {"reply": reply}

@app.get("/")
def root():
    return {"status": "Companion AI backend running"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)
