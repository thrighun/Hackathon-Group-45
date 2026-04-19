# pyright: reportMissingImports=false
import torch
import json
import re
import traceback
from datetime import datetime

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM
from pymongo import MongoClient
from json_repair import repair_json   # ⭐ important

app = FastAPI()

# =========================
# 🔥 MONGODB CONFIG
# =========================
MONGO_URI = "mongodb+srv://shubhammeenastudent_db_user:Shubham1234@cluster0.2gea5pf.mongodb.net/?appName=Cluster0"
client = MongoClient(MONGO_URI)

db = client["job_parser_db"]
collection = db["job_history"]

print("✅ MongoDB Connected")

# =========================
# DEVICE FIX
# =========================
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# =========================
# MODEL CONFIG
# =========================
MODEL_PATH = "/DATA/teaching/Hackathon/models/gemma-2b-it-fine-tuned"
BASE_MODEL = "/DATA/teaching/Hackathon/models/gemma-2b-it"

SYSTEM_INSTRUCTION = "Return ONLY valid JSON. No explanation."
OUTPUT_SCHEMA = """{
  "job_title": "",
  "location": "",
  "skills": [],
  "experience": "",
  "responsibilities": []
}"""

def prompt_formatter(content):
    return f"""
### SYSTEM:
{SYSTEM_INSTRUCTION}

### USER:
Convert this into JSON.

Schema:
{OUTPUT_SCHEMA}

Input:
{content}
"""

# =========================
# JSON FIX
# =========================
def extract_json(text):
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        json_str = match.group(0)
    else:
        json_str = text

    try:
        json_str = repair_json(json_str)
        return json.loads(json_str)
    except:
        return {
            "error": "json parsing failed",
            "raw": text
        }

# =========================
# LOAD MODEL
# =========================
print("Loading Model...")

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

base = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float16
).to(device)

model = PeftModel.from_pretrained(base, MODEL_PATH)
model.to(device)
model.eval()

print("✅ Model Loaded")

# =========================
# API
# =========================
class JobInput(BaseModel):
    description: str

@app.post("/parse")
async def parse_job(data: JobInput):
    try:
        prompt = prompt_formatter(data.description)

        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        input_len = inputs["input_ids"].shape[1]

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=400,
                do_sample=False,
                temperature=0.1,
                repetition_penalty=1.1,
                eos_token_id=tokenizer.eos_token_id
            )

        decoded = tokenizer.decode(
            outputs[0][input_len:],
            skip_special_tokens=True
        )

        parsed = extract_json(decoded)

        # save to DB
        try:
            collection.insert_one({
                "input_text": data.description,
                "output_json": parsed,
                "timestamp": datetime.utcnow()
            })
        except Exception as e:
            print("MongoDB error:", e)

        return {"json_data": parsed}

    except Exception as e:
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))