import torch  # pyright: ignore[reportMissingImports]
import json
import re
import traceback
from fastapi import FastAPI, HTTPException   # pyright: ignore[reportMissingImports]
from pydantic import BaseModel  # pyright: ignore[reportMissingImports]
from peft import PeftModel  # pyright: ignore[reportMissingImports]
from transformers import AutoTokenizer, AutoModelForCausalLM  # pyright: ignore[reportMissingImports]
from typing import Optional
from json_repair import repair_json  # pyright: ignore[reportMissingImports]

app = FastAPI()

# --- CONFIG ---
BASE_MODEL_PATH = "/DATA/teaching/Hackathon/models/gemma-2b-it"
GEN_ADAPTER_PATH = "/DATA/teaching/Hackathon/models/gemma-2b-it-fine-tuned"
EDIT_ADAPTER_PATH = "/DATA/teaching/Hackathon/models/gemma-2b-it-fine-tuned-edit"

# --- DEVICE FIX ---
device = "cuda" if torch.cuda.is_available() else "cpu"

# --- JSON EXTRACTION + REPAIR ---
def extract_robust_json(output: str):
    try:
        # Extract JSON block
        match = re.search(r"\{.*\}", output, re.DOTALL)

        if match:
            json_str = match.group(0)
        else:
            json_str = output

        # Repair JSON
        json_str = repair_json(json_str)

        return json.loads(json_str)

    except Exception:
        return {
            "error": "JSON parsing failed",
            "raw": output
        }

# --- MODEL LOADING ---
print("Loading Model...")

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH)

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_PATH,
    torch_dtype=torch.float16
).to(device)   # ✅ FORCE SAME DEVICE

model = PeftModel.from_pretrained(base_model, GEN_ADAPTER_PATH, adapter_name="generator")
model.load_adapter(EDIT_ADAPTER_PATH, adapter_name="editor")

model.eval()

# --- REQUEST SCHEMA ---
class JDRequest(BaseModel):
    user_input: str
    current_state: Optional[dict] = None

# --- API ---
@app.post("/process")
async def process_job_data(data: JDRequest):
    try:
        if data.current_state:
            model.set_adapter("editor")

            prompt = f"""
You must return ONLY valid JSON.

Update this JSON based on instruction.

JSON:
{json.dumps(data.current_state)}

Instruction:
{data.user_input}

Return only JSON.
"""

        else:
            model.set_adapter("generator")

            prompt = f"""
Convert the following text into valid JSON.

Rules:
- Only JSON output
- No explanation
- No extra text

Input:
{data.user_input}
"""

        # Tokenize
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=1024,        # ✅ prevent truncation
                do_sample=False,
                temperature=0.1,
                repetition_penalty=1.1,
                eos_token_id=tokenizer.eos_token_id
            )

        decoded = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True
        )

        return extract_robust_json(decoded)

    except Exception as e:
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))