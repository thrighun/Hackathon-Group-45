"""
inference.py
Unified inference for the JD Generator + Editor combined model.
Loads one base model (google/gemma-2b-it) with two LoRA adapters
(thrighun/jd-generator and thrighun/jd-editor) swapped at runtime.

Usage:
    from inference import JDModel
    jd = JDModel()

    # First call → GENERATE from raw text
    result = jd("Python developer, 2 years experience, Django, REST APIs")

    # Subsequent calls → EDIT the existing JD
    result = jd("add Docker to requirements")
    result = jd("change location to New York")

    # Reset to start a new JD
    jd.reset()
"""

import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

from postprocess import extract_json, validate_json, enforce_input_constraints, safe_parse


# ── Model config ──────────────────────────────────────────────────────────────
BASE_MODEL   = "google/gemma-2b-it"
GEN_ADAPTER  = "thrighun/jd-generator"
EDIT_ADAPTER = "thrighun/jd-editor"


# ── Prompts ───────────────────────────────────────────────────────────────────

SYSTEM_GEN = """You are an AI system designed to transform raw, unstructured job descriptions into structured, professional, and ATS-friendly job descriptions.

Your task:
- Extract AND rewrite content into a polished, professional format.
- Expand short or vague statements into clear, detailed, and actionable bullet points.
- Improve grammar, clarity, and tone while preserving original meaning.

Rules:
- Output MUST be valid JSON only.
- Follow the exact schema provided.
- Do NOT include explanations or extra text.
- Do NOT hallucinate unrealistic details.
- Do NOT copy sentences directly from input — always rewrite them professionally.

Enhancement Rules:
- Convert short phrases into complete, professional sentences.
- Add clarity by specifying intent.
- Use strong action verbs (manage, ensure, deliver, coordinate, analyze).
- Maintain ATS-friendly language with relevant keywords.
- Avoid vague wording like 'do', 'work on', 'handle'.
- Only include information explicitly present in the input.
- Do NOT infer industry or qualifications unless clearly mentioned.

Writing Style:
- Use concise but complete sentences.
- Each bullet point should be meaningful and self-contained.
- Maintain consistency across all sections.
- Generate ONLY ONE JSON object.
- Stop immediately after closing }."""

OUTPUT_SCHEMA = """
{
  "job_title": "",
  "location": "",
  "industry": "",
  "responsibilities": [],
  "requirements": [],
  "qualifications": [],
  "experience": [],
  "other_requirements": []
}
"""

SYSTEM_EDIT = """You are an AI system designed to MODIFY an existing job description JSON.

Your task:
- Update the given JSON based ONLY on the user instruction.
- Do NOT regenerate the entire job description.
- Make ONLY the necessary changes.

---

RULES:
- Output MUST be valid JSON only.
- Return ONLY ONE JSON object.
- Do NOT include explanations or extra text.
- Do NOT change fields that are not related to the instruction.
- Preserve all existing data unless modification is required.
- Do NOT hallucinate new fields or unnecessary content.

---

EDITING GUIDELINES:

1. ADD: Add new items to the correct field without removing existing ones.
2. REMOVE: Remove only the specified content.
3. UPDATE: Modify only the specified field value.
4. REPLACE: Replace only the mentioned parts.
5. REFINE: Improve wording while preserving meaning.
6. IMPROVE: Make content more professional or detailed without changing intent.
7. REGENERATE: Rewrite the entire JSON ONLY if explicitly requested."""


def gen_prompt(content: str) -> str:
    return f"""### SYSTEM:
{SYSTEM_GEN}

### USER:
Convert the following raw job description into structured JSON.

### Expected OUTPUT FORMAT:
Return a fully populated JSON following this schema:
{OUTPUT_SCHEMA}

### INPUT:
{content}

### RESPONSE:
"""


def edit_prompt(current_json: str, instruction: str) -> str:
    return f"""
### SYSTEM:
{SYSTEM_EDIT}

---

### USER:

Existing JSON:
{current_json}

Instruction:
{instruction}

---

### RESPONSE:
"""


# ── Model loader ──────────────────────────────────────────────────────────────

def load_model(device: str = "cuda"):
    """
    Load base model once and register both LoRA adapters.
    Returns (model, tokenizer).
    """
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    print("Loading base model...")
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        device_map={"": device},
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    )

    print(f"Loading gen adapter from {GEN_ADAPTER}...")
    model = PeftModel.from_pretrained(base, GEN_ADAPTER, adapter_name="gen")

    print(f"Loading edit adapter from {EDIT_ADAPTER}...")
    model.load_adapter(EDIT_ADAPTER, adapter_name="edit")

    model.eval()
    torch.set_grad_enabled(False)

    print("✅ Model ready. Adapters: 'gen', 'edit'")
    print(f"   Device : {next(model.parameters()).device}")
    print(f"   Dtype  : {next(model.parameters()).dtype}")

    return model, tokenizer


# ── Core inference ────────────────────────────────────────────────────────────

def run_generation(model, tokenizer, user_input: str) -> dict | None:
    """Switch to gen adapter and generate a structured JD from raw text."""
    model.set_adapter("gen")

    prompt  = gen_prompt(user_input)
    inputs  = tokenizer(prompt, return_tensors="pt").to(model.device)
    in_len  = inputs["input_ids"].shape[1]

    outputs = model.generate(
        **inputs,
        max_new_tokens=1536,
        do_sample=False,
        eos_token_id=tokenizer.eos_token_id,
        repetition_penalty=1.2,
    )

    text   = tokenizer.decode(outputs[0][in_len:], skip_special_tokens=True)
    result = extract_json(text)

    if result is None or not isinstance(result, dict):
        print("[WARN] Generation returned invalid JSON.")
        return None

    result = validate_json(result)
    result = enforce_input_constraints(result, user_input)
    return result


def run_editing(model, tokenizer, state: dict, instruction: str) -> dict:
    """Switch to edit adapter and apply instruction to existing JD JSON."""
    model.set_adapter("edit")

    prompt  = edit_prompt(json.dumps(state, indent=2), instruction)
    inputs  = tokenizer(prompt, return_tensors="pt").to(model.device)
    in_len  = inputs["input_ids"].shape[1]

    outputs = model.generate(
        **inputs,
        max_new_tokens=1024,
        do_sample=False,
        eos_token_id=tokenizer.eos_token_id,
        repetition_penalty=1.2,
    )

    text   = tokenizer.decode(outputs[0][in_len:], skip_special_tokens=True)
    result = safe_parse(text) or extract_json(text)

    if result is None or not isinstance(result, dict):
        print("[WARN] Edit returned invalid JSON — keeping previous state.")
        return state

    return validate_json(result)


# ── Unified JDModel class ─────────────────────────────────────────────────────

class JDModel:
    """
    Unified interface for generation + editing.

    state = None  →  next call triggers GENERATION
    state = dict  →  next call triggers EDITING

    Example:
        jd = JDModel()
        result = jd("Python developer, 2 years exp")   # generates
        result = jd("add Docker to requirements")       # edits
        jd.reset()                                      # start over
    """

    def __init__(self, device: str = "cuda"):
        self.model, self.tokenizer = load_model(device)
        self.state = None

    def __call__(self, user_input: str) -> dict | None:
        if self.state is None:
            print("[MODE] GENERATION")
            result = run_generation(self.model, self.tokenizer, user_input)
            if result is not None:
                self.state = result
            else:
                print("[ERROR] Generation failed — please try again.")
        else:
            print("[MODE] EDITING")
            self.state = run_editing(self.model, self.tokenizer, self.state, user_input)

        return self.state

    def reset(self):
        """Clear state — next input will trigger generation."""
        self.state = None
        print("[INFO] State reset. Next input will trigger generation.")

    def show(self):
        """Pretty-print current JD state."""
        if self.state:
            print(json.dumps(self.state, indent=2))
        else:
            print("[No JD generated yet]")