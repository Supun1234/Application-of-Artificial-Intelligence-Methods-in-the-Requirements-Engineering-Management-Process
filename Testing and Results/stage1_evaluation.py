# ============================================================
# Stage 1 Completeness Detection — Full Evaluation
# Input: NER-tagged JSON with ACTOR, GOAL, RATIONALE tags
# ============================================================

import os
import json
import re
import time
import requests
import numpy as np
import pandas as pd
from nltk.tokenize import sent_tokenize
from pydantic import BaseModel, Field, ValidationError
from typing import List
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, classification_report, confusion_matrix)

# ══════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════
TAGGED_DATA_PATH = r"C:\Users\User\Desktop\New folder (2)\Full Pipeline\r1.json"
SAVE_PATH        = r"C:\Users\User\Desktop\New folder (2)\Full Pipeline\stage1_results.json"
OLLAMA_URL       = "http://localhost:11434/api/generate"
OLLAMA_MODEL     = "qwen2.5:7b-instruct"
THRESHOLD        = 0.80

# ══════════════════════════════════════════════════════════════
# OLLAMA LLM HANDLER — same as your pipeline
# ══════════════════════════════════════════════════════════════
class OllamaLLMHandler:
    def __init__(self, model=OLLAMA_MODEL, url=OLLAMA_URL):
        self.model = model
        self.url   = url

    def generate(self, prompt, temperature=0.1):
        try:
            response = requests.post(
                self.url,
                json={
                    "model":   self.model,
                    "prompt":  prompt,
                    "stream":  False,
                    "options": {"temperature": temperature,
                                "num_predict": 512}
                },
                timeout=120
            )
            raw   = response.json().get("response", "").strip()
            clean = re.sub(r"```json|```", "", raw).strip()
            match = re.search(r"\{.*\}", clean, re.DOTALL)
            if match:
                return json.loads(match.group(0))
        except Exception as e:
            pass
        return None

# ══════════════════════════════════════════════════════════════
# STAGE 1 — exact copy of your class
# ══════════════════════════════════════════════════════════════
class StructuredRequirement(BaseModel):
    actors:       List[str] = Field(default_factory=list)
    action:       str       = ""
    object:       str       = ""
    constraints:  List[str] = Field(default_factory=list)
    missing_info: List[str] = Field(default_factory=list)

class Stage1Structuring:
    def __init__(self, llm_handler):
        self.llm           = llm_handler
        self.clean_pattern = re.compile(r'[^a-zA-Z0-9\s\.,;\'\"]')
        self.vague_regex   = re.compile(
            r'\b(user-friendly|fast|efficient|robust|optimal|easy|'
            r'appropriate|better|clean|soon)\b', re.IGNORECASE)

    def _preprocess(self, text):
        text      = text.lower()
        text      = self.clean_pattern.sub('', text)
        sentences = sent_tokenize(text)
        return " ".join(sentences)

    def _construct_prompt(self, text):
        return f"""You are a Requirements Extraction Tool. Extract fields from the text.

        OUTPUT JSON:
        {{
            "actors": ["List of actors"],
            "action": "Main action verb",
            "object": "Target object",
            "constraints": ["List of constraints"],
            "missing_info": ["List CRITICAL gaps only"]
        }}

        RULES:
        1. If the sentence has an Actor, Action, and Object, it is valid. Leave "missing_info" EMPTY.
        2. Only populate "missing_info" if the sentence is VAGUE (e.g. "make it fast") or missing a Subject.
        3. Do NOT ask for extra details like "security" or "frequency" if they are not mentioned.

        EXAMPLES:
        Input: "The System shall maintain a Patient Database."
        Output: {{ "actors": ["System"], "action": "maintain", "object": "Patient Database", "missing_info": [] }}

        Input: "System needs backup."
        Output: {{ "actors": ["System"], "action": "backup", "missing_info": ["What data?", "Frequency?"] }}

        TARGET: "{text}"
        """

    def _call_llm_with_retry(self, prompt):
        max_attempts = 3
        for attempt in range(max_attempts):
            try:
                raw_json = self.llm.generate(prompt, temperature=0.1)
                if not raw_json:
                    raise ValueError("Empty response")
                return StructuredRequirement(**raw_json)
            except Exception as e:
                time.sleep(1)
        return StructuredRequirement(missing_info=["Analysis Failed"])

    def _calculate_completeness(self, data: StructuredRequirement):
        fields        = [data.actors, data.action, data.object, data.constraints]
        present       = sum(1 for f in fields if f)
        score_presence = present / 4.0
        vague_hits    = sum(1 for c in data.constraints
                            if self.vague_regex.search(c))
        penalty_vague   = min(vague_hits * 0.5, 1.0)
        penalty_missing = min(len(data.missing_info) * 0.5, 1.0)
        score = ((0.4 * score_presence) +
                 (0.3 * (1.0 - penalty_vague)) +
                 (0.3 * (1.0 - penalty_missing)))
        return round(max(score, 0.0), 2)

    def analyze(self, raw_text):
        clean_text = self._preprocess(raw_text)
        structured = self._call_llm_with_retry(
                         self._construct_prompt(clean_text))
        score      = self._calculate_completeness(structured)

        questions = []
        if score < 0.80:
            q_prompt = (f"Requirement: '{clean_text}'. "
                        f"Gaps: {structured.missing_info}. "
                        f"Generate 2 clarification questions in JSON: "
                        f"{{'questions': ['Q1', 'Q2']}}")
            q_res     = self.llm.generate(q_prompt)
            questions = (q_res.get('questions', [])
                         if q_res else ["Please clarify."])

        return {
            "text":       clean_text,
            "structured": structured.model_dump(),
            "score":      score,         # ← correct key
            "questions":  questions
        }

# ══════════════════════════════════════════════════════════════
# GROUND TRUTH from NER tags
# ACTOR + GOAL both present → complete
# Missing either             → incomplete
# ══════════════════════════════════════════════════════════════
def get_ground_truth(item):
    tags       = item["ner_tags"]
    has_actor  = any("ACTOR" in t for t in tags)
    has_goal   = any("GOAL"  in t for t in tags)
    return "complete" if (has_actor and has_goal) else "incomplete"

def get_missing(item):
    tags    = item["ner_tags"]
    missing = []
    if not any("ACTOR" in t for t in tags): missing.append("ACTOR")
    if not any("GOAL"  in t for t in tags): missing.append("GOAL")
    return missing

def reconstruct_text(item):
    tokens = item["tokens"]
    text   = ""
    for i, tok in enumerate(tokens):
        if tok in [".", ",", "!", "?", ":", ";", ")"]:
            text += tok
        elif i > 0 and tokens[i-1] == "(":
            text += tok
        else:
            text += (" " + tok) if text else tok
    return text.strip()

# ══════════════════════════════════════════════════════════════
# 1. LOAD DATA
# ══════════════════════════════════════════════════════════════
print("="*60)
print("Loading tagged requirements...")
with open(TAGGED_DATA_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

print(f"✅ Loaded {len(data)} tagged requirements")

ground_truths = []
texts         = []
missing_info  = []

for item in data:
    ground_truths.append(get_ground_truth(item))
    texts.append(reconstruct_text(item))
    missing_info.append(get_missing(item))

print(f"\nGround truth distribution:")
print(f"  Complete   : {ground_truths.count('complete')}")
print(f"  Incomplete : {ground_truths.count('incomplete')}")

# ══════════════════════════════════════════════════════════════
# 2. INIT STAGE 1
# ══════════════════════════════════════════════════════════════
print(f"\nConnecting to Ollama ({OLLAMA_MODEL})...")
try:
    import requests as req
    r = req.get("http://localhost:11434/api/tags", timeout=5)
    print(f"✅ Ollama connected")
except Exception as e:
    print(f"❌ Ollama not reachable: {e}")
    print("   Run: ollama serve")
    exit(1)

llm    = OllamaLLMHandler()
stage1 = Stage1Structuring(llm)

# Quick sanity check
print("\nRunning sanity check on one requirement...")
test_result = stage1.analyze("The system shall allow users to log in.")
print(f"  Text  : The system shall allow users to log in.")
print(f"  Score : {test_result['score']}")
print(f"  Actors: {test_result['structured']['actors']}")
print(f"  Action: {test_result['structured']['action']}")
print(f"  Pred  : {'complete' if test_result['score'] >= THRESHOLD else 'incomplete'}")

# ══════════════════════════════════════════════════════════════
# 3. LOAD EXISTING PROGRESS (auto-resume)
# ══════════════════════════════════════════════════════════════
saved_results = []
start_at      = 0

if os.path.exists(SAVE_PATH):
    with open(SAVE_PATH, "r") as f:
        saved = json.load(f)
    if "results" in saved and len(saved["results"]) > 0:
        saved_results = saved["results"]
        start_at      = len(saved_results)
        print(f"\n⚡ Resuming from requirement {start_at+1}/{len(data)}")
    else:
        print(f"\n🔹 Starting fresh — {len(data)} requirements")
else:
    print(f"\n🔹 Starting fresh — {len(data)} requirements")

import os

# ══════════════════════════════════════════════════════════════
# 4. RUN STAGE 1 ON ALL REQUIREMENTS
# ══════════════════════════════════════════════════════════════
print(f"   Threshold: {THRESHOLD}  |  Auto-saves every 25 requirements\n")

start_t = time.time()

for i, (text, gt, missing) in enumerate(
        zip(texts, ground_truths, missing_info)):

    if i < start_at:
        continue

    result = stage1.analyze(text)
    score  = result["score"]                          # correct key
    pred   = "complete" if score >= THRESHOLD else "incomplete"

    saved_results.append({
        "index":        i,
        "text":         text,
        "ground_truth": gt,
        "score":        score,
        "prediction":   pred,
        "questions":    result["questions"],
        "structured":   result["structured"],
        "missing_ner":  missing,
    })

    # Progress
    if (i+1) % 25 == 0 or i == 0:
        elapsed = time.time() - start_t
        done    = (i + 1) - start_at
        rate    = done / elapsed if elapsed > 0 else 1
        eta     = (len(data) - i - 1) / rate
        print(f"   {i+1:>4}/{len(data)}  |  "
              f"Elapsed: {elapsed/60:.1f}min  |  "
              f"ETA: {eta/60:.1f}min  |  "
              f"Score: {score:.2f}  |  Pred: {pred}")

    # Auto-save every 25
    if (i+1) % 25 == 0:
        with open(SAVE_PATH, "w") as f:
            json.dump({"results": saved_results}, f, indent=2)

# Final save
with open(SAVE_PATH, "w") as f:
    json.dump({"results": saved_results}, f, indent=2)

elapsed_total = time.time() - start_t
print(f"\n✅ All {len(data)} requirements processed in "
      f"{elapsed_total/60:.1f} minutes")

# ══════════════════════════════════════════════════════════════
# 5. METRICS
# ══════════════════════════════════════════════════════════════
y_true  = [r["ground_truth"] for r in saved_results]
y_pred  = [r["prediction"]   for r in saved_results]
scores  = [r["score"]        for r in saved_results]

acc  = accuracy_score(y_true, y_pred)
prec = precision_score(y_true, y_pred,
                       pos_label="incomplete", zero_division=0)
rec  = recall_score(y_true, y_pred,
                    pos_label="incomplete", zero_division=0)
f1   = f1_score(y_true, y_pred,
                pos_label="incomplete", zero_division=0)

print(f"\n{'='*65}")
print(f"📊  STAGE 1 COMPLETENESS DETECTION  (n={len(data)})")
print(f"{'='*65}")
print(f"  Accuracy  : {acc:.4f}  ({acc*100:.1f}%)")
print(f"  Precision : {prec:.4f}  (incomplete class)")
print(f"  Recall    : {rec:.4f}  (incomplete class)")
print(f"  F1-Score  : {f1:.4f}")

# ── Full report ───────────────────────────────────────────────
print(f"\n📋 Full Classification Report:")
print(classification_report(y_true, y_pred,
      target_names=["complete","incomplete"], zero_division=0))

# ── Confusion matrix ──────────────────────────────────────────
print(f"📋 Confusion Matrix:")
cm = confusion_matrix(y_true, y_pred,
                      labels=["complete","incomplete"])
print(pd.DataFrame(cm,
      index=["True complete","True incomplete"],
      columns=["Pred complete","Pred incomplete"]).to_string())

# ── Score distributions ───────────────────────────────────────
sc_complete   = [r["score"] for r in saved_results
                 if r["ground_truth"] == "complete"]
sc_incomplete = [r["score"] for r in saved_results
                 if r["ground_truth"] == "incomplete"]

print(f"\n📊 Score Distribution:")
print(f"  Complete   — Mean:{np.mean(sc_complete):.3f}  "
      f"Std:{np.std(sc_complete):.3f}  "
      f"Min:{min(sc_complete):.3f}  Max:{max(sc_complete):.3f}")
print(f"  Incomplete — Mean:{np.mean(sc_incomplete):.3f}  "
      f"Std:{np.std(sc_incomplete):.3f}  "
      f"Min:{min(sc_incomplete):.3f}  Max:{max(sc_incomplete):.3f}")

# ── Threshold sensitivity ─────────────────────────────────────
print(f"\n📊 Threshold Sensitivity:")
print(f"  {'Threshold':>10} {'Accuracy':>10} {'Precision':>10} "
      f"{'Recall':>8} {'F1':>8}")
print(f"  {'-'*50}")
for thresh in [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]:
    pt = ["complete" if s >= thresh else "incomplete" for s in scores]
    a  = accuracy_score(y_true, pt)
    p  = precision_score(y_true, pt, pos_label="incomplete", zero_division=0)
    r  = recall_score(y_true, pt,    pos_label="incomplete", zero_division=0)
    f  = f1_score(y_true, pt,        pos_label="incomplete", zero_division=0)
    mk = " ← current" if thresh == THRESHOLD else ""
    print(f"  {thresh:>10.2f} {a:>10.1%} {p:>10.1%} "
          f"{r:>8.1%} {f:>8.3f}{mk}")

# ── Automation metrics ────────────────────────────────────────
auto        = sum(1 for p in y_pred if p == "complete")
flagged     = sum(1 for p in y_pred if p == "incomplete")
auto_rate   = auto / len(data)
false_flags = sum(1 for t, p in zip(y_true, y_pred)
                  if t == "complete" and p == "incomplete")
ff_rate     = false_flags / y_true.count("complete")
missed      = sum(1 for t, p in zip(y_true, y_pred)
                  if t == "incomplete" and p == "complete")
miss_rate   = missed / y_true.count("incomplete") \
              if y_true.count("incomplete") > 0 else 0

avg_q = np.mean([len(r["questions"]) for r in saved_results
                 if r["prediction"] == "incomplete"])

print(f"\n📊 Operational Metrics:")
print(f"  Auto-approved : {auto}/{len(data)}  ({auto_rate:.1%})")
print(f"  Flagged       : {flagged}/{len(data)}")
print(f"  False flag rate: {ff_rate:.1%}  "
      f"(complete reqs wrongly flagged)")
print(f"  Miss rate     : {miss_rate:.1%}  "
      f"(incomplete reqs not caught)")
print(f"  Avg questions : {avg_q:.1f} per flagged req")

# ── Error analysis ────────────────────────────────────────────
fn_items = [r for r in saved_results
            if r["ground_truth"] == "incomplete"
            and r["prediction"]  == "complete"]
fp_items = [r for r in saved_results
            if r["ground_truth"] == "complete"
            and r["prediction"]  == "incomplete"]

print(f"\n🔍 Missed incomplete requirements ({len(fn_items)}):")
for r in fn_items[:5]:
    print(f"  Score:{r['score']:.2f} | {r['text'][:70]}...")

print(f"\n🔍 Wrongly flagged complete ({len(fp_items)}) — first 5:")
for r in fp_items[:5]:
    print(f"  Score:{r['score']:.2f} | {r['text'][:70]}...")

# ── Thesis table ──────────────────────────────────────────────
print(f"\n{'='*65}")
print(f"  THESIS TABLE — Stage 1 Completeness Detection (n={len(data)})")
print(f"{'='*65}")
print(f"  Accuracy          : {acc*100:.1f}%")
print(f"  Precision         : {prec*100:.1f}%  (incomplete detection)")
print(f"  Recall            : {rec*100:.1f}%  (incomplete detection)")
print(f"  F1-Score          : {f1:.3f}")
print(f"  Automation Rate   : {auto_rate:.1%}")
print(f"  False Flag Rate   : {ff_rate:.1%}")
print(f"  Miss Rate         : {miss_rate:.1%}")
print(f"  Threshold         : {THRESHOLD}")
print(f"  Avg questions/req : {avg_q:.1f}")
print(f"{'='*65}")
print(f"\n✅ Results saved to {SAVE_PATH}")
