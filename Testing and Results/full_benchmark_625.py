# ============================================================
# Full Stage 2 Benchmark — all 625 PROMISE requirements
# NoRBERT: CPU | LLM: Ollama | Rules: CPU
# Auto-saves every 50 requirements — safe to restart
# Run: python full_benchmark_625.py
# ============================================================

import os
import re
import json
import time
import yaml
import numpy as np
import pandas as pd
import torch
import requests
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, classification_report, confusion_matrix)

# ══════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════
PROMISE_PATH  = r"C:\Users\User\Desktop\New folder (2)\Full Pipeline\Promise_NFR_dataset_orginal.csv"
NOBERT_PATH   = r"C:\Users\User\Desktop\New folder (2)\Full Pipeline\BestModel"
KEYWORDS_PATH = r"C:\Users\User\Desktop\New folder (2)\Full Pipeline\nfr_keywords.yaml"
SAVE_PATH     = r"C:\Users\User\Desktop\New folder (2)\Full Pipeline\full_625_results.json"
OLLAMA_URL    = "http://localhost:11434/api/generate"
OLLAMA_MODEL  = "qwen2.5:7b-instruct"

# ══════════════════════════════════════════════════════════════
# LABEL MAPS
# ══════════════════════════════════════════════════════════════
ID2LABEL = {
    0:"FR", 1:"A",  2:"FT", 3:"L",  4:"LF",
    5:"MN", 6:"O",  7:"PE", 8:"PO", 9:"SC", 10:"US"
}
NOBERT_CODE_MAP = {
    "FR":"FR",  "A":"Availability",  "FT":"Reliability",
    "L":"Legal","LF":"Look_and_Feel","MN":"Maintainability",
    "O":"Operational","PE":"Performance","PO":"Portability",
    "SC":"Security","US":"Usability"
}
SUBTYPE_MAP = {
    "F":"FR",   "PE":"Performance","SC":"Security","SE":"Security",
    "US":"Usability","A":"Availability","FT":"Reliability",
    "L":"Legal","LF":"Look_and_Feel","MN":"Maintainability",
    "O":"Operational","PO":"Portability"
}

# ══════════════════════════════════════════════════════════════
# 1. LOAD DATASET
# ══════════════════════════════════════════════════════════════
print("="*60)
print("Loading PROMISE dataset...")
df = pd.read_csv(PROMISE_PATH, sep=";")
df.columns = df.columns.str.strip()
df["truth_binary"]  = df["F"].apply(lambda x: "FR" if x == 1 else "NFR")
df["truth_subtype"] = df["class"].str.strip().map(SUBTYPE_MAP).fillna("Unknown")
df = df[df["truth_binary"].isin(["FR","NFR"])].reset_index(drop=True)

print(f"✅ {len(df)} requirements loaded")
print(df["truth_binary"].value_counts().to_string())
TOTAL = len(df)

# ══════════════════════════════════════════════════════════════
# 2. LOAD MODELS
# ══════════════════════════════════════════════════════════════

# ── Keywords ─────────────────────────────────────────────────
if os.path.exists(KEYWORDS_PATH):
    with open(KEYWORDS_PATH, "r") as f:
        keywords = yaml.safe_load(f)
    print(f"✅ Keywords loaded from {KEYWORDS_PATH}")
else:
    keywords = {
        "Performance":     ["fast","ms","seconds","response time",
                            "throughput","latency","concurrent"],
        "Security":        ["encrypt","password","auth","ssl","tls",
                            "access control","certificate"],
        "Usability":       ["user-friendly","easy to use","intuitive",
                            "accessible","navigate","interface"],
        "Availability":    ["available","uptime","fault tolerant",
                            "failover","downtime","backup"],
        "Scalability":     ["scale","grow","expand",
                            "concurrent users","future"],
        "Maintainability": ["modular","documented","maintainable",
                            "configurable","extensible"],
        "Portability":     ["platform","browser","operating system",
                            "compatible","mobile","desktop"],
        "Reliability":     ["reliable","recover","redundant","mtbf"],
    }
    print("⚠️  keywords.yaml not found — using defaults")

# ── NoRBERT ───────────────────────────────────────────────────
print(f"\nLoading NoRBERT from {NOBERT_PATH}...")
tokenizer = AutoTokenizer.from_pretrained(NOBERT_PATH)
nb_model  = AutoModelForSequenceClassification.from_pretrained(
                NOBERT_PATH).to("cpu")
nb_model.eval()
print("✅ NoRBERT loaded on CPU")

# ── Ollama check ──────────────────────────────────────────────
print(f"\nChecking Ollama ({OLLAMA_MODEL})...")
try:
    r = requests.get("http://localhost:11434/api/tags", timeout=5)
    print(f"✅ Ollama connected")
except Exception as e:
    print(f"❌ Ollama not reachable: {e}")
    print("   Run: ollama serve")
    exit(1)

# ══════════════════════════════════════════════════════════════
# 3. CLASSIFIER FUNCTIONS
# ══════════════════════════════════════════════════════════════

def classify_rules(text):
    text_lower = text.lower()
    scores = {cat: 0 for cat in keywords}
    for cat, kws in keywords.items():
        if any(kw in text_lower for kw in kws):
            scores[cat] += 1
    if sum(scores.values()) == 0:
        return "FR", "None"
    best = max(scores, key=scores.get)
    return "NFR", best


def classify_nobert(text):
    inputs = tokenizer(
        text, return_tensors="pt",
        truncation=True, padding=True, max_length=128
    )
    with torch.no_grad():
        logits   = nb_model(**inputs).logits
        probs    = torch.softmax(logits, dim=1).squeeze().numpy()
    pred_id  = int(np.argmax(probs))
    raw_code = ID2LABEL.get(pred_id, "Unknown")
    subtype  = NOBERT_CODE_MAP.get(raw_code, raw_code)
    if raw_code == "FR":
        return "FR", "None"
    return "NFR", subtype


def classify_llm(text):
    prompt = f"""You are an expert Requirements Engineer specializing in
classifying software requirements.

DEFINITIONS:
- FR (Functional Requirement): Describes WHAT the system does.
  Specifies a behavior, feature, or function. Uses action verbs
  like "shall allow", "shall process", "shall display", "shall store".

- NFR (Non-Functional Requirement): Describes HOW WELL the system
  performs. Specifies quality attributes, constraints, or standards.

NFR CATEGORIES AND STRONG SIGNALS:
- Performance: response time, speed, latency, throughput, seconds,
  milliseconds, concurrent users, load, capacity
- Security: encrypt, authenticate, authorize, password, access control,
  SSL, TLS, certificate, audit, log
- Usability: user-friendly, easy to use, intuitive, accessible,
  learn, navigate, interface appearance
- Reliability: available, uptime, fault tolerant, recover, backup,
  failover, MTBF, downtime
- Maintainability: modular, documented, testable, configurable,
  upgradeable, extensible
- Portability: platform, browser, operating system, device,
  compatible, mobile, desktop
- Scalability: scale, grow, expand, additional users, future

CRITICAL RULE: If the requirement mentions TIME LIMITS, PERCENTAGES,
QUANTITIES as constraints, SIZE LIMITS, or QUALITY STANDARDS —
it is almost certainly NFR even if it uses "shall".

EXAMPLES:
"The system shall refresh the display every 60 seconds." -> NFR (Performance)
"The system shall encrypt all passwords using SHA-256."  -> NFR (Security)
"The system shall allow users to log in."                -> FR
"The application shall be available 99.9% of the time." -> NFR (Reliability)
"The system shall process 1000 transactions per second." -> NFR (Performance)
"The system shall display a list of products."           -> FR

NOW CLASSIFY THIS REQUIREMENT:
"{text}"

Respond in JSON only, no explanation outside JSON:
{{
  "type": "FR" or "NFR",
  "subtype": null if FR, or one of [Performance, Security, Usability,
             Reliability, Maintainability, Portability, Scalability] if NFR,
  "confidence": "high", "medium", or "low",
  "reasoning": "one sentence explanation"
}}"""

    try:
        resp = requests.post(
            OLLAMA_URL,
            json={
                "model":   OLLAMA_MODEL,
                "prompt":  prompt,
                "stream":  False,
                "options": {"temperature": 0.1, "num_predict": 256}
            },
            timeout=120
        )
        raw   = resp.json().get("response", "").strip()
        clean = re.sub(r"```json|```", "", raw).strip()
        match = re.search(r"\{.*\}", clean, re.DOTALL)
        if match:
            data = json.loads(match.group(0))
            return data.get("type","Unknown"), data.get("subtype") or "None"
    except Exception as e:
        pass
    return "Unknown", "None"


def resolve_subtype(r_nobert, r_llm, r_rules, ensemble):
    if ensemble != "NFR":
        return "None"
    for sub in [r_nobert[1], r_llm[1], r_rules[1]]:
        if sub and sub not in ("None","Unknown","FR","NFR_Subtype"):
            return sub
    return "None"


def ensemble_vote(r_rules, r_llm, r_nobert):
    votes = [v for v in [r_rules[0], r_llm[0], r_nobert[0]]
             if v != "Unknown"]
    if not votes:
        return "Unknown"
    return "FR" if votes.count("FR") >= votes.count("NFR") else "NFR"

# ══════════════════════════════════════════════════════════════
# 4. LOAD EXISTING PROGRESS (auto-resume)
# ══════════════════════════════════════════════════════════════
results = {
    "rules_preds":    [],
    "llm_preds":      [],
    "nobert_preds":   [],
    "ensemble_preds": [],
    "nobert_subs":    [],
    "llm_subs":       [],
    "completed":      0
}

if os.path.exists(SAVE_PATH):
    with open(SAVE_PATH, "r") as f:
        saved = json.load(f)
    if "completed" in saved and saved["completed"] > 0:
        results  = saved
        start_at = saved["completed"]
        print(f"\n⚡ Resuming from requirement {start_at+1}/{TOTAL}")
    else:
        start_at = 0
        print(f"\n🔹 Starting fresh — {TOTAL} requirements")
else:
    start_at = 0
    print(f"\n🔹 Starting fresh — {TOTAL} requirements")

# ══════════════════════════════════════════════════════════════
# 5. RUN EVALUATION
# ══════════════════════════════════════════════════════════════
print(f"   NoRBERT: CPU | LLM: Ollama ({OLLAMA_MODEL}) | Rules: CPU")
print(f"   Auto-saves every 50 requirements\n")

start_t = time.time()

for i, row in enumerate(df.itertuples()):

    # Skip already completed
    if i < start_at:
        continue

    text = str(row.RequirementText).strip()

    r_rules  = classify_rules(text)
    r_nobert = classify_nobert(text)
    r_llm    = classify_llm(text)

    ensemble = ensemble_vote(r_rules, r_llm, r_nobert)
    subtype  = resolve_subtype(r_nobert, r_llm, r_rules, ensemble)

    results["rules_preds"].append(r_rules[0])
    results["llm_preds"].append(r_llm[0])
    results["nobert_preds"].append(r_nobert[0])
    results["ensemble_preds"].append(ensemble)
    results["nobert_subs"].append(r_nobert[1])
    results["llm_subs"].append(r_llm[1])
    results["completed"] = i + 1

    # Progress
    if (i+1) % 25 == 0 or i == 0:
        elapsed = time.time() - start_t
        done    = (i + 1) - start_at
        rate    = done / elapsed if elapsed > 0 else 1
        eta     = (TOTAL - i - 1) / rate
        print(f"   {i+1:>4}/{TOTAL}  |  "
              f"Elapsed: {elapsed/60:.1f}min  |  "
              f"ETA: {eta/60:.1f}min")

    # Auto-save every 50
    if (i+1) % 50 == 0:
        with open(SAVE_PATH, "w") as f:
            json.dump(results, f)

# Final save
with open(SAVE_PATH, "w") as f:
    json.dump(results, f)

elapsed_total = time.time() - start_t
print(f"\n✅ All {TOTAL} requirements processed in "
      f"{elapsed_total/60:.1f} minutes")

# ══════════════════════════════════════════════════════════════
# 6. COMPUTE METRICS
# ══════════════════════════════════════════════════════════════
y_true = df["truth_binary"].tolist()

rules_preds    = results["rules_preds"]
llm_preds      = results["llm_preds"]
nobert_preds   = results["nobert_preds"]
ensemble_preds = results["ensemble_preds"]

# Subtype ground truth and predictions
subtype_truths = []
subtype_preds  = []
for i, row in enumerate(df.itertuples()):
    if row.truth_binary == "NFR" and ensemble_preds[i] == "NFR":
        nb_sub  = results["nobert_subs"][i]
        llm_sub = results["llm_subs"][i]
        sub = "None"
        for s in [nb_sub, llm_sub]:
            if s and s not in ("None","Unknown","FR","NFR_Subtype"):
                sub = s
                break
        subtype_truths.append(row.truth_subtype)
        subtype_preds.append(sub)

# ── Metrics function ──────────────────────────────────────────
def metrics(y_true, y_pred, name):
    pairs = [(t,p) for t,p in zip(y_true,y_pred)
             if t in ("FR","NFR") and p in ("FR","NFR")]
    if not pairs:
        print(f"  {name:<36} — no data")
        return None
    yt = [p[0] for p in pairs]
    yp = [p[1] for p in pairs]
    acc  = accuracy_score(yt, yp)
    prec = precision_score(yt, yp, pos_label="NFR", zero_division=0)
    rec  = recall_score(yt, yp,    pos_label="NFR", zero_division=0)
    f1   = f1_score(yt, yp,        pos_label="NFR", zero_division=0)
    print(f"  {name:<36} n={len(yt)}  Acc:{acc:.1%}  "
          f"Prec:{prec:.1%}  Rec:{rec:.1%}  F1:{f1:.3f}")
    return dict(model=name, n=len(yt),
                accuracy=acc, precision=prec, recall=rec, f1=f1)

# ── Print results ─────────────────────────────────────────────
print(f"\n{'='*72}")
print(f"📊  BINARY FR/NFR RESULTS  (PROMISE, n={TOTAL})")
print(f"{'='*72}")
rows = []
rows.append(metrics(y_true, rules_preds,    "Rules Classifier"))
rows.append(metrics(y_true, llm_preds,      "LLM (Qwen2.5-7B-Instruct)"))
rows.append(metrics(y_true, nobert_preds,   "NoRBERT (10-fold CV)"))
rows.append(metrics(y_true, ensemble_preds, "Ensemble — 3 model ★"))

# ── Confusion matrix ──────────────────────────────────────────
print(f"\n📋 Ensemble Confusion Matrix:")
cm = confusion_matrix(y_true, ensemble_preds, labels=["FR","NFR"])
print(pd.DataFrame(cm,
      index=["True FR","True NFR"],
      columns=["Pred FR","Pred NFR"]).to_string())

# ── Full report ───────────────────────────────────────────────
print(f"\n📋 Ensemble Full Classification Report:")
print(classification_report(
    [t for t in y_true if t in ("FR","NFR")],
    [p for p in ensemble_preds if p in ("FR","NFR")],
    target_names=["FR","NFR"], zero_division=0))

# ── Subtype accuracy ──────────────────────────────────────────
print(f"\n{'='*72}")
print(f"📊  NFR SUBTYPE ACCURACY  (n={len(subtype_truths)})")
print(f"{'='*72}")
if subtype_truths:
    sub_acc = accuracy_score(subtype_truths, subtype_preds)
    print(f"  Overall: {sub_acc:.1%}")
    print(f"\n  Per-subtype breakdown:")
    print(classification_report(subtype_truths, subtype_preds,
                                zero_division=0))
else:
    print("  No subtype data")

# ── Thesis table ──────────────────────────────────────────────
print(f"\n{'='*72}")
print(f"  TABLE 3 — Classification Performance (PROMISE NFR, n={TOTAL})")
print(f"{'='*72}")
print(f"  {'Model':<36} {'n':>5} {'Accuracy':>9} "
      f"{'Precision':>10} {'Recall':>8} {'F1':>8}")
print(f"  {'-'*68}")
for r in rows:
    if r:
        print(f"  {r['model']:<36} {r['n']:>5} {r['accuracy']:>9.1%} "
              f"{r['precision']:>10.1%} {r['recall']:>8.1%} "
              f"{r['f1']:>8.3f}")
print(f"  {'='*68}")

# ── Error analysis ────────────────────────────────────────────
df_eval = df.copy()
df_eval["ensemble_pred"] = ensemble_preds
errors  = df_eval[df_eval["truth_binary"] != df_eval["ensemble_pred"]]
fp      = errors[errors["ensemble_pred"] == "NFR"]
fn      = errors[errors["ensemble_pred"] == "FR"]

print(f"\n🔍 Ensemble errors: {len(errors)}/{TOTAL}")
print(f"   FP (pred NFR, true FR): {len(fp)}")
print(f"   FN (pred FR, true NFR): {len(fn)}")
if len(fn) > 0:
    print(f"\n   FN breakdown by subtype:")
    print(fn["truth_subtype"].value_counts().to_string())

print(f"\n✅ Done. Results saved to {SAVE_PATH}")
