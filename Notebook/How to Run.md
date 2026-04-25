# How to Run

## Requirements

- Google account (for Colab and Drive)
- Neo4j AuraDB free account — https://neo4j.com/cloud/aura

---

## Files You Need

Before opening Colab, make sure you have all these files ready:

| File | Where to get it |
|---|---|
| `Application-of-Artificial-Intelligence-Methods-in-the-Requirements-Engineering-Management-Process.ipynb` | This repository |
| `nfr_keywords.yaml` | This repository |
| `promise-nfr.csv` | This repository |
| `NFR_Classifier/` folder | Google Drive link — see Step 1 |

---

## Step 1 — Get the NoRBERT Model

Download the fine-tuned NoRBERT model folder from Google Drive:

**[Download NFR_Classifier — Google Drive](https://drive.google.com/drive/folders/1fG1e6A_K3zXze-R7bmizlrAsgjstu4N0?usp=drive_link)**

Upload the `NFR_Classifier` folder to the root of your own Google Drive.

---

## Step 2 — Set Up Neo4j

1. Go to https://neo4j.com/cloud/aura and create a free account
2. Create a new free instance
3. Save the connection URI, username, and password when shown — you will need these later
4. Wait for the instance status to show Running

---

## Step 3 — Open the Notebook in Colab

1. Upload `Final_Pipeline.ipynb` to Google Colab
2. Set runtime to GPU — Runtime → Change runtime type → T4 GPU

---

## Step 4 — Upload Required Files to Colab

In the Colab left sidebar click the **Files** icon (folder icon).
Upload these three files directly into the Colab session storage:

- `nfr_keywords.yaml`
- `promise-nfr.csv`

They will appear at:
```
/content/nfr_keywords.yaml
/content/promise-nfr.csv
```

> **Note:** Colab session storage is temporary. If your session
> disconnects you will need to re-upload these three files.

---

## Step 5 — Enter Credentials When Prompted

The notebook will ask for your Neo4j credentials at runtime
using secure input fields. Do not hardcode them anywhere.

```
Neo4j URI:      paste your AuraDB URI here
Neo4j Username: neo4j
Neo4j Password: paste your AuraDB password here
```

---

## Step 6 — Run the Full Pipeline

Runtime → Run all

The pipeline will:
1. Install all dependencies
2. Load Qwen2.5-7B-Instruct (4-bit quantized) and NoRBERT models
3. Load the NFR keyword dictionary from `nfr_keywords.yaml`
4. Run the banking sample dataset through all three stages
5. Print a final metrics report showing Stage 1, Stage 2, and Stage 3 results
6. Build the Neo4j knowledge graph with detected dependencies and conflicts

Total runtime: approximately 8-12 minutes on Colab T4 GPU.

---

## Step 7 — Evaluate Stage 2 on the PROMISE Dataset

After the full pipeline has completed, run 'Stage 2 Evaluation' cell to evaluate
Stage 2 classification performance on the PROMISE NFR benchmark.
This compares individual classifiers against the ensemble
on 125 real-world software requirements.

---

## Troubleshooting

**Neo4j connection refused**
Your AuraDB instance may have paused. Log in to neo4j.com,
resume the instance, then re-run the connection cell only.

**Model not found**
Confirm `NFR_Classifier` folder is in the root of your Google
Drive and Drive is mounted. Check the path in the notebook
configuration cell.

**promise-nfr.csv not found**
Re-upload the file to Colab session storage.
Session storage clears when Colab disconnects.

**CUDA out of memory**
Runtime → Restart runtime, then run all cells again.
