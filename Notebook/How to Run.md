# How to Run

## Requirements

- Google account (for Colab and Drive)
- Neo4j AuraDB free account — https://neo4j.com/cloud/aura

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

1. Upload `Application-of-Artificial-Intelligence-Methods-in-the-Requirements-Engineering-Management-Process.ipynb` to Google Colab
2. Set runtime to GPU — Runtime → Change runtime type → T4 GPU

---

## Step 4 — Enter Credentials When Prompted

The notebook will ask for your Neo4j credentials at runtime using secure input fields. Do not hardcode them anywhere.

```
Neo4j URI:      paste your AuraDB URI here
Neo4j Username: neo4j
Neo4j Password: paste your AuraDB password here
```

---

## Step 5 — Run All Cells

Runtime → Run all

The pipeline will install dependencies, load all models, and automatically run the banking sample dataset through all three stages.

Total runtime: approximately 8-12 minutes on Colab T4.

---

## Troubleshooting

**Neo4j connection refused**
Your AuraDB instance may have paused due to inactivity. Log in to neo4j.com and resume it, then re-run.

**Model not found**
Confirm the `NFR_Classifier` folder is in the root of your Google Drive and Drive is mounted. Check the path in the notebook configuration cell.

**CUDA out of memory**
Runtime → Restart runtime, then run all cells again.
