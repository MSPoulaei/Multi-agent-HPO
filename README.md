Multi-agent HPO (LangGraph + Gemini) for an anonymized image classifier

Features
- Generator (3 LLM agents): Two consultants discuss hyperparameter changes for X turns; a pro expert outputs a final set.
- Executor: Trains and evaluates, saves metrics/plots, detects issues (heuristics + LLM analysis).
- Researcher: Uses Google Search (Gemini grounded search or CSE fallback) to propose actionable changes, fed back to Generator.
- Anonymization: Agents and saved text never mention real dataset/model names.

Run
- pip install -r requirements.txt
- Create .env from .env.example (set your Gemini keys and optional CSE keys)
- python main.py --rounds 10 --epochs 20 --consult-turns 2

CLI flags (common)
- --rounds, --epochs, --patience, --seed, --output-dir
- --consult-turns
- --models.gen_a, --models.gen_b, --models.supervisor, --models.exec, --models.researcher
- --amp, --scheduler, --augment, --num-workers
- --save-checkpoints, --anonymize
- --search-provider [gemini|cse]

Outputs
- runs/<timestamp>/trial_<k>/:
  - metrics_epoch.csv
  - summary.json
  - analysis.json
  - analysis_llm.txt
  - plots: loss.png, acc.png, f1.png, val_loss.png, combined.png
  - checkpoint_best.pt (if enabled)
- runs/<timestamp>/trials_summary.csv (all trials)