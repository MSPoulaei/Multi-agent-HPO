# Multi-agent HPO

A complete, reproducible multi-agent system for hyperparameter optimization (HPO) of an image classifier. Internally it trains a ResNet‑9 on CIFAR‑10, but all LLM agents are anonymized and never see or mention the dataset/model names.

- Frameworks: LangGraph + LangChain (ChatOpenAI client)
- LLMs: Google Gemini 2.5 Flash/Pro via OpenAI-compatible endpoint
- Agents:
  1) Generator: Two consultants discuss; a supervisor finalizes hyperparameters.
  2) Executor: Trains/evaluates, runs heuristics + LLM analysis, saves metrics/plots (including val loss).
  3) Researcher: Uses Google Search (Gemini grounded search or Google CSE) to suggest actionable fixes.
- HPO space:
  - optimizer ∈ {adam, sgd}
  - learning_rate ∈ [1e-4, 1e-1]
  - train_batch_size ∈ {32, 64, 128, 256, 512}
  - weight_decay ∈ [1e-5, 1e-1]
  - label_smoothing ∈ [0.0, 0.2]

Important: Per your requirement, training uses only two datasets total:
- Train split is used for training.
- Test split is used for validation and final “test.” This will overestimate generalization; it’s recorded in summary.json as "val_and_test_same_dataset": true.

---

## Highlights

- Multi-agent HPO loop with consultation -> supervision -> training -> analysis -> web research -> next proposal
- Agents never see dataset/model names (anonymized prompts/logs)
- Heuristics + LLM-driven diagnosis from trajectories (loss/acc/F1)
- Plots: loss, val_loss (separate), accuracy, F1, combined
- Reproducible (seed), AMP on GPU by default
- Colab/Kaggle-friendly; handles OOM by stepping down batch size

---

## How it works (short)

1) Two consultant agents propose changes to the 5 hyperparameters.  
2) A supervising expert merges suggestions and outputs a final configuration.  
3) Executor trains the model for N epochs, tracks metrics, saves plots and CSVs, runs heuristics + LLM analysis to extract keywords.  
4) Researcher uses Google Search to suggest specific remedies, returned to the consultants for the next round.  
5) Loop until the HPO round budget is exhausted.

---

## Requirements

- Python 3.10+ (tested with 3.10/3.11)
- GPU recommended (Colab/Kaggle OK)
- pip install -r requirements.txt

---

## Setup

1) Install dependencies
```bash
pip install -r requirements.txt
```

2) Create a .env file in the project root (or set environment variables via your platform). Example:
```env
# OpenAI-compatible Gemini base URL
GEMINI_OPENAI_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai/

# One key per agent (or set GEMINI_API_KEY for all)
GEMINI_API_KEY_GEN_A=your_gen_a_key
GEMINI_API_KEY_GEN_B=your_gen_b_key
GEMINI_API_KEY_SUP=your_supervisor_key
GEMINI_API_KEY_EXEC=your_executor_key
GEMINI_API_KEY_RES=your_researcher_key
GEMINI_API_KEY=optional_shared_fallback

# Use Gemini grounded Google Search (true/false)
GEMINI_USE_GOOGLE_SEARCH=true

# Fallback Google Programmable Search (CSE)
GOOGLE_CSE_API_KEY=your_google_cse_api_key
GOOGLE_CSE_ID=your_cse_id
```

3) Verify your LLM model IDs
- Defaults:
  - Generator A: gemini-2.5-flash
  - Generator B: gemini-2.5-flash
  - Supervisor: gemini-2.5-pro
  - Executor analyzer: gemini-2.5-flash
  - Researcher: gemini-2.5-flash

Adjust via CLI if you prefer other Gemini IDs.

---

## Quick start

Local (GPU)
```bash
python main.py --rounds 10 --epochs 20 --consult-turns 2 --search-provider gemini
```

Colab
- Runtime -> Change runtime type -> GPU
- Upload the repo or copy the files into the workspace
- Run:
```bash
!pip install -r requirements.txt
# Write your .env (or set %env variables)
%env GEMINI_OPENAI_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai/
%env GEMINI_API_KEY=YOUR_KEY
# Optional per-role keys:
# %env GEMINI_API_KEY_GEN_A=...
# %env GEMINI_API_KEY_GEN_B=...
# ...

!python main.py --rounds 10 --epochs 20 --consult-turns 2 --search-provider gemini --draw-graph
```

Kaggle
- Set Accelerator: GPU in Notebook settings
- Internet access may be restricted depending on the environment; if grounded search fails, use `--search-provider cse` with CSE keys, or accept reduced research.
- In a cell:
```python
!pip install -r requirements.txt
import os
os.environ["GEMINI_OPENAI_BASE_URL"] = "https://generativelanguage.googleapis.com/v1beta/openai/"
os.environ["GEMINI_API_KEY"] = "YOUR_KEY"
# Optional per-role keys...
```
- Then:
```bash
!python main.py --rounds 10 --epochs 20 --consult-turns 2 --search-provider gemini --num-workers 2
```

---

## Sample runs and tips

- Fast smoke test (3 rounds, 8 epochs)
```bash
python main.py --rounds 3 --epochs 8 --consult-turns 1 --num-workers 2
```

- Full run (default)
```bash
python main.py --rounds 10 --epochs 20
```

- Try SGD explicitly (the agents can switch automatically, but you can nudge via initial hparams by editing src/graph/hpo_graph.py init or let the agents decide)
- Stronger data augmentation
```bash
python main.py --augment strong
```

- No scheduler (flat LR)
```bash
python main.py --scheduler none
```

- Use Google CSE search instead of Gemini grounded search
```bash
python main.py --search-provider cse
```

- Save a graph image of the workflow
```bash
python main.py --draw-graph
# Saves runs/<ts>/graph.png
```

- Draw the graph inside a notebook (fixes InvalidConcurrentGraphUpdate error):
```python
from IPython.display import Image, display
from src.graph.hpo_graph import build_hpo_graph

compiled, builder = build_hpo_graph(
    run_dir="runs/preview",
    epochs=20, patience=8, consult_turns=2,
    scheduler="cosine", augment="basic",
    num_workers=4, amp=True, save_checkpoints=True,
    anonymize=True, rounds=10,
    model_ids={
        "gen_a": "gemini-2.5-flash",
        "gen_b": "gemini-2.5-flash",
        "supervisor": "gemini-2.5-pro",
        "exec": "gemini-2.5-flash",
        "researcher": "gemini-2.5-flash",
    },
    search_provider="gemini",
)
png = builder.get_graph().draw_mermaid_png()
display(Image(png))
```

---

## Outputs

All artifacts are saved under `runs/<timestamp>/trial_<k>/`:
- metrics_epoch.csv (per-epoch: train_loss, val_loss, train_acc, val_acc, train_f1, val_f1, lr)
- summary.json (best_val_acc, test_acc/f1, effective_hparams, oom_adjusted, val_and_test_same_dataset)
- analysis.json (heuristics, trend features, keyword flags)
- analysis_llm.txt (LLM analyzer output)
- Plots:
  - loss.png (train vs val)
  - val_loss.png (dedicated val loss plot)
  - acc.png (train vs val accuracy)
  - f1.png (train vs val F1 macro)
  - combined.png
- checkpoint_best.pt (if enabled)
- runs/<timestamp>/trials_summary.csv (aggregated trials)
- runs/<timestamp>/best_overall.json

Note: Validation and test are the same dataset by design here.

---

## CLI options

- --rounds: HPO rounds (default 10)
- --epochs: epochs per trial (default 20)
- --patience: early stopping patience (default 8)
- --seed: random seed (default 1337)
- --output-dir: output root (default runs)
- --consult-turns: number of consultant exchanges before supervisor (default 2)
- --amp: AMP enabled (default True)
- --num-workers: dataloader workers (default 4; use 2 on Colab/Kaggle if needed)
- --scheduler: none|cosine (default cosine)
- --augment: basic|strong (default basic)
- --save-checkpoints: save best model per trial (default True)
- --anonymize: anonymize dataset/model names in any agent-facing text (default True)
- --models.gen_a|gen_b|supervisor|exec|researcher: LLM model ids
- --search-provider: gemini|cse (default gemini)
- --draw-graph: save a graph PNG of the workflow

---

## Environment variables (LLMs and Search)

LLM (Gemini OpenAI-compatible):
- GEMINI_OPENAI_BASE_URL (default: https://generativelanguage.googleapis.com/v1beta/openai/)
- GEMINI_API_KEY_GEN_A, GEMINI_API_KEY_GEN_B, GEMINI_API_KEY_SUP, GEMINI_API_KEY_EXEC, GEMINI_API_KEY_RES
- GEMINI_API_KEY (fallback for all roles)

Research:
- GEMINI_USE_GOOGLE_SEARCH=true enables Gemini grounded search
- GOOGLE_CSE_API_KEY + GOOGLE_CSE_ID for Google Programmable Search fallback

---

## Hyperparameter space (strictly enforced)

- optimizer ∈ {adam, sgd}
- learning_rate ∈ [1e-4, 1e-1]
- train_batch_size ∈ {32, 64, 128, 256, 512}
- weight_decay ∈ [1e-5, 1e-1]
- label_smoothing ∈ [0.0, 0.2]

Agents propose changes; supervisor finalizes within bounds. The executor validates/clamps any inputs to the valid ranges.

---

## Tips

- Speed:
  - Reduce --rounds or --epochs for quick iteration.
  - Keep AMP enabled on GPU.
  - Use --num-workers 2 on Colab/Kaggle if you see dataloader stalls.
- OOM handling:
  - The executor auto-steps down batch size (512 → 256 → 128 → 64 → 32) and records "oom_adjusted" in summary.json.
- Learning-rate schedules:
  - Cosine works well across many trials. If you see plateaus with low LR changes, try --scheduler none and let the agents adjust LR.
- Data augmentation:
  - "basic" uses flip/crop; "strong" adds mild color jitter. Agents can also adjust label smoothing and weight decay.

---

## Troubleshooting

- Graph drawing error: InvalidConcurrentGraphUpdate
  - Draw from the builder object, not the compiled Pregel:
    - Use `compiled, builder = build_hpo_graph(...)` and then `builder.get_graph().draw_mermaid_png()`.
  - Or run with `--draw-graph` to save runs/<ts>/graph.png.

- Gemini auth or 404 base URL
  - Ensure `GEMINI_OPENAI_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai/`
  - Ensure the correct API key is set for each role or set `GEMINI_API_KEY` for all.

- Web search returns empty
  - Try `--search-provider cse` with `GOOGLE_CSE_API_KEY` and `GOOGLE_CSE_ID`.
  - Some environments restrict outbound calls; the system will still run but with weaker research.

- Training very slow on Kaggle/Colab
  - Lower `--rounds` and `--epochs`, and set `--num-workers 2`. Keep AMP enabled.

---

## Notes on anonymization

- Agents are explicitly instructed not to use or ask for dataset/model names.
- Prompts/logs sent to agents are sanitized. The training code uses the actual dataset/model internally; this choice does not leak into agent-facing text.

---

## Project layout

```
.
├── main.py
├── requirements.txt
├── .env.example
├── src/
│   ├── agents/
│   │   ├── models.py
│   │   └── prompts.py
│   ├── executor/
│   │   ├── data.py              # train split for training, test split for validation/test
│   │   ├── train.py
│   │   ├── metrics.py
│   │   └── model/resnet9.py
│   ├── graph/
│   │   ├── state.py
│   │   └── hpo_graph.py
│   ├── tools/
│   │   └── google_search.py
│   └── utils/
│       ├── io.py
│       ├── plotting.py
│       ├── seed.py
│       └── sanitize.py
└── README.md
```

---

## License

Internal/research use. Add your preferred license if publishing.