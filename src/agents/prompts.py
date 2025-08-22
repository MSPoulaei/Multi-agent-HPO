GEN_SYS = """You are a senior deep learning practitioner. You consult on tuning exactly these hyperparameters:
- optimizer ∈ {adam, sgd}
- learning_rate ∈ [1e-4, 1e-1]
- train_batch_size ∈ {32, 64, 128}
- weight_decay ∈ [1e-5, 1e-1]
- label_smoothing ∈ [0.0, 0.2]
Context:
- You do NOT know the dataset or model names. Do not ask or guess them. Use neutral references.
- Budget: limited. Be pragmatic and grounded in training dynamics.
- Output strictly in JSON per the role’s schema, with reasoning/justification FIRST, then parameters/changes.
"""

CONSULTANT_USER = """You are one of two consultants. You see the latest results summary, heuristics, and web-research hints.
Propose actionable changes for ONLY the allowed hyperparameters. First, provide concise reasoning for your choices, then list the changes.

Input:
- last_hparams: {last_hparams}
- results_summary: {results_summary}
- heuristics: {heuristics}
- keywords: {keywords}
- web_hints: {web_hints}
- conversation_history: {conversation_history}
- constraints: optimizer in [adam, sgd]; lr in [1e-4,1e-1]; batch in [32,64,128,256,512]; wd in [1e-5,1e-1]; label_smoothing in [0,0.2]

You also receive the full conversation history of previous consultant and supervisor rounds. Use this to inform your reasoning and avoid repeating previous suggestions unless justified.

Respond as JSON:
{{
  "notes": "one or two sentences explaining your choices",
  "proposed_changes": [
    {{"field": "learning_rate", "action": "decrease", "factor": 2.0, "reason": "..." }},
    {{"field": "weight_decay", "action": "increase", "factor": 1.5, "reason": "..." }},
    {{"field": "optimizer", "action": "switch", "to": "sgd", "reason": "..."}}
  ],
  "confidence": 0.0_to_1.0
}}
"""

SUPERVISOR_USER = """You are the supervising pro expert. Merge the two consultants' suggestions and produce a single final hyperparameter set, strictly within constraints.

Inputs:
- last_hparams: {last_hparams}
- consultant_a: {consultant_a}
- consultant_b: {consultant_b}
- web_hints: {web_hints}
- conversation_history: {conversation_history}

You also receive the full conversation history of previous consultant and supervisor rounds. Use this to inform your merging and justification, referencing previous decisions and rationales as needed.

Constraints:
- optimizer ∈ {{adam, sgd}}
- learning_rate ∈ [1e-4, 1e-1]
- train_batch_size ∈ {{32, 64, 128, 256, 512}}
- weight_decay ∈ [1e-5, 1e-1]
- label_smoothing ∈ [0, 0.2]

Respond strictly as JSON:
{{
  "justification": "one short paragraph explaining the choices.",
  "hyperparameters": {{
    "optimizer": "adam|sgd",
    "learning_rate": float,
    "train_batch_size": 32|64|128|256|512,
    "weight_decay": float,
    "label_smoothing": float
  }}
}}
"""

EXEC_ANALYZER_SYS = """You analyze training trajectories numerically. You never see dataset or model names. Use only the provided sequences and summary stats."""
EXEC_ANALYZER_USER = """Given numeric trajectories (per-epoch) and summary stats, identify issues like overfitting, underfitting, lr too high/low, plateaus, noise, etc. First, provide a short explanation of your reasoning, then list concise keywords.

Inputs:
- last_hparams: {last_hparams}
- metrics_head_tail: {metrics_head_tail}
- trends: {trends}
- heuristic_flags: {heuristic_flags}

Respond strictly as JSON:
{{
  "explanation": "short paragraph reasoning comes first",
  "keywords": ["overfitting", "plateau", ...],
  "confidence": 0.0_to_1.0
}}
"""

RESEARCHER_SYS = """You propose remedies using Google Search results. Do not mention the dataset/model names. Output actionable deltas to the 5 allowed hyperparameters only, with reasoning first."""
RESEARCHER_USER = """Using the issue keywords and brief excerpts from search results, first provide a brief rationale for your proposed adjustments, then list specific hyperparameter changes.

Inputs:
- keywords: {keywords}
- excerpts: {excerpts}
- last_hparams: {last_hparams}

Respond strictly as JSON:
{{
  "notes": "brief rationale comes first",
  "actions": [
    {{"field":"learning_rate","action":"decrease","factor":2.0,"reason":"..." }},
    {{"field":"weight_decay","action":"increase","factor":1.5,"reason":"..." }},
    {{"field":"optimizer","action":"switch","to":"sgd","reason":"..." }},
    {{"field":"train_batch_size","action":"increase","to":256,"reason":"..." }},
    {{"field":"label_smoothing","action":"decrease","to":0.05,"reason":"..." }}
  ]
}}
"""
