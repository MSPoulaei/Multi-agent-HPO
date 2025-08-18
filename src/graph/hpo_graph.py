import os
import json
import pandas as pd
from typing import Dict, Any
from langgraph.graph import StateGraph, END

from ..agents.models import llm_for_role
from ..agents.prompts import GEN_SYS, CONSULTANT_USER, SUPERVISOR_USER, EXEC_ANALYZER_SYS, EXEC_ANALYZER_USER, RESEARCHER_SYS, RESEARCHER_USER
from ..executor.train import train_and_eval, validate_hparams
from ..executor.metrics import heuristic_analysis
from ..tools.google_search import web_search
from ..utils.io import save_json, save_csv
from ..utils.sanitize import anonymize_text
from .state import HPOState

def clamp_actions_to_ranges(last_hp: Dict[str, Any], actions: list):
    hp = dict(last_hp) if last_hp else {
        "optimizer":"adam", "learning_rate":3e-3, "train_batch_size":128, "weight_decay":5e-4, "label_smoothing":0.05
    }
    for a in actions:
        field = a.get("field")
        if field not in ["optimizer","learning_rate","train_batch_size","weight_decay","label_smoothing"]:
            continue
        act = a.get("action")
        if field == "optimizer":
            to = a.get("to")
            if to in ["adam","sgd"]:
                hp["optimizer"] = to
        elif field in ["learning_rate","weight_decay"]:
            if act == "increase":
                factor = float(a.get("factor", 1.1))
                hp[field] = hp[field] * max(factor, 0.1)
            elif act == "decrease":
                factor = float(a.get("factor", 2.0))
                hp[field] = hp[field] / max(factor, 1.01)
            elif "to" in a:
                hp[field] = float(a["to"])
        elif field == "train_batch_size":
            if "to" in a:
                hp[field] = int(a["to"])
            elif act == "increase":
                choices = [32,64,128,256,512]
                greater = [x for x in choices if x > hp[field]]
                hp[field] = min(greater) if greater else choices[-1]
            elif act == "decrease":
                choices = [32,64,128,256,512]
                lesser = [x for x in choices if x < hp[field]]
                hp[field] = max(lesser) if lesser else choices[0]
        elif field == "label_smoothing":
            if "to" in a:
                hp[field] = float(a["to"])
            elif act == "increase":
                hp[field] += float(a.get("delta", 0.02))
            elif act == "decrease":
                hp[field] -= float(a.get("delta", 0.02))
    return validate_hparams(hp)

def parse_json_safe(text: str, default: Any):
    try:
        return json.loads(text)
    except Exception:
        import re
        m = re.search(r"\{.*\}", text, re.S)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                return default
        return default

def build_hpo_graph(
    run_dir: str,
    epochs: int,
    patience: int,
    consult_turns: int,
    scheduler: str,
    augment: str,
    num_workers: int,
    amp: bool,
    save_checkpoints: bool,
    anonymize: bool,
    rounds: int,
    model_ids: Dict[str, str],
    search_provider: str = "gemini",
):
    os.makedirs(run_dir, exist_ok=True)

    def init_state(_):
        return {
            "trial_idx": 0,
            "rounds": rounds,
            "consult_turn": 0,
            "consult_limit": consult_turns,
            "last_hparams": {
                "optimizer":"adam","learning_rate":3e-3,"train_batch_size":128,"weight_decay":5e-4,"label_smoothing":0.05
            },
            "gen_consult_a": [],
            "gen_consult_b": [],
            "web_hints": {"actions": [], "notes": ""},
            "run_dir": run_dir,
            "anonymize": anonymize,
            "model_ids": model_ids,
            "trainer_cfg": {
                "epochs": epochs, "patience": patience, "scheduler": scheduler, "augment": augment,
                "num_workers": num_workers, "amp": amp, "save_checkpoints": save_checkpoints
            },
            "best_so_far": {"val_acc": -1.0, "trial_idx": -1, "hparams": None},
            "trials_summary_rows": [],
            "search_provider": search_provider,
        }

    builder = StateGraph(HPOState)

    def consultant_a_node(state: HPOState):
        llm = llm_for_role("gen_a", state["model_ids"]["gen_a"], temperature=0.4)
        inp = CONSULTANT_USER.format(
            last_hparams=json.dumps(state["last_hparams"]),
            results_summary=json.dumps(state.get("train_results", {})),
            heuristics=json.dumps(state.get("analysis", {}).get("trends", {})),
            keywords=json.dumps(state.get("keywords", [])),
            web_hints=json.dumps(state.get("web_hints", {}))
        )
        resp = llm.invoke([("system", GEN_SYS), ("user", inp)]).content
        data = parse_json_safe(resp, {"proposed_changes": [], "notes":"", "confidence":0.5})
        # Append-only update, and increment consult_turn
        return {"gen_consult_a": [data], "consult_turn": state["consult_turn"] + 1}

    def consultant_b_node(state: HPOState):
        llm = llm_for_role("gen_b", state["model_ids"]["gen_b"], temperature=0.4)
        inp = CONSULTANT_USER.format(
            last_hparams=json.dumps(state["last_hparams"]),
            results_summary=json.dumps(state.get("train_results", {})),
            heuristics=json.dumps(state.get("analysis", {}).get("trends", {})),
            keywords=json.dumps(state.get("keywords", [])),
            web_hints=json.dumps(state.get("web_hints", {}))
        )
        resp = llm.invoke([("system", GEN_SYS), ("user", inp)]).content
        data = parse_json_safe(resp, {"proposed_changes": [], "notes":"", "confidence":0.5})
        return {"gen_consult_b": [data], "consult_turn": state["consult_turn"] + 1}

    def consult_router(state: HPOState):
        if state["consult_turn"] < state["consult_limit"]:
            return "consultant_a" if state["consult_turn"] % 2 == 0 else "consultant_b"
        else:
            return "supervisor"

    def supervisor_node(state: HPOState):
        llm = llm_for_role("supervisor", state["model_ids"]["supervisor"], temperature=0.2)
        a = state["gen_consult_a"][-1] if state.get("gen_consult_a") else {"proposed_changes": []}
        b = state["gen_consult_b"][-1] if state.get("gen_consult_b") else {"proposed_changes": []}
        inp = SUPERVISOR_USER.format(
            last_hparams=json.dumps(state["last_hparams"]),
            consultant_a=json.dumps(a),
            consultant_b=json.dumps(b),
            web_hints=json.dumps(state.get("web_hints", {}))
        )
        resp = llm.invoke([("system", GEN_SYS), ("user", inp)]).content
        data = parse_json_safe(resp, {"hyperparameters": state["last_hparams"], "justification": ""})
        hp = validate_hparams(data.get("hyperparameters", state["last_hparams"]))
        return {"supervisor_out": {"hyperparameters": hp, "justification": data.get("justification","")}}

    def executor_node(state: HPOState):
        trial_idx = state["trial_idx"]
        trial_dir = os.path.join(state["run_dir"], f"trial_{trial_idx:03d}")
        hp = state["supervisor_out"]["hyperparameters"]

        summary, metrics_df = train_and_eval(
            trial_dir=trial_dir, hp=hp,
            epochs=state["trainer_cfg"]["epochs"],
            patience=state["trainer_cfg"]["patience"],
            scheduler_type=state["trainer_cfg"]["scheduler"],
            augment=state["trainer_cfg"]["augment"],
            num_workers=state["trainer_cfg"]["num_workers"],
            amp=state["trainer_cfg"]["amp"],
            save_checkpoints=state["trainer_cfg"]["save_checkpoints"],
            seed=1337,
        )

        val_acc = float(summary["best_val_acc"])
        best_so_far = state["best_so_far"]
        if val_acc > best_so_far["val_acc"]:
            best_so_far = {"val_acc": val_acc, "trial_idx": trial_idx, "hparams": summary["effective_hparams"]}

        row = {
            "trial_idx": trial_idx,
            "optimizer": summary["effective_hparams"]["optimizer"],
            "learning_rate": summary["effective_hparams"]["learning_rate"],
            "train_batch_size": summary["effective_hparams"]["train_batch_size"],
            "weight_decay": summary["effective_hparams"]["weight_decay"],
            "label_smoothing": summary["effective_hparams"]["label_smoothing"],
            "best_val_acc": summary["best_val_acc"],
            "test_acc": summary["test_acc"],
            "test_f1": summary["test_f1"],
            "oom_adjusted": summary["oom_adjusted"],
        }

        metrics_path = os.path.join(trial_dir, "metrics_epoch.csv")
        return {
            "train_results": summary,
            "metrics_df_path": metrics_path,
            "best_so_far": best_so_far,
            "trials_summary_rows": [row],
        }

    def analyzer_node(state: HPOState):
        llm = llm_for_role("exec", state["model_ids"]["exec"], temperature=0.2)
        metrics_df = pd.read_csv(state["metrics_df_path"])
        def clip(arr, k=5):
            arr = list(arr)
            if len(arr) <= 2*k:
                return arr
            return arr[:k] + ["..."] + arr[-k:]
        m = {
            "train_loss": clip(metrics_df["train_loss"].round(4).tolist()),
            "val_loss": clip(metrics_df["val_loss"].round(4).tolist()),
            "train_acc": clip(metrics_df["train_acc"].round(4).tolist()),
            "val_acc": clip(metrics_df["val_acc"].round(4).tolist()),
            "train_f1": clip(metrics_df["train_f1"].round(4).tolist()),
            "val_f1": clip(metrics_df["val_f1"].round(4).tolist()),
        }
        feats, keywords_h = heuristic_analysis(metrics_df)
        inp = EXEC_ANALYZER_USER.format(
            last_hparams=json.dumps(state["supervisor_out"]["hyperparameters"]),
            metrics_head_tail=json.dumps(m),
            trends=json.dumps(feats),
            heuristic_flags=json.dumps(feats["flags"])
        )
        resp = llm.invoke([("system", EXEC_ANALYZER_SYS), ("user", inp)]).content
        data = parse_json_safe(resp, {"keywords": keywords_h, "explanation": "", "confidence": 0.6})
        merged_kw = list(set(keywords_h + data.get("keywords", [])))
        trial_dir = os.path.join(state["run_dir"], f"trial_{state['trial_idx']:03d}")
        with open(os.path.join(trial_dir, "analysis_llm.txt"), "w") as f:
            f.write(resp)
        save_json(os.path.join(trial_dir, "analysis.json"), {"heuristics": feats, "keywords": merged_kw})
        return {"analysis": {"trends": feats, "keywords": merged_kw}, "keywords": merged_kw}

    def researcher_node(state: HPOState):
        llm = llm_for_role("researcher", state["model_ids"]["researcher"], temperature=0.3)
        query = " ".join(state.get("keywords", [])) + " image classification training remedies generalization optimization"
        excerpts = web_search(query, provider=state["search_provider"], top_k=5)
        inp = RESEARCHER_USER.format(
            keywords=json.dumps(state.get("keywords", [])),
            excerpts=json.dumps(excerpts),
            last_hparams=json.dumps(state["supervisor_out"]["hyperparameters"]),
        )
        resp = llm.invoke([("system", RESEARCHER_SYS), ("user", inp)]).content
        data = parse_json_safe(resp, {"actions": [], "notes": ""})
        candidate = clamp_actions_to_ranges(state["supervisor_out"]["hyperparameters"], data.get("actions", []))
        return {"web_hints": {"actions": data.get("actions", []), "notes": data.get("notes",""), "candidate": candidate}}

    def loop_or_end(state: HPOState):
        df = pd.DataFrame(state.get("trials_summary_rows", []))
        save_csv(os.path.join(state["run_dir"], "trials_summary.csv"), df)

        next_trial = state["trial_idx"] + 1
        updates = {
            "trial_idx": next_trial,
            "consult_turn": 0,
            "gen_consult_a": [],  # reset lists
            "gen_consult_b": [],
            "last_hparams": state["supervisor_out"]["hyperparameters"],
        }
        if next_trial >= state["rounds"]:
            save_json(os.path.join(state["run_dir"], "best_overall.json"), state["best_so_far"])
            return END
        else:
            return "consultant_a", updates

    # Build graph
    builder.add_node("init", init_state)
    builder.add_node("consultant_a", consultant_a_node)
    builder.add_node("consultant_b", consultant_b_node)
    builder.add_node("supervisor", supervisor_node)
    builder.add_node("executor", executor_node)
    builder.add_node("analyzer", analyzer_node)
    builder.add_node("researcher", researcher_node)

    builder.set_entry_point("init")
    builder.add_conditional_edges("consultant_a", consult_router, {
        "consultant_a":"consultant_a","consultant_b":"consultant_b","supervisor":"supervisor"
    })
    builder.add_conditional_edges("consultant_b", consult_router, {
        "consultant_a":"consultant_a","consultant_b":"consultant_b","supervisor":"supervisor"
    })
    builder.add_edge("init", "consultant_a")
    builder.add_edge("supervisor", "executor")
    builder.add_edge("executor", "analyzer")
    builder.add_edge("analyzer", "researcher")
    builder.add_conditional_edges("researcher", lambda s: "consultant_a" if s["trial_idx"]+1 < s["rounds"] else END,
                                  {"consultant_a":"consultant_a", END: END})

    compiled = builder.compile()
    return compiled