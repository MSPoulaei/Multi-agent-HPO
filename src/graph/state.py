from typing import Any, Dict, List, TypedDict, Annotated
import operator

class HPOState(TypedDict, total=False):
    trial_idx: int
    rounds: int
    consult_turn: int
    consult_limit: int
    last_hparams: Dict[str, Any]
    gen_consult_a: Annotated[List[Dict[str, Any]], operator.add]
    gen_consult_b: Annotated[List[Dict[str, Any]], operator.add]
    supervisor_out: Dict[str, Any]
    train_results: Dict[str, Any]
    metrics_df_path: str
    analysis: Dict[str, Any]
    keywords: Annotated[List[str], operator.add]
    web_hints: Dict[str, Any]
    run_dir: str
    anonymize: bool
    model_ids: Dict[str, str]
    trainer_cfg: Dict[str, Any]
    search_provider: str
    best_so_far: Dict[str, Any]
    trials_summary_rows: Annotated[List[Dict[str, Any]], operator.add]
    consultant_history: Annotated[List[Dict[str, Any]], operator.add]