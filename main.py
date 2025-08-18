import argparse
import os
import time
from dotenv import load_dotenv

from src.graph.hpo_graph import build_hpo_graph
from src.utils.seed import fix_seed

def parse_args():
    p = argparse.ArgumentParser(description="Multi-agent HPO with LangGraph + Gemini")
    p.add_argument("--rounds", type=int, default=10)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--output-dir", type=str, default="runs")
    p.add_argument("--consult-turns", type=int, default=2)
    p.add_argument("--amp", action="store_true", default=True)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--scheduler", type=str, default="cosine", choices=["none", "cosine"])
    p.add_argument("--augment", type=str, default="basic", choices=["basic", "strong"])
    p.add_argument("--save-checkpoints", action="store_true", default=True)
    p.add_argument("--anonymize", action="store_true", default=True)
    # Model names (OpenAI-compatible Gemini IDs)
    p.add_argument("--models.gen_a", dest="model_gen_a", type=str, default="gemini-2.5-flash")
    p.add_argument("--models.gen_b", dest="model_gen_b", type=str, default="gemini-2.5-flash")
    p.add_argument("--models.supervisor", dest="model_supervisor", type=str, default="gemini-2.5-pro")
    p.add_argument("--models.exec", dest="model_exec", type=str, default="gemini-2.5-flash")
    p.add_argument("--models.researcher", dest="model_researcher", type=str, default="gemini-2.5-flash")
    p.add_argument("--search-provider", type=str, default="gemini", choices=["gemini", "cse"])
    return p.parse_args()

def main():
    load_dotenv()
    args = parse_args()
    fix_seed(args.seed)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(args.output_dir, timestamp)
    os.makedirs(run_dir, exist_ok=True)

    graph = build_hpo_graph(
        run_dir=run_dir,
        epochs=args.epochs,
        patience=args.patience,
        consult_turns=args.consult_turns,
        scheduler=args.scheduler,
        augment=args.augment,
        num_workers=args.num_workers,
        amp=args.amp,
        save_checkpoints=args.save_checkpoints,
        anonymize=args.anonymize,
        rounds=args.rounds,
        model_ids={
            "gen_a": args.model_gen_a,
            "gen_b": args.model_gen_b,
            "supervisor": args.model_supervisor,
            "exec": args.model_exec,
            "researcher": args.model_researcher,
        },
        search_provider=args.search_provider,
    )

    # Run the LangGraph
    final_state = graph.invoke({})
    print(f"Done. Results saved to: {run_dir}")

if __name__ == "__main__":
    main()