"""
LoCoMo 评测入口。

用法:
  cd /home/wbw/Code/agent_memory

  # 用 SimpleDemoMemory (baseline, 无语义检索)
  python -m AgentSocket.experiments.run_locomo --smoke

  # 用 A-mem backend (语义检索)
  python -m AgentSocket.experiments.run_locomo --smoke --backend amem
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Ensure project root is in sys.path so AgentSocket package resolves
_ROOT = Path(__file__).parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# Import AgentSocket first — its __init__ adds AgentSocket/ dir to sys.path
import AgentSocket  # noqa: F401

from agent.clients import from_env
from AgentSocket.memory_backends.simple_demo import SimpleDemoMemory
from AgentSocket.datasets.locomo import load
from AgentSocket.evaluation.locomo_harness import LoCoMoConfig, LoCoMoHarness


def _make_memory_factory(backend: str):
    if backend == "simple":
        return SimpleDemoMemory
    if backend == "amem":
        from AgentSocket.memory_backends.amem_backend import AMEMBackend
        api_key = os.environ.get("OPENAI_API_KEY")
        api_base = os.environ.get("OPENAI_BASE_URL")
        model = os.environ.get("AGENT_MODEL", "gpt-4o-mini")
        def factory():
            return AMEMBackend(
                llm_backend="openai",
                llm_model=model,
                api_key=api_key,
                api_base=api_base,
                fast_ingest=True,
            )
        return factory
    raise ValueError(f"Unknown backend: {backend!r}. Choose 'simple' or 'amem'.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run LoCoMo evaluation")
    parser.add_argument("--smoke", action="store_true", help="Run 1 sample, first 2 QAs only")
    parser.add_argument("--samples", type=int, default=None, help="Number of samples to evaluate")
    parser.add_argument("--output", type=str, default=None, help="Save results to JSON file")
    parser.add_argument(
        "--backend", choices=["simple", "amem"], default="simple",
        help="Memory backend: 'simple' (baseline) or 'amem' (A-mem semantic retrieval)",
    )
    args = parser.parse_args()

    # Load .env so OPENAI_API_KEY etc. are available
    from agent.clients import _load_dotenv_if_present
    _load_dotenv_if_present(_ROOT / ".env")

    print("Loading dataset...")
    all_samples = load()
    print(f"Loaded {len(all_samples)} samples")

    if args.smoke:
        samples = all_samples[:1]
        max_qa = 2
        print(f"Smoke mode: 1 sample, up to {max_qa} QAs")
    elif args.samples:
        samples = all_samples[: args.samples]
        max_qa = None
    else:
        samples = all_samples
        max_qa = None

    model_client = from_env()
    print(f"Model client: {type(model_client).__name__}, model={model_client.model}")
    print(f"Memory backend: {args.backend}")

    memory_factory = _make_memory_factory(args.backend)

    config = LoCoMoConfig(
        name=f"locomo-{args.backend}",
        memory_factory=memory_factory,
        model_client=model_client,
        samples=samples,
        max_qa_per_sample=max_qa,
    )

    harness = LoCoMoHarness()
    print("Running evaluation...")
    results = harness.run(config)

    print("\n=== Results ===")
    for r in results:
        print(f"\nQ: {r.question}")
        print(f"Pred: {r.prediction}")
        print(f"GT:   {r.ground_truth}")
        print(f"F1={r.scores['token_f1']:.3f}  EM={r.scores['exact_match']:.3f}  cat={r.category}")

    agg = harness.aggregate(results)
    print("\n=== Aggregate ===")
    print(json.dumps(agg, indent=2))

    if args.output:
        out = [
            {
                "sample_id": r.sample_id,
                "question": r.question,
                "prediction": r.prediction,
                "ground_truth": r.ground_truth,
                "category": r.category,
                "scores": r.scores,
            }
            for r in results
        ]
        Path(args.output).write_text(json.dumps(out, indent=2, ensure_ascii=False))
        print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
