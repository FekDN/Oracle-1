"""Run a non-persistent intervention simulation on the symbolic Oracle graph."""

from __future__ import annotations

import argparse
import base64
import json
import logging
from pathlib import Path
from typing import Any, Dict

from graph_update import build_symbolic_system
from oracle1 import InterventionOptimizer, InterventionSpec, InterventionType, load_graph_snapshot


logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")
logger = logging.getLogger("oracle1.simulate")


def simulate(args: argparse.Namespace) -> Dict[str, Any]:
    if not Path(args.snapshot).is_file():
        raise FileNotFoundError(f"Graph snapshot not found: {args.snapshot}")
    if not 1 <= args.n_simulations <= 1000:
        raise ValueError("n_simulations must be between 1 and 1000")
    if not 0.0 <= args.noise_sigma <= 2.0:
        raise ValueError("noise_sigma must be between 0 and 2")

    params_text = base64.urlsafe_b64decode(args.params_b64).decode("utf-8")
    params = json.loads(params_text)
    if not isinstance(params, dict):
        raise ValueError("Simulation parameters must decode to a JSON object")
    intervention_type = InterventionType[args.intervention_type]

    system = build_symbolic_system(llm_client=None)
    load_graph_snapshot(args.snapshot, system)
    spec = InterventionSpec(
        target_node_id=args.target_node_id,
        intervention_type=intervention_type,
        params=params,
        label=args.label,
    )
    result = InterventionOptimizer(system).simulate(
        spec,
        n_simulations=args.n_simulations,
        noise_sigma=args.noise_sigma,
        random_seed=args.seed,
    )
    return {
        "kind": "counterfactual_simulation",
        "persistent_graph_changed": False,
        "interpretation": "Temporary intervention estimate; the saved graph was not changed.",
        "target_node_id": spec.target_node_id,
        "intervention_type": intervention_type.name,
        "label": spec.label,
        "parameters": params,
        "n_simulations": result.n_simulations,
        "elapsed_s": result.elapsed_s,
        "mean_delta_readiness": result.mean_delta_readiness,
        "std_delta_readiness": result.std_delta_readiness,
        "mean_delta_forecast": result.mean_delta_forecast,
        "std_delta_forecast": result.std_delta_forecast,
        "mean_years_accelerated": result.mean_years_accelerated,
        "p10_years_accelerated": result.p10_years_accelerated,
        "p50_years_accelerated": result.p50_years_accelerated,
        "p90_years_accelerated": result.p90_years_accelerated,
        "prob_acceleration": result.prob_acceleration,
        "cascade_effects": [vars(effect) for effect in result.top_cascade_effects],
        "bottleneck_changes": [vars(change) for change in result.bottleneck_changes],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Symbolic Oracle-1 what-if simulation")
    parser.add_argument("--snapshot", required=True)
    parser.add_argument("--target_node_id", required=True)
    parser.add_argument("--intervention_type", required=True,
                        choices=[kind.name for kind in InterventionType])
    parser.add_argument("--params_b64", required=True)
    parser.add_argument("--label", default="")
    parser.add_argument("--n_simulations", type=int, default=100)
    parser.add_argument("--noise_sigma", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_json", default=None)
    args = parser.parse_args()

    try:
        output = simulate(args)
    except (KeyError, ValueError, OSError, json.JSONDecodeError) as exc:
        logger.error("Simulation failed: %s", exc)
        raise SystemExit(2) from exc

    rendered = json.dumps(output, ensure_ascii=False, indent=2)
    print(rendered)
    if args.output_json:
        Path(args.output_json).write_text(rendered, encoding="utf-8")


if __name__ == "__main__":
    main()
