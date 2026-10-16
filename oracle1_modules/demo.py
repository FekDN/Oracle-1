"""Runnable historical and intervention demo for the Oracle-1 system."""

from __future__ import annotations

import logging

from .examples import run_real_historical_laser_path
from .interventions import (
    InterventionOptimizer,
    spec_adjacent_breakthrough,
    spec_investment,
    spec_remove_regulation,
    spec_talent_influx,
)
from .system import build_oracle1_system


def run_intervention_demo() -> None:
    """Build the historical laser graph and run intervention examples."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
    )
    system = build_oracle1_system()
    run_real_historical_laser_path(system)
    optimizer = InterventionOptimizer(system)

    investment = spec_investment(
        node_id="theory_1958",
        amount_usd=200_000_000,
        year=1959,
        label="Pentagon $200M coherent-light fund (1959)",
    )
    print(optimizer.simulate(investment, n_simulations=300).summary())

    regulation = spec_remove_regulation(
        node_id="theory_1958",
        scale=0.3,
        label="Reduced vacuum-tube industry restrictions",
    )
    print(optimizer.simulate(regulation, n_simulations=300).summary())

    candidates = [
        investment,
        regulation,
        spec_adjacent_breakthrough(
            "maser_1954", scientific_boost=2.5, label="Population-inversion breakthrough"
        ),
        spec_talent_influx(
            "theory_1958", headcount_multiplier=3.0, label="Bell Labs talent program"
        ),
    ]
    best, gain = optimizer.find_optimal(
        candidates=candidates, target_years=2.0, target_probability=0.70, max_interventions=2
    )
    print(f"\nBest combination (expected acceleration {gain:+.2f} years):")
    for spec in best:
        print(f"  {spec.label}")
