# Copyright (c) 2026 Dmitry Feklin (FeklinDN@gmail.com)
"""Compatibility facade for the modular Oracle-1 implementation.

Existing scripts can continue importing from ``oracle1`` while implementation
code lives in focused modules under ``oracle1_modules``.
"""
from __future__ import annotations

from oracle1_modules._shared import *
from oracle1_modules.config import *
from oracle1_modules.llm import *
from oracle1_modules.entities import *
from oracle1_modules.neural import *
from oracle1_modules.time_utils import *
from oracle1_modules.graph import *
from oracle1_modules.graph_dynamics import *
from oracle1_modules.forecasting import *
from oracle1_modules.topology_dynamics import *
from oracle1_modules.validation import *
from oracle1_modules.label_status import *
from oracle1_modules.ingestion import *
from oracle1_modules.consolidation import *
from oracle1_modules.training import *
from oracle1_modules.orchestrators import *
from oracle1_modules.significance import *
from oracle1_modules.json_utils import *
from oracle1_modules.canonical_keys import *
from oracle1_modules.canonical import *
from oracle1_modules.tensorization import *
from oracle1_modules.system import *
from oracle1_modules.snapshots import *
from oracle1_modules.examples import *
from oracle1_modules.interventions import *
from oracle1_modules.visualizer import *

if __name__ == "__main__":
    from oracle1_modules.demo import run_intervention_demo

    run_intervention_demo()
