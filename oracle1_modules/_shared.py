# Copyright (c) 2026 Dmitry Feklin (FeklinDN@gmail.com)
"""Shared standard-library, numerical, and optional PyTorch dependencies."""

from __future__ import annotations
  
import bisect
import copy
import hashlib
import heapq
import html as html_lib
import json
import math
import os
import pickle
import re
import statistics
import struct
import time
import uuid
import logging
import urllib.error
import urllib.request
from collections import Counter, defaultdict, deque
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum, auto
from pathlib import Path
from typing import Any, Callable, Deque, Dict, FrozenSet, Iterable, List, Optional, Set, Tuple

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity as cos_sim

logger = logging.getLogger("oracle1")

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch_geometric.nn import GATv2Conv
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch / torch-geometric not installed — stubs active")

    class nn:
        class Module:
            def __init__(self): pass
            def parameters(self): return iter([])
        class Linear:
            def __init__(self, *a, **kw): pass
        class LayerNorm:
            def __init__(self, *a, **kw): pass
        class Dropout:
            def __init__(self, *a, **kw): pass
        class ModuleList(list): pass
        class ModuleDict(dict): pass
        class Sequential:
            def __init__(self, *a): pass
        class GRU:
            def __init__(self, *a, **kw): pass
        GELU = object

    class torch:
        class Tensor: pass
        @staticmethod
        def cat(t, dim=0): return None
        @staticmethod
        def zeros(*a, **kw): return None
        @staticmethod
        def ones(*a, **kw): return None
        @staticmethod
        def sigmoid(x): return x
        @staticmethod
        def stack(t): return None
        @staticmethod
        def arange(*a): return None
        @staticmethod
        def exp(x): return x
        @staticmethod
        def log(x): return x
        @staticmethod
        def tensor(x): return x
        @staticmethod
        def relu(x): return x
        @staticmethod
        def clamp(x, *a): return x

    F = None

    class GATv2Conv:
        def __init__(self, *a, **kw): pass


@contextmanager
def _maybe_no_grad():
    if TORCH_AVAILABLE:
        with torch.no_grad():
            yield
    else:
        yield


__all__ = [name for name in globals() if not name.startswith("__")]
