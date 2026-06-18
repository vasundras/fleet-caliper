"""Caliper: cost-bounded autonomy for LLM agent fleets.

Bounds spend along two independent axes:
  * temporal     - step/run/session/fleet ceilings (BudgetPolicy / CostMeter)
  * dimensional  - per-agent / per-task / per-(agent,task) ceilings
                   (AttributionBudget / LabeledMeter), with statistical
                   spike/trend grounding (BaselineTracker).

Public API:
    Caliper                   - facade tying metering, budgets, baselines, detection together
    BudgetPolicy              - declarative temporal soft/hard ceilings per scope
    CostMeter                 - hierarchical temporal token/dollar accumulation
    Scope, Usage              - temporal scope enum; per-scope usage record
    LabeledMeter              - dimensional usage keyed by labels (agent/task/...)
    BudgetRule, AttributionBudget - composable per-dimension ceilings
    BaselineTracker, SpikeVerdict - online per-scope stats; spike/trend grounding
    Alert, AlertKind          - exhaustion / spike / trend alerts
    PriceBook                 - token -> dollar mapping
    LoopDetector              - online loop / thrash / oscillation detection
    CaliperCallbackHandler    - LangChain callback that meters, attributes, grounds, enforces
    BudgetExceeded / LoopDetected / CaliperTripped - breaker exceptions
"""

from .alerts import Alert, AlertKind
from .attribution import AttributionBudget, AttributionBreach, BudgetRule, LabeledMeter
from .baselines import BaselineTracker, ScopeStats, SpikeVerdict
from .budget import BudgetPolicy, CostMeter, Scope, Usage
from .exceptions import BudgetExceeded, CaliperTripped, LoopDetected
from .loop_detection import LoopDetector
from .pricing import PriceBook
from .caliper import Caliper

# CaliperCallbackHandler is the only langchain-dependent surface. Import it
# lazily so the framework-agnostic core (budgets, attribution, baselines, loop
# detection) is usable without langchain installed.
def __getattr__(name: str):  # PEP 562
    if name == "CaliperCallbackHandler":
        from .callbacks import CaliperCallbackHandler
        return CaliperCallbackHandler
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "Caliper",
    "BudgetPolicy",
    "CostMeter",
    "Scope",
    "Usage",
    "LabeledMeter",
    "BudgetRule",
    "AttributionBudget",
    "AttributionBreach",
    "BaselineTracker",
    "ScopeStats",
    "SpikeVerdict",
    "Alert",
    "AlertKind",
    "PriceBook",
    "LoopDetector",
    "CaliperCallbackHandler",
    "BudgetExceeded",
    "LoopDetected",
    "CaliperTripped",
]

__version__ = "0.1.0"
