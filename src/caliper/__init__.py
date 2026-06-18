"""Caliper: cost-bounded autonomy for LLM agent fleets.

Public API:
    Caliper                   - facade tying metering, policy, and detection together
    BudgetPolicy              - declarative soft/hard ceilings per scope
    CostMeter                 - hierarchical token/dollar accumulation
    PriceBook                 - token -> dollar mapping
    LoopDetector              - online loop / thrash / oscillation detection
    CaliperCallbackHandler    - LangChain callback that meters and trips the breaker
    Scope                     - budget scope enum (STEP/RUN/SESSION/FLEET)
    BudgetExceeded            - raised when a hard budget ceiling is crossed
    LoopDetected              - raised when a pathological loop is detected
    CaliperTripped            - common base for the two breaker exceptions
"""

from .budget import BudgetPolicy, CostMeter, Scope, Usage
from .callbacks import CaliperCallbackHandler
from .exceptions import BudgetExceeded, CaliperTripped, LoopDetected
from .loop_detection import LoopDetector
from .pricing import PriceBook
from .caliper import Caliper

__all__ = [
    "Caliper",
    "BudgetPolicy",
    "CostMeter",
    "Scope",
    "Usage",
    "PriceBook",
    "LoopDetector",
    "CaliperCallbackHandler",
    "BudgetExceeded",
    "LoopDetected",
    "CaliperTripped",
]

__version__ = "0.0.1"
