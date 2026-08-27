"""
The cold-start rule: mechanism only.

This module is deliberately dependency-free (standard library, no pydantic,
no numpy, no sklearn). Three reasons:

1. The rule is the safety-critical part of this API — it decides whether a
   prediction is presented as actionable or withheld. It should be testable
   and importable without loading a 400 KB model artifact or a web
   framework.
2. Stage 3 Layer 3's GitHub Action needs to answer "is this commit a
   cold-start case?" and should not have to import the serving stack to do
   it.
3. Layer 2's extractor needs the same predicate to report, per feature,
   whether its absence triggers cold-start.

The *canonical documentation* of the rule — what counts as cold-start and
why — lives in `schema.py`, which is this API's stated single source of
truth for the input/output contract. This module holds the executable form
of what that docstring describes. If the two ever disagree, schema.py is
the specification and this file is the bug.

Null handling: a value counts as absent if the key is missing, the value is
None, or the value is a float NaN. The NaN case matters because by the time
a payload reaches the model it has already been through
`PredictionService._row_from_features`, which converts None to np.nan; the
`v != v` identity check catches NaN without importing numpy.
"""

from typing import Any, List, Mapping, Optional, Tuple

# --- The two history groups -------------------------------------------------
#
# These are two INDEPENDENT groups, and conflating them is the mistake this
# module exists to prevent. See schema.py for the full argument.

# Null if and only if this repo has no build before the one being scored.
PROJECT_HISTORY_FEATURES: Tuple[str, ...] = (
    "previous_build_status",
    "project_prior_failure_rate",
)

# Null if and only if this author has no prior build *in this repo*. Happens
# every time a new contributor opens their first PR against a repo that is
# otherwise mature and well-understood by the model.
AUTHOR_HISTORY_FEATURES: Tuple[str, ...] = (
    "author_prior_builds_in_project",
    "author_prior_failure_rate_in_project",
    "author_days_since_last_build_in_project",
)

# The 5 features the API permits to arrive null. Unchanged from Layer 1 —
# widening or narrowing this set is a contract change.
NAN_ALLOWED_FEATURES = frozenset(PROJECT_HISTORY_FEATURES + AUTHOR_HISTORY_FEATURES)

# Of those 5, only these trigger cold-start. This is the whole rule.
COLD_START_TRIGGER_FEATURES = frozenset(PROJECT_HISTORY_FEATURES)

# --- Response vocabulary ----------------------------------------------------

STATUS_OK = "ok"
STATUS_COLD_START = "cold_start"

CONFIDENCE_NORMAL = "normal"
CONFIDENCE_LOW = "low"

CONTRIBUTORS_SCOPE_ALL = "all_features"
CONTRIBUTORS_SCOPE_NON_NULL = "non_null_features_only"

COLD_START_MESSAGE = (
    "Insufficient build history for a reliable prediction; history-based "
    "scoring activates once this repo accrues builds. This repo has no build "
    "recorded before this one, and roughly two thirds of what this model "
    "keys on is prior-build history. The probability below is reported for "
    "transparency only and should not be treated as a risk tier: on a "
    "brand-new repo this model reads 'no history' as a failure signal in "
    "its own right, so it over-flags new repos as high risk. The drivers "
    "listed are the features that do have values."
)

# Kept next to the message so the two cannot drift apart in the docs.
COLD_START_UPGRADE_PATH = (
    "Planned: route cold-start requests to a dedicated history-free model "
    "trained only on transferable change/process features. Expected "
    "performance is approximately the Stage 2 held-out-projects result "
    "(~0.69 PR-AUC), versus 0.804 for the history-using model on the "
    "temporal split. It would sit behind this same cold_start check."
)


def _is_absent(value: Any) -> bool:
    """True if the value is missing, None, or NaN."""
    if value is None:
        return True
    # NaN is the only value that is not equal to itself. Guarded by the
    # isinstance check so we never call __ne__ on something exotic.
    return isinstance(value, float) and value != value


def null_history_features(values: Mapping[str, Any]) -> List[str]:
    """Which of the 5 nullable history features are absent, in declared order.

    Reported on every response, cold-start or not, so a caller can see that
    (say) author history was missing even when the prediction was served
    normally.
    """
    ordered = PROJECT_HISTORY_FEATURES + AUTHOR_HISTORY_FEATURES
    return [f for f in ordered if _is_absent(values.get(f))]


def cold_start_trigger_features(values: Mapping[str, Any]) -> List[str]:
    """Which project-history features are absent, i.e. what fired the rule."""
    return [f for f in PROJECT_HISTORY_FEATURES if _is_absent(values.get(f))]


def is_cold_start(values: Mapping[str, Any]) -> bool:
    """THE RULE: cold-start iff any PROJECT-history feature is absent.

    Author-history absence alone is NOT cold-start. See schema.py.
    """
    return bool(cold_start_trigger_features(values))


def cold_start_reason(values: Mapping[str, Any]) -> Optional[str]:
    """A short machine-stable reason code, or None if not cold-start.

    Three codes rather than one, because the middle case is a malformed
    payload and a caller should be able to tell it apart from a normal
    first build without parsing prose.
    """
    triggered = cold_start_trigger_features(values)
    if not triggered:
        return None
    if len(triggered) == len(PROJECT_HISTORY_FEATURES):
        # The expected, well-formed cold-start case: a repo's first build.
        return "no_prior_build_in_repo"
    # Exactly one of the two null. Not reachable from a correctly built
    # feature vector — both derive from the same "is there a previous build"
    # question — so it means the caller's extractor is inconsistent. We fail
    # toward cold_start rather than serving a tier off half-known history.
    return "partial_project_history_inconsistent"
