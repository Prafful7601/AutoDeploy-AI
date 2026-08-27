"""
Heuristics for classifying a changed file path as test / doc / source /
other. TravisTorrent's own numbers came from a research tool with
per-language parsers we don't have access to and can't exactly replicate
live from the GitHub API — these are documented approximations, not a
claim of matching TravisTorrent's methodology exactly. See
outputs/reports/stage3_feature_parity.md for the honest assessment of how
much this matters per feature.
"""

import re

_TEST_PATH_RE = re.compile(
    r"(^|/)(tests?|specs?|__tests__)(/|$)|(^|/)(test_|spec_)[^/]*$|_(test|spec)\.[^/.]+$|\.(test|spec)\.[^/.]+$",
    re.IGNORECASE,
)
_DOC_PATH_RE = re.compile(
    r"^(docs?|documentation)(/|$)|\.(md|rst|adoc|txt)$|^(readme|changelog|license|contributing|authors)(\.[^/.]+)?$",
    re.IGNORECASE,
)


def classify_file(path: str) -> str:
    """Returns 'test', 'doc', or 'src'. There's no live 'other' bucket in
    this heuristic (TravisTorrent's gh_diff_other_files covers things like
    build config, generated files, binary assets) — anything not test or
    doc is classified 'src' here, which will overcount src_files_changed
    relative to training for repos with a lot of config/CI-yaml churn.
    Flagged in the parity report."""
    if _TEST_PATH_RE.search(path):
        return "test"
    if _DOC_PATH_RE.search(path):
        return "doc"
    return "src"
