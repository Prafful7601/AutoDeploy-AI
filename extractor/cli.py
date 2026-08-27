"""
Command-line entry point: extract a commit's feature vector and print it as
JSON. Used for manual checks here, and by Layer 3's GitHub Action to turn
a push/PR event into a payload for POST /predict.

    python -m extractor.cli OWNER/REPO SHA [--branch BRANCH] [--is-pr]

Requires GITHUB_TOKEN in the environment (or a .env file) for a reasonable
rate limit; works unauthenticated at 60 requests/hour for a quick check.
"""

import argparse
import json
import sys

from .extract import build_feature_vector
from .github_client import GitHubClient, GitHubRateLimitError


def main():
    parser = argparse.ArgumentParser(description="Extract AutoDeploy AI features for one commit.")
    parser.add_argument("repo", help="owner/name, e.g. octocat/Hello-World")
    parser.add_argument("sha", help="commit SHA")
    parser.add_argument("--branch", default=None, help="branch name, if known (improves is_main_branch accuracy)")
    parser.add_argument("--is-pr", action="store_true", default=None, help="set if this commit is a PR build")
    parser.add_argument("--raw", action="store_true", help="also print the raw provenance + fetched API data")
    args = parser.parse_args()

    owner, repo = args.repo.split("/", 1)
    client = GitHubClient()
    if not client.authenticated:
        print("WARNING: no GITHUB_TOKEN found — running unauthenticated (60 requests/hour). "
              "Set GITHUB_TOKEN in your environment or .env for real use.", file=sys.stderr)

    try:
        result = build_feature_vector(owner, repo, args.sha, branch=args.branch, is_pr=args.is_pr, client=client)
    except GitHubRateLimitError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)

    output = {"features": result.features}
    if args.raw:
        output["provenance"] = result.provenance
    print(json.dumps(output, indent=2, default=str))


if __name__ == "__main__":
    main()
