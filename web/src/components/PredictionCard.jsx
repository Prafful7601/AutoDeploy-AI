import { ShapBar } from "./ShapBar";
import "./PredictionCard.css";

const TIER_COPY = {
  Low: { word: "Low risk", className: "tier-low" },
  Medium: { word: "Medium risk", className: "tier-medium" },
  High: { word: "High risk", className: "tier-high" },
};

/**
 * Renders one prediction result. Three distinct visual treatments,
 * deliberately not variations on a theme:
 *   - "ok"              -> a risk-tier card (colored by tier)
 *   - "cold_start"       -> a neutral, distinctly different "insufficient
 *                          history" card — never uses tier colors
 *   - "could_not_score"  -> a plain, muted "we don't know" card
 *
 * The headline is always the tier word (or the cold-start/error state),
 * never a bare percentage — the probability, when shown, lives inside
 * .prediction-card__probability, which is deliberately small/secondary.
 */
export function PredictionCard({ example }) {
  const { label, input_summary: inputSummary, _provenance: provenance, result } = example;

  return (
    <article className={`prediction-card prediction-card--${stateClass(result)}`}>
      <header className="prediction-card__header">
        <div>
          <h3>{label}</h3>
          <p className="prediction-card__input-summary">
            {inputSummary.repo_description}
            {inputSummary.language && (
              <>
                {" "}
                · <span className="mono">{inputSummary.language}</span>
              </>
            )}
          </p>
        </div>
        {result.status === "ok" && <TierBadge tier={result.risk_tier} />}
        {result.status === "cold_start" && <ColdStartBadge />}
      </header>

      {result.status === "ok" && <OkBody result={result} />}
      {result.status === "cold_start" && <ColdStartBody result={result} />}
      {result.status !== "ok" && result.status !== "cold_start" && <CouldNotScoreBody result={result} />}

      <footer className="prediction-card__provenance">{provenance}</footer>
    </article>
  );
}

function stateClass(result) {
  if (result.status === "cold_start") return "coldstart";
  if (result.status !== "ok") return "error";
  return result.risk_tier.toLowerCase();
}

function TierBadge({ tier }) {
  const copy = TIER_COPY[tier] ?? { word: tier, className: "" };
  return <span className={`tier-badge ${copy.className}`}>{copy.word}</span>;
}

function ColdStartBadge() {
  return <span className="tier-badge tier-badge--coldstart">Insufficient history</span>;
}

function OkBody({ result }) {
  return (
    <div className="prediction-card__body">
      <h4 className="prediction-card__section-heading">What drove this</h4>
      <ShapBar contributors={result.top_contributors} />
      <p className="prediction-card__probability">
        Reference only, not a calibrated risk assessment — experimental, low-confidence
        signal: raw model output <span className="mono">{(result.failure_probability * 100).toFixed(0)}%</span>.
      </p>
    </div>
  );
}

function ColdStartBody({ result }) {
  return (
    <div className="prediction-card__body">
      <p className="prediction-card__coldstart-message">
        <strong>Insufficient build history for a reliable signal.</strong> This repository
        has no qualifying prior build recorded yet — about two-thirds of what this model
        relies on is prior-build history, so no risk tier is shown here. Guessing at one
        would be worse than saying nothing.
      </p>
      {result.top_contributors?.length > 0 && (
        <>
          <h4 className="prediction-card__section-heading">
            Signal available from this commit's change characteristics
          </h4>
          <ShapBar contributors={result.top_contributors} />
        </>
      )}
      <p className="prediction-card__probability prediction-card__probability--coldstart">
        Raw model output <span className="mono">{(result.failure_probability * 100).toFixed(0)}%</span> is shown
        for transparency only — it is not a risk score and should not be treated as one.
      </p>
    </div>
  );
}

const ERROR_HEADLINES = {
  not_found: "Repo or commit not found.",
  rate_limited: "Could not fetch enough history.",
};

function CouldNotScoreBody({ result }) {
  const headline = ERROR_HEADLINES[result.status] || "Could not generate a signal.";
  const showNotACodeIssueNote = result.status !== "not_found"; // a typo'd repo/SHA isn't "the code's" fault either way, but the phrasing specifically reassures about infra hiccups — doesn't fit a user-input error
  return (
    <div className="prediction-card__body">
      <p className="prediction-card__coldstart-message">
        <strong>{headline}</strong> {result.detail || "Required data could not be retrieved this run."}
        {showNotACodeIssueNote && " This is not a signal that anything is wrong with the code."}
      </p>
    </div>
  );
}
