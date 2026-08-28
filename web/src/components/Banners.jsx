import "./Banners.css";

/**
 * The cross-CI caveat. Per the brief: "a persistent, visible note... Not
 * buried in a tooltip." Rendered once, near the top of the page, always
 * visible regardless of mode or which example is selected — never inside
 * a collapsed/hover-only element.
 */
export function CrossCiCaveat() {
  return (
    <div className="banner banner--caveat" role="note">
      <span className="banner__icon" aria-hidden="true">
        ⚠
      </span>
      <p>
        <strong>Experimental, cross-CI-system signal — not validated.</strong> This
        model was trained on Travis CI build outcomes and is applied here to GitHub
        Actions builds — a different CI system, with known miscalibration on real
        repos. Treat every prediction on this page as a demonstration of the
        pipeline, not a verdict on any real build.{" "}
        <a
          href="https://github.com/Prafful7601/AutoDeploy-AI/blob/main/outputs/reports/stage3_feature_parity.md"
          target="_blank"
          rel="noreferrer"
        >
          Full feature-parity report ↗
        </a>
      </p>
    </div>
  );
}

/** Shown only in demo mode — distinguishes bundled fixtures from a live call. */
export function DemoModeBanner() {
  return (
    <div className="banner banner--demo" role="status">
      <span className="banner__icon" aria-hidden="true">
        ●
      </span>
      <p>
        Showing sample data — live API not connected. These are real captured outputs
        from the trained model (see each card's provenance note), not a live prediction.
      </p>
    </div>
  );
}

/** Shown only in live mode, once the API is confirmed reachable. */
export function LiveModeBanner() {
  return (
    <div className="banner banner--live" role="status">
      <span className="banner__icon" aria-hidden="true">
        ●
      </span>
      <p>Live mode — connected to the prediction API. Predictions below are real-time.</p>
    </div>
  );
}
