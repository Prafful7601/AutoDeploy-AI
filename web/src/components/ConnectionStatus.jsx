import { useState } from "react";
import { API_BASE, MANUAL_HEALTH_CHECK_TIMEOUT_MS, checkHealth } from "../lib/api";
import "./ConnectionStatus.css";

/**
 * A visible, user-triggered "is the live backend actually up right now"
 * check — distinct from the silent, fast check App.jsx runs once on load.
 * Exists specifically because Render's free tier sleeps after ~15 minutes
 * idle and can take up to ~50s to wake back up: the passive on-load check
 * gives up in 2.5s (so a portfolio visitor never stares at a blank page),
 * but a visitor who deliberately clicks this button is choosing to wait
 * through a cold start, so this one is patient (60s).
 */
export function ConnectionStatus({ mode, onModeChange }) {
  const [checking, setChecking] = useState(false);
  const [lastResult, setLastResult] = useState(null); // null | "connected" | "unreachable"

  async function handleCheck() {
    setChecking(true);
    setLastResult(null);
    const reachable = await checkHealth(MANUAL_HEALTH_CHECK_TIMEOUT_MS);
    setChecking(false);
    setLastResult(reachable ? "connected" : "unreachable");
    onModeChange(reachable ? "live" : "demo");
  }

  return (
    <div className={`connection-status connection-status--${mode}`}>
      <div className="connection-status__dot" aria-hidden="true" />
      <div className="connection-status__text">
        <strong>{mode === "live" ? "Live — backend connected" : "Demo mode — backend not connected"}</strong>
        <span className="connection-status__url mono">{API_BASE}</span>
      </div>
      <button
        type="button"
        className="connection-status__button"
        onClick={handleCheck}
        disabled={checking}
      >
        {checking ? "Checking… (up to 60s, free-tier can be asleep)" : "Check connection"}
      </button>
      {lastResult === "connected" && !checking && (
        <span className="connection-status__result connection-status__result--ok">
          ✓ Connected — switched to live mode
        </span>
      )}
      {lastResult === "unreachable" && !checking && (
        <span className="connection-status__result connection-status__result--fail">
          Still unreachable — showing demo data
        </span>
      )}
    </div>
  );
}
