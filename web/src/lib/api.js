// Talks to the backend. The API base URL is a build-time env var
// (VITE_-prefixed so Vite exposes it to the client bundle) — never a
// secret; the GitHub token stays server-side in api/main.py and is never
// referenced anywhere in this frontend.
export const API_BASE = import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";

// The passive check on page load: short, so demo mode never makes a
// visitor wait on a hung request before seeing a working dashboard.
export const PASSIVE_HEALTH_CHECK_TIMEOUT_MS = 2500;
// A manual "check connection" click is a deliberate user action, so it's
// allowed to actually wait through a Render free-tier cold start (up to
// ~50s) rather than give up in 2.5s like the passive check does.
export const MANUAL_HEALTH_CHECK_TIMEOUT_MS = 60000;

/**
 * Returns true if the API is reachable, false otherwise (network error,
 * non-2xx, or timeout) — never throws.
 */
export async function checkHealth(timeoutMs = PASSIVE_HEALTH_CHECK_TIMEOUT_MS) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const res = await fetch(`${API_BASE}/health`, { signal: controller.signal });
    return res.ok;
  } catch {
    return false;
  } finally {
    clearTimeout(timer);
  }
}

/**
 * Runs a live prediction for a real repo + commit. Returns
 * { ok: true, result } on success, or { ok: false, status, detail } on
 * any failure — the caller renders `detail` through the same
 * "could not score" card path demo mode already uses, never a fabricated
 * or zeroed prediction.
 */
export async function predictLive({ owner, repo, sha, branch, isPr }) {
  try {
    const res = await fetch(`${API_BASE}/predict-live`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ owner, repo, sha, branch: branch || null, is_pr: isPr ?? null }),
    });
    const body = await res.json().catch(() => ({}));
    if (!res.ok) {
      return { ok: false, status: res.status, detail: body.detail || `Request failed (HTTP ${res.status}).` };
    }
    return { ok: true, result: body };
  } catch (err) {
    return { ok: false, status: null, detail: `Could not reach the API: ${err.message}` };
  }
}
