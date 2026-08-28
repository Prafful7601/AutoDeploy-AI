// Talks to the backend. The API base URL is a build-time env var
// (VITE_-prefixed so Vite exposes it to the client bundle) — never a
// secret; the GitHub token stays server-side in api/main.py and is never
// referenced anywhere in this frontend.
const API_BASE = import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";

const HEALTH_CHECK_TIMEOUT_MS = 2500;

/**
 * Returns true if the API is reachable, false otherwise (network error,
 * non-2xx, or timeout) — never throws. A short timeout keeps the
 * demo-mode fallback feeling instant rather than making a portfolio
 * visitor wait on a hung request before seeing a working dashboard.
 */
export async function checkHealth() {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), HEALTH_CHECK_TIMEOUT_MS);
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
