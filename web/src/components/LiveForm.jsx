import { useState } from "react";
import "./LiveForm.css";

/**
 * Repo + commit SHA input for live mode. Deliberately minimal — this is a
 * demonstration input, not a production form (no repo autocomplete, no
 * SHA validation beyond non-empty).
 */
export function LiveForm({ onSubmit, loading }) {
  const [repoInput, setRepoInput] = useState("spf13/cobra");
  const [sha, setSha] = useState("");
  const [error, setError] = useState(null);

  function handleSubmit(e) {
    e.preventDefault();
    const trimmedRepo = repoInput.trim();
    const trimmedSha = sha.trim();
    if (!trimmedRepo.includes("/")) {
      setError('Repo must be in "owner/name" format.');
      return;
    }
    if (!trimmedSha) {
      setError("Commit SHA is required.");
      return;
    }
    setError(null);
    const [owner, repo] = trimmedRepo.split("/");
    onSubmit({ owner, repo, sha: trimmedSha });
  }

  return (
    <form className="live-form" onSubmit={handleSubmit}>
      <div className="live-form__field">
        <label htmlFor="repo-input">Repo (owner/name)</label>
        <input
          id="repo-input"
          type="text"
          value={repoInput}
          onChange={(e) => setRepoInput(e.target.value)}
          placeholder="spf13/cobra"
        />
      </div>
      <div className="live-form__field">
        <label htmlFor="sha-input">Commit SHA</label>
        <input
          id="sha-input"
          type="text"
          value={sha}
          onChange={(e) => setSha(e.target.value)}
          placeholder="adbc8813901bba65827259daa8e22ff94ec1f30e"
        />
      </div>
      <button type="submit" disabled={loading}>
        {loading ? "Checking…" : "Check this commit"}
      </button>
      {error && <p className="live-form__error">{error}</p>}
    </form>
  );
}
