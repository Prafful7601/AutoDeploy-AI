import "./LiveForm.css";

/**
 * Repo + commit SHA input for live mode. Deliberately minimal — this is a
 * demonstration input, not a production form (no repo autocomplete, no
 * SHA validation beyond non-empty).
 *
 * Controlled by the parent (repoInput/sha/onRepoChange/onShaChange) rather
 * than holding its own state, so the "try an example" quick-fill buttons
 * in LiveModeGuide can populate these fields directly — one source of
 * truth for what's in the form, not two components racing to own it.
 */
export function LiveForm({ repoInput, sha, onRepoChange, onShaChange, onSubmit, loading, error, onErrorChange }) {
  function handleSubmit(e) {
    e.preventDefault();
    const trimmedRepo = repoInput.trim();
    const trimmedSha = sha.trim();
    if (!trimmedRepo.includes("/")) {
      onErrorChange('Repo must be in "owner/name" format.');
      return;
    }
    if (!trimmedSha) {
      onErrorChange("Commit SHA is required.");
      return;
    }
    onErrorChange(null);
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
          onChange={(e) => onRepoChange(e.target.value)}
          placeholder="spf13/cobra"
        />
      </div>
      <div className="live-form__field">
        <label htmlFor="sha-input">Commit SHA</label>
        <input
          id="sha-input"
          type="text"
          value={sha}
          onChange={(e) => onShaChange(e.target.value)}
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
