import "./LiveModeGuide.css";

// A real, previously-verified example — same commit used throughout this
// project's own development and testing (Stage 3/4 READMEs). Clicking it
// doesn't fabricate anything; it just pre-fills the form with known-good
// real inputs so a first-time visitor has something to click besides a
// blank form.
const EXAMPLE = { repo: "spf13/cobra", sha: "adbc8813901bba65827259daa8e22ff94ec1f30e", note: "real result: Low risk, 19%" };

const STEPS = [
  { icon: "🔍", title: "Pick a real public repo", body: "Any GitHub repo works — one of yours, or a well-known open-source one." },
  { icon: "📋", title: "Grab a commit SHA", body: "Open any commit on GitHub and copy the long hex string from the URL or the commit page." },
  { icon: "▶️", title: "Drop them in and hit “Check this commit”", body: "That's it — no account, no setup." },
  { icon: "⏳", title: "Watch it actually work", body: "Real GitHub API calls, real feature extraction, real model inference — a few seconds, not instant, on purpose." },
  { icon: "💡", title: "Read the drivers, not just the tier", body: "The tier is the headline. The bars underneath are the actual point." },
];

export function LiveModeGuide({ onTryExample }) {
  return (
    <section className="live-guide" aria-labelledby="live-guide-heading">
      <h3 id="live-guide-heading">How to try it with real data</h3>
      <ol className="live-guide__steps">
        {STEPS.map((s) => (
          <li key={s.title} className="live-guide__step">
            <span className="live-guide__icon" aria-hidden="true">{s.icon}</span>
            <div>
              <strong>{s.title}</strong>
              <p>{s.body}</p>
            </div>
          </li>
        ))}
      </ol>
      <div className="live-guide__cta">
        <span>Don't have one handy?</span>
        <button type="button" onClick={() => onTryExample(EXAMPLE)}>
          Try {EXAMPLE.repo} ({EXAMPLE.note})
        </button>
      </div>
    </section>
  );
}
