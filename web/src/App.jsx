import { useEffect, useState } from "react";
import { CrossCiCaveat, DemoModeBanner, LiveModeBanner } from "./components/Banners";
import { ConnectionStatus } from "./components/ConnectionStatus";
import { HowItWorks } from "./components/HowItWorks";
import { ModelStats } from "./components/ModelStats";
import { ExampleSelector } from "./components/ExampleSelector";
import { PredictionCard } from "./components/PredictionCard";
import { LiveForm } from "./components/LiveForm";
import { LiveModeGuide } from "./components/LiveModeGuide";
import { checkHealth, predictLive } from "./lib/api";
import demoData from "./data/demoFixtures.json";
import "./App.css";

// Starts as "demo" (not "checking") on purpose: a portfolio visitor with
// no API running sees a fully working dashboard immediately, with zero
// flash of blank/loading content. If /health turns out to be reachable,
// the effect below upgrades to "live" — the fallback is the default, live
// is the upgrade, never the other way around. Returning the setter too
// lets ConnectionStatus's manual "Check connection" button drive the same
// state directly, rather than duplicating it.
function useApiMode() {
  const [mode, setMode] = useState("demo");
  useEffect(() => {
    let cancelled = false;
    checkHealth().then((reachable) => {
      if (!cancelled && reachable) setMode("live");
    });
    return () => {
      cancelled = true;
    };
  }, []);
  return [mode, setMode];
}

/** Adapts a /predict-live response (or an error) into the same shape
 * PredictionCard already renders for demo fixtures — one component, two
 * data sources, per the brief. */
function toLiveExample({ owner, repo, sha }, outcome) {
  const repoLabel = `${owner}/${repo}`;
  const shaLabel = sha.slice(0, 12);
  if (!outcome.ok) {
    return {
      id: "live-result",
      label: `${repoLabel} @ ${shaLabel}`,
      _provenance: "Live prediction attempt — real-time extraction against the entered repo/commit.",
      input_summary: { repo_description: repoLabel, language: null },
      result: { status: statusLabelFor(outcome.status), detail: outcome.detail },
    };
  }
  const { result } = outcome;
  return {
    id: "live-result",
    label: `${repoLabel} @ ${shaLabel}`,
    _provenance: "Live prediction — real-time extraction (Stage 3 Layer 2) + inference against the entered repo/commit.",
    input_summary: { repo_description: repoLabel, language: result.features?.language },
    result,
  };
}

function statusLabelFor(httpStatus) {
  if (httpStatus === 404) return "not_found";
  if (httpStatus === 429) return "rate_limited";
  return "could_not_score";
}

function App() {
  const [mode, setMode] = useApiMode();
  const examples = demoData.examples;
  const [selectedId, setSelectedId] = useState(examples[0].id);
  const selectedDemo = examples.find((e) => e.id === selectedId);

  const [repoInput, setRepoInput] = useState("spf13/cobra");
  const [shaInput, setShaInput] = useState("");
  const [formError, setFormError] = useState(null);
  const [liveExample, setLiveExample] = useState(null);
  const [liveLoading, setLiveLoading] = useState(false);

  async function handleLiveSubmit(input) {
    setLiveLoading(true);
    setLiveExample(null);
    const outcome = await predictLive(input);
    setLiveExample(toLiveExample(input, outcome));
    setLiveLoading(false);
  }

  function handleTryExample({ repo, sha }) {
    setRepoInput(repo);
    setShaInput(sha);
    setFormError(null);
    handleLiveSubmit({ owner: repo.split("/")[0], repo: repo.split("/")[1], sha });
  }

  return (
    <div className="app">
      <header className="app__header">
        <h1>AutoDeploy AI</h1>
        <p className="app__tagline">Predicts whether a CI build will fail — before it runs.</p>
      </header>

      <div className="app__banners">
        <CrossCiCaveat />
        {mode === "demo" ? <DemoModeBanner /> : <LiveModeBanner />}
        <ConnectionStatus mode={mode} onModeChange={setMode} />
      </div>

      <HowItWorks />
      <ModelStats />

      {mode === "live" ? (
        <section className="app__demo" aria-labelledby="try-it-heading">
          <h2 id="try-it-heading">Try it — live</h2>
          <p className="app__demo-intro">
            Enter a real public repo and commit SHA. The backend runs the same extractor and
            model this whole project is built on — nothing here is a second, simplified code
            path.
          </p>
          <LiveModeGuide onTryExample={handleTryExample} />
          <LiveForm
            repoInput={repoInput}
            sha={shaInput}
            onRepoChange={setRepoInput}
            onShaChange={setShaInput}
            onSubmit={handleLiveSubmit}
            loading={liveLoading}
            error={formError}
            onErrorChange={setFormError}
          />
          <div className="app__card-wrap">
            {liveLoading && <p className="app__loading">Extracting features and running the model…</p>}
            {!liveLoading && liveExample && <PredictionCard example={liveExample} />}
          </div>
        </section>
      ) : (
        <section className="app__demo" aria-labelledby="try-it-heading">
          <h2 id="try-it-heading">Try it</h2>
          <p className="app__demo-intro">
            Four real outputs from the trained model — a healthy repo, an active failure
            streak, a brand-new repo with no history yet, and a case that leans on the
            model's weaker signal. Pick one to see the full explanation.
          </p>
          <ExampleSelector examples={examples} selectedId={selectedId} onSelect={setSelectedId} />
          <div className="app__card-wrap">
            <PredictionCard example={selectedDemo} />
          </div>
        </section>
      )}

      <footer className="app__footer">
        <p>
          Stage 4 of a scoped demo project — see the{" "}
          <a href="https://github.com/Prafful7601/AutoDeploy-AI" target="_blank" rel="noreferrer">
            full README and reports
          </a>{" "}
          for methodology, honest performance numbers, and every judgment call made along
          the way.
        </p>
      </footer>
    </div>
  );
}

export default App;
