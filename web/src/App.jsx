import { useState } from "react";
import { CrossCiCaveat, DemoModeBanner } from "./components/Banners";
import { HowItWorks } from "./components/HowItWorks";
import { ModelStats } from "./components/ModelStats";
import { ExampleSelector } from "./components/ExampleSelector";
import { PredictionCard } from "./components/PredictionCard";
import demoData from "./data/demoFixtures.json";
import "./App.css";

// Phase 1 (this build): demo mode only, hardcoded. Phase 2 will check
// GET /health on load and switch to live mode + a repo/SHA input form when
// reachable — deliberately not wired yet, per the brief's explicit request
// to review demo mode on real fixtures before adding the live path.
const MODE = "demo";

function App() {
  const examples = demoData.examples;
  const [selectedId, setSelectedId] = useState(examples[0].id);
  const selected = examples.find((e) => e.id === selectedId);

  return (
    <div className="app">
      <header className="app__header">
        <h1>AutoDeploy AI</h1>
        <p className="app__tagline">Predicts whether a CI build will fail — before it runs.</p>
      </header>

      <div className="app__banners">
        <CrossCiCaveat />
        {MODE === "demo" && <DemoModeBanner />}
      </div>

      <HowItWorks />
      <ModelStats />

      <section className="app__demo" aria-labelledby="try-it-heading">
        <h2 id="try-it-heading">Try it</h2>
        <p className="app__demo-intro">
          Four real outputs from the trained model — a healthy repo, an active failure
          streak, a brand-new repo with no history yet, and a case that leans on the
          model's weaker signal. Pick one to see the full explanation.
        </p>
        <ExampleSelector examples={examples} selectedId={selectedId} onSelect={setSelectedId} />
        <div className="app__card-wrap">
          <PredictionCard example={selected} />
        </div>
      </section>

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
