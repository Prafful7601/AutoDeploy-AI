import "./HowItWorks.css";

const STEPS = [
  {
    title: "A commit happens",
    body: "A push or pull request lands on a repo — before any CI pipeline has run.",
  },
  {
    title: "Gather 31 facts",
    body: "The extractor pulls facts available at that moment: how big the change is, who wrote it and their track record here, how this project's recent builds have gone, and static project context — all from the GitHub API, nothing from the future.",
  },
  {
    title: "Compare to learned patterns",
    body: "A gradient-boosted model, trained on 261k historical CI builds, scores how similar this situation looks to past builds that failed.",
  },
  {
    title: "Explain, not just score",
    body: "SHAP attributes the prediction back to specific facts, translated into plain language — the explanation is the point, not the raw number.",
  },
];

export function HowItWorks() {
  return (
    <section className="how-it-works" aria-labelledby="how-it-works-heading">
      <h2 id="how-it-works-heading">How this works</h2>
      <ol className="how-it-works__steps">
        {STEPS.map((step, i) => (
          <li key={step.title} className="how-it-works__step">
            <span className="how-it-works__number">{i + 1}</span>
            <div>
              <h3>{step.title}</h3>
              <p>{step.body}</p>
            </div>
          </li>
        ))}
      </ol>
    </section>
  );
}
