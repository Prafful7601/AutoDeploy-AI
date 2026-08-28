import "./ModelStats.css";

// Real numbers from outputs/reports/stage2_model_results.md and
// stage1_data_report.md — nothing here is rounded for effect beyond
// standard 2-3 significant figures already used in those reports.
const STATS = [
  { value: "261,139", label: "builds trained on" },
  { value: "243", label: "real projects (2011–2016)" },
  { value: "0.804", label: "PR-AUC, temporal split" },
  { value: "0.690", label: "PR-AUC, unseen projects" },
  { value: "+0.24", label: "PR-AUC over the real baseline*" },
];

export function ModelStats() {
  return (
    <section className="model-stats" aria-label="Model statistics">
      <div className="model-stats__row">
        {STATS.map((s) => (
          <div className="model-stats__item" key={s.label}>
            <div className="model-stats__value mono">{s.value}</div>
            <div className="model-stats__label">{s.label}</div>
          </div>
        ))}
      </div>
      <p className="model-stats__footnote">
        *the baseline is "previous build in this project failed → predict fail," not
        majority-class — that baseline was already at 0.565 PR-AUC (temporal).{" "}
        <a
          href="https://github.com/Prafful7601/AutoDeploy-AI/blob/main/outputs/reports/stage2_model_results.md"
          target="_blank"
          rel="noreferrer"
        >
          Full results ↗
        </a>
      </p>
    </section>
  );
}
