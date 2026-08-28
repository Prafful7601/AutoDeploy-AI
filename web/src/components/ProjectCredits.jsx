import "./ProjectCredits.css";

const TEAM = [
  { name: "Mohd Aatir", roll: "202510116100142" },
  { name: "Prafful Gupta", roll: "202510116100161" },
  { name: "Pragya Mishra", roll: "202510116100162" },
  { name: "Prashant K. Singh", roll: "202510116100167" },
];

export function ProjectCredits() {
  return (
    <section className="project-credits" aria-labelledby="project-credits-heading">
      <div className="project-credits__eyebrow">Major Project Synopsis · 2026–27</div>
      <p id="project-credits-heading" className="project-credits__subtitle">
        An Intelligent CI/CD Pipeline Optimizer with Predictive Build Failure Analysis
      </p>

      <div className="project-credits__team">
        {TEAM.map((m) => (
          <div className="project-credits__member" key={m.roll}>
            <strong>{m.name}</strong>
            <span className="mono">{m.roll}</span>
          </div>
        ))}
      </div>

      <div className="project-credits__meta">
        <span>Section C &nbsp;|&nbsp; MCA</span>
        <span>
          Mentor: <strong>Ms. Sonam Jain</strong>
        </span>
      </div>

      <div className="project-credits__footer">
        <span>
          <strong>KIET Deemed to be University</strong> · Delhi-NCR, Ghaziabad
        </span>
        <span className="project-credits__sdg">SDG 9</span>
      </div>
    </section>
  );
}
