import { explainDriver } from "../lib/driverLabels";
import "./ShapBar.css";

/**
 * Horizontal bars, one per top contributor, sized by |SHAP value| and
 * colored by direction. This is the "explanation," not the raw feature
 * name — explainDriver() does the plain-language translation; the raw
 * feature name is kept only as a title="" for anyone curious.
 */
export function ShapBar({ contributors }) {
  if (!contributors || contributors.length === 0) return null;
  const maxAbs = Math.max(...contributors.map((c) => Math.abs(c.shap_value)));

  return (
    <div className="shap-bar">
      {contributors.map((c) => {
        const widthPct = maxAbs > 0 ? (Math.abs(c.shap_value) / maxAbs) * 100 : 0;
        const increases = c.direction === "increases risk";
        return (
          <div className="shap-bar__row" key={c.feature}>
            <div className="shap-bar__track">
              <div
                className={`shap-bar__fill ${increases ? "shap-bar__fill--up" : "shap-bar__fill--down"}`}
                style={{ width: `${widthPct}%` }}
              />
            </div>
            <p className="shap-bar__label" title={`${c.feature} = ${c.feature_value ?? "null"}`}>
              <span className={`shap-bar__arrow ${increases ? "shap-bar__arrow--up" : "shap-bar__arrow--down"}`}>
                {increases ? "▲" : "▼"}
              </span>{" "}
              {explainDriver(c)}
            </p>
          </div>
        );
      })}
    </div>
  );
}
