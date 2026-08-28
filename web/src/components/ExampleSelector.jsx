import "./ExampleSelector.css";

export function ExampleSelector({ examples, selectedId, onSelect }) {
  return (
    <div className="example-selector" role="tablist" aria-label="Demo examples">
      {examples.map((ex) => (
        <button
          key={ex.id}
          role="tab"
          aria-selected={ex.id === selectedId}
          className={`example-selector__tab ${ex.id === selectedId ? "example-selector__tab--active" : ""}`}
          onClick={() => onSelect(ex.id)}
        >
          {ex.label}
        </button>
      ))}
    </div>
  );
}
