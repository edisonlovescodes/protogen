import { useRef, useState } from "react";
import type { CSSProperties, FormEvent } from "react";
import "./App.css";

const API = (import.meta.env.VITE_API_BASE_URL || "/api").replace(/\/$/, "");
const source = "https://github.com/edisonlovescodes/protogen";
const controls = [
  ["thermostable", "Hydrophobic / aromatic", "Composition balance"],
  ["enzyme", "Charged / polar", "Composition balance"],
  ["membrane", "Hydrophobic fraction", "Residue enrichment"],
  ["structural", "Aromatic / polar", "Composition balance"],
  ["flexible", "Polar fraction", "Residue enrichment"],
];
const alphabet = "ARNDCQEGHILKMFPSTWYV";
const category = (residue: string) =>
  "AILMFVP".includes(residue)
    ? "hydrophobic"
    : "DEKR".includes(residue)
      ? "charged"
      : "other";

type Notice =
  | { kind: "idle" }
  | { kind: "copying"; index: number }
  | { kind: "copied"; index: number }
  | { kind: "exporting" }
  | { kind: "failed"; message: string };
type Studio =
  | { kind: "empty" }
  | { kind: "generating" }
  | { kind: "failed"; message: string }
  | { kind: "ready"; sequences: string[]; notice: Notice };
const icons = {
  arrow: "M4 12h15m-6-6 6 6-6 6",
  copy: "M8 8h12v12H8zM4 16V4h12",
  download: "M12 3v12m-5-5 5 5 5-5M5 17v4h14v-4",
};
function assert(condition: unknown, message: string): asserts condition {
  if (!condition) throw new Error(message);
}
function Icon({ kind }: { kind: keyof typeof icons }) {
  return (
    <svg viewBox="0 0 24 24" fill="none" aria-hidden="true">
      <path d={icons[kind]} />
    </svg>
  );
}
function PolymerMark() {
  return (
    <svg
      className="polymer-mark"
      viewBox="0 0 36 36"
      fill="none"
      aria-hidden="true"
    >
      <path d="m18 3 13 7.5v15L18 33 5 25.5v-15Z" />
      <path d="M5 10.5 18 18l13-7.5M18 18v15M18 3v15" />
    </svg>
  );
}
function SequenceMap() {
  return (
    <svg
      viewBox="0 0 480 240"
      className="sequence-map"
      role="img"
      aria-label="Schematic of the 20 amino-acid building blocks"
    >
      <path className="map-path" d="M25 62H405Q455 62 455 114T405 166H25" />
      {alphabet.split("").map((aa, i) => {
        const x = i < 10 ? 25 + i * 43 : 412 - (i - 10) * 43;
        const y = i < 10 ? 62 : 166;
        return (
          <g key={aa} className={`map-node ${category(aa)}`}>
            <circle cx={x} cy={y} r="14" />
            <text x={x} y={y + 4}>
              {aa}
            </text>
          </g>
        );
      })}
      <text x="25" y="24" className="map-caption">
        20 RESIDUES / ONE ALPHABET
      </text>
      <text x="25" y="219" className="map-caption">
        A SCHEMATIC, NOT A GENERATED SEQUENCE
      </text>
      <path className="map-tick" d="M465 52v-8h-8M8 176v8h8" />
    </svg>
  );
}
async function post(path: string, payload: object) {
  const response = await fetch(`${API}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  const body = await response.json();
  if (!response.ok) {
    assert(
      typeof body.detail === "string" || Array.isArray(body.detail),
      "Invalid API error response",
    );
    throw new Error(
      typeof body.detail === "string"
        ? body.detail
        : body.detail.map((item: { msg: string }) => item.msg).join("; "),
    );
  }
  return body;
}

export default function App() {
  const [properties, setProperties] = useState(
    Object.fromEntries(controls.map(([key]) => [key, 0.5])) as Record<
      string,
      number
    >,
  );
  const [studio, setStudio] = useState<Studio>({ kind: "empty" });
  const batchNumber = useRef(0);
  const sequences = studio.kind === "ready" ? studio.sequences : [];
  const notice =
    studio.kind === "ready" ? studio.notice : { kind: "idle" as const };
  const loading =
    studio.kind === "generating" ||
    notice.kind === "copying" ||
    notice.kind === "exporting";
  const error =
    studio.kind === "failed"
      ? studio.message
      : notice.kind === "failed"
        ? notice.message
        : null;
  const copied = notice.kind === "copied" ? notice.index : null;

  async function generate(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const settings = Object.fromEntries(new FormData(event.currentTarget));
    assert(
      ["max_length", "temperature", "num_sequences"].every(
        (key) => typeof settings[key] === "string",
      ),
      "Expected sampling settings",
    );
    setStudio({ kind: "generating" });
    try {
      const body = await post("/generate", {
        properties,
        max_length: Number(settings.max_length),
        temperature: Number(settings.temperature),
        num_sequences: Number(settings.num_sequences),
      });
      assert(
        Array.isArray(body.sequences) &&
          body.sequences.every(
            (seq: unknown) =>
              typeof seq === "string" && /^[ARNDCQEGHILKMFPSTWYV]+$/.test(seq),
          ),
        "Invalid generated sequences",
      );
      batchNumber.current += 1;
      setStudio({
        kind: "ready",
        sequences: body.sequences,
        notice: { kind: "idle" },
      });
    } catch (error) {
      assert(error instanceof Error, "Expected a generation error");
      setStudio({ kind: "failed", message: error.message });
    }
  }
  async function exportFasta() {
    assert(studio.kind === "ready", "Generate a batch before exporting");
    setStudio({ ...studio, notice: { kind: "exporting" } });
    try {
      const body = await post("/export", { sequences });
      assert(typeof body.fasta === "string", "Invalid FASTA response");
      const url = URL.createObjectURL(
        new Blob([body.fasta], { type: "text/plain" }),
      );
      const link = document.createElement("a");
      link.href = url;
      link.download = "protogen.fasta";
      link.click();
      URL.revokeObjectURL(url);
      setStudio({ ...studio, notice: { kind: "idle" } });
    } catch (error) {
      assert(error instanceof Error, "Expected an export error");
      setStudio({
        ...studio,
        notice: { kind: "failed", message: error.message },
      });
    }
  }
  async function copy(index: number) {
    assert(studio.kind === "ready", "Generate a batch before copying");
    setStudio({ ...studio, notice: { kind: "copying", index } });
    try {
      await navigator.clipboard.writeText(studio.sequences[index]);
      setStudio({ ...studio, notice: { kind: "copied", index } });
    } catch {
      setStudio({
        ...studio,
        notice: {
          kind: "failed",
          message:
            "Copy is unavailable in this browser. Use the FASTA download instead.",
        },
      });
    }
  }

  return (
    <div className="proto-page">
      <a className="skip-link" href="#studio">
        Skip to sequence studio
      </a>
      <header className="proto-header">
        <a className="proto-wordmark" href="#">
          <PolymerMark />
          protogen<span>/</span>
        </a>
        <div className="proto-nav">
          <span>SEQUENCE STUDIO</span>
          <a href={source} target="_blank" rel="noreferrer">
            Explore the code <Icon kind="arrow" />
          </a>
        </div>
      </header>
      <main>
        <section className="proto-hero" aria-labelledby="proto-title">
          <div>
            <p className="eyebrow">
              <span />
              EXPLORATIONS IN PROTEIN SEQUENCES
            </p>
            <h1 id="proto-title">
              Sequence,
              <br />
              <em>by design.</em>
            </h1>
            <p className="hero-description">
              Shape the composition.
              <br />
              Explore what comes next.
            </p>
          </div>
          <div className="hero-map">
            <SequenceMap />
            <div className="map-legend">
              <span className="hydrophobic">Hydrophobic</span>
              <span className="charged">Charged</span>
              <span className="other">Other residues</span>
            </div>
          </div>
        </section>
        <div className="studio-heading">
          <span>01 / GENERATION STUDIO</span>
          <span>COMPOSITION → SEQUENCE → FASTA</span>
        </div>
        <section
          className="proto-studio"
          id="studio"
          aria-label="Sequence generation studio"
        >
          <form className="control-panel" onSubmit={generate}>
            <div className="panel-heading">
              <h2>Set the balance.</h2>
              <span>5 CONTROLS</span>
            </div>
            <p className="control-note">
              These controls tune composition proxies, not experimentally
              verified biological traits.
            </p>
            <fieldset disabled={loading} className="composition-controls">
              <legend className="sr-only">Composition controls</legend>
              {controls.map(([key, label, note], i) => (
                <div className="property" key={key}>
                  <label htmlFor={key}>
                    <span className="property-index">0{i + 1}</span>
                    <span>
                      {label}
                      <small>{note}</small>
                    </span>
                    <output htmlFor={key}>{properties[key].toFixed(2)}</output>
                  </label>
                  <input
                    id={key}
                    aria-label={label}
                    type="range"
                    min="0"
                    max="1"
                    step="0.05"
                    value={properties[key]}
                    style={
                      { "--fill": `${properties[key] * 100}%` } as CSSProperties
                    }
                    onChange={(e) =>
                      setProperties({
                        ...properties,
                        [key]: Number(e.target.value),
                      })
                    }
                  />
                </div>
              ))}
            </fieldset>
            <fieldset disabled={loading} className="sampling">
              <legend>SAMPLING SETTINGS</legend>
              <div className="sampling-grid">
                <label htmlFor="length">
                  Token limit
                  <input
                    id="length"
                    type="number"
                    min="2"
                    max="150"
                    required
                    name="max_length"
                    defaultValue={100}
                  />
                </label>
                <label htmlFor="temperature">
                  Temperature
                  <input
                    id="temperature"
                    type="number"
                    min="0.1"
                    max="2"
                    step="0.1"
                    required
                    name="temperature"
                    defaultValue={1}
                  />
                </label>
                <label htmlFor="count">
                  Sequences
                  <input
                    id="count"
                    type="number"
                    min="1"
                    max="10"
                    required
                    name="num_sequences"
                    defaultValue={1}
                  />
                </label>
              </div>
              <p>
                Token limit includes start/end. Sequences may finish earlier.
              </p>
            </fieldset>
            <button
              className="generate-button"
              type="submit"
              disabled={loading}
            >
              {studio.kind === "generating" ? "Generating…" : "Generate sequences"}
              <Icon kind="arrow" />
            </button>
            {error && (
              <p className="error-message" role="alert">
                {error}
              </p>
            )}
          </form>
          <div
            className="sequence-panel"
            aria-live="polite"
            aria-busy={loading}
          >
            <div className="sequence-toolbar">
              <span>
                OUTPUT /{" "}
                {sequences.length
                  ? `BATCH ${String(batchNumber.current).padStart(2, "0")}`
                  : "NO BATCH YET"}
              </span>
              {sequences.length > 0 && (
                <button
                  className="export-button"
                  onClick={exportFasta}
                  disabled={loading}
                >
                  Download FASTA <Icon kind="download" />
                </button>
              )}
            </div>
            {sequences.length ? (
              <div className="sequence-list">
                {sequences.map((sequence, i) => (
                  <article className="sequence" key={i}>
                    <header>
                      <h3>
                        <span>0{i + 1}</span>Sequence {i + 1}
                      </h3>
                      <div>
                        <span>{sequence.length} residues</span>
                        <button
                          aria-label={`Copy sequence ${i + 1}`}
                          onClick={() => copy(i)}
                          disabled={loading}
                        >
                          <Icon kind="copy" />
                        </button>
                      </div>
                    </header>
                    <div className="residue-string" aria-label={sequence}>
                      {sequence.split("").map((aa, j) => (
                        <span
                          className={category(aa)}
                          key={j}
                          title={`Residue ${j + 1}: ${aa}`}
                        >
                          {aa}
                        </span>
                      ))}
                    </div>
                    <div className="sequence-caption">
                      <span>
                        {copied === i
                          ? "COPIED TO CLIPBOARD"
                          : "MODEL-GENERATED / FUNCTION UNVALIDATED"}
                      </span>
                      <span>
                        N <i>—</i> C
                      </span>
                    </div>
                  </article>
                ))}
              </div>
            ) : (
              <div className={`sequence-empty ${loading ? "working" : ""}`}>
                <div className="empty-graphic" aria-hidden="true">
                  {["A", "R", "N", "D", "C", "Q", "E", "G"].map((aa, i) => (
                    <span key={aa} style={{ "--index": i } as CSSProperties}>
                      {aa}
                    </span>
                  ))}
                </div>
                <p className="empty-index">YOUR NEXT EXPLORATION</p>
                <h2>
                  {loading ? (
                    <>
                      Finding the
                      <br />
                      next sequence.
                    </>
                  ) : (
                    <>
                      A blank canvas.
                      <br />A new sequence.
                    </>
                  )}
                </h2>
                <p>
                  Adjust the five composition controls and generate a batch.
                  Your sequences will appear here, ready to inspect and export.
                </p>
                <div className="empty-bottom">
                  <span>20 BUILDING BLOCKS</span>
                  <span>
                    ONE STARTING POINT <Icon kind="arrow" />
                  </span>
                </div>
              </div>
            )}
            {sequences.length > 0 && (
              <div className="sequence-footnote">
                <p>
                  Sequences are generated by a small research model. Composition
                  adherence and biological function remain unvalidated.
                </p>
                <button
                  onClick={() => setStudio({ kind: "empty" })}
                  disabled={loading}
                >
                  Clear batch
                </button>
              </div>
            )}
          </div>
        </section>
        <section className="proto-method">
          <p className="eyebrow">THE EXPERIMENT</p>
          <div>
            <h2>
              A small model.
              <br />
              An open process.
            </h2>
            <p>
              This studio explores a causal sequence model trained on a small,
              deduplicated dataset. The full training protocol, evaluation, and
              limitations are documented alongside the code.
            </p>
            <a href={source} target="_blank" rel="noreferrer">
              Read the experiment <Icon kind="arrow" />
            </a>
          </div>
        </section>
      </main>
      <footer>
        <span>PROTOGEN / SEQUENCE EXPERIMENTS</span>
        <span>DESIGNED & BUILT BY EDISON ESSIEN</span>
      </footer>
    </div>
  );
}
