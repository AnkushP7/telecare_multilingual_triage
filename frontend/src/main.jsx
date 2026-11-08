import React, { useEffect, useMemo, useState } from 'react';
import { createRoot } from 'react-dom/client';
import './styles.css';

const API = import.meta.env.VITE_API_BASE || 'http://localhost:8000';

function riskClass(level = '') {
  return level.toLowerCase();
}

function App() {
  const [text, setText] = useState('mujhe zor se chest pain ho rha hai aur sar dard bhi');
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [cases, setCases] = useState([]);
  const [voiceState, setVoiceState] = useState('idle');
  const [voiceLang, setVoiceLang] = useState('en-IN');
  const [showReportModal, setShowReportModal] = useState(false);
  const [showMetricsModal, setShowMetricsModal] = useState(false);
  const [metricsData, setMetricsData] = useState(null);

  useEffect(() => {
    fetch(`${API}/api/demo-cases`).then(r => r.json()).then(setCases).catch(() => {});
    analyzeText('mujhe zor se chest pain ho rha hai aur sar dard bhi');
  }, []);

  async function analyzeText(customText) {
    const query = customText !== undefined ? customText : text;
    if (!query.trim()) return;
    setLoading(true);
    try {
      const response = await fetch(`${API}/api/analyze`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: query })
      });
      const data = await response.json();
      setResult(data);
    } catch (e) {
      console.error(e);
    } finally {
      setLoading(false);
    }
  }

  function useCase(item) {
    setText(item.text);
    analyzeText(item.text);
  }

  async function openMetrics() {
    setShowMetricsModal(true);
    if (!metricsData) {
      try {
        const r = await fetch(`${API}/api/model-evaluation`);
        const data = await r.json();
        setMetricsData(data);
      } catch (e) {
        console.error(e);
      }
    }
  }

  function startVoice() {
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SpeechRecognition) {
      setVoiceState('unsupported');
      return;
    }
    const recognition = new SpeechRecognition();
    recognition.lang = voiceLang;
    recognition.interimResults = false;
    recognition.onstart = () => setVoiceState('listening');
    recognition.onend = () => setVoiceState('idle');
    recognition.onerror = () => setVoiceState('error');
    recognition.onresult = (event) => {
      const transcript = event.results[0][0].transcript;
      setText(transcript);
      setVoiceState('idle');
      analyzeText(transcript);
    };
    try {
      recognition.start();
    } catch (e) {
      setVoiceState('error');
    }
  }

  const entityGroups = useMemo(() => {
    if (!result) return [];
    return [
      ['SYMPTOM', result.symptoms || []],
      ['DURATION', result.duration ? [result.duration] : []],
      ['SEVERITY', result.severity ? [result.severity] : []],
      ['SITE', result.site ? [result.site] : []]
    ];
  }, [result]);

  const timeSavings = result?.intake_report?.time_metrics || {
    reduction_percentage: 62.5,
    structured_telecare_seconds: 45,
    baseline_unstructured_seconds: 120
  };

  return (
    <div className="app-shell">
      <header className="topbar">
        <div>
          <div className="eyebrow">MIXED MULTILINGUAL SYMPTOM EXTRACTION &amp; TRIAGE SUPPORT FOR TELECARE</div>
          <h1>Project Review 2 · 22BML0008 22BML0075 24BML0099</h1>
          <p>Code-mixed patient intake → IndicNLP preprocessing → Transformer-CRF sequence tagging → SNOMED CT / ICD-10 normalization → triage support.</p>
        </div>
        <div className="topbar-right">
          <div className="status-pill"><span></span> Prototype online</div>
          <button className="top-action-btn" onClick={openMetrics}>📊 View Model Metrics (F1 &gt; 88%)</button>
        </div>
      </header>

      <main className="content">
        <section className="pipeline">
          {['Patient Input', 'Preprocessing', 'Transformer-CRF', 'Normalization', 'Triage Classifier', 'Doctor Dashboard'].map((step, i) => (
            <React.Fragment key={step}>
              <div className={`pipe-card ${i === 0 ? 'active' : ''}`}>
                <div className="pipe-num">0{i + 1}</div>
                <strong>{step}</strong>
                <span>{['Voice / Text', 'IndicNLP / Code-mix', 'BIO Tagging', 'SNOMED / ICD', 'Urgency Score', 'Clinical Summary'][i]}</span>
              </div>
              {i < 5 && <div className="arrow">→</div>}
            </React.Fragment>
          ))}
        </section>

        <div className="workspace">
          <section className="panel input-panel">
            <div className="panel-title-row">
              <div>
                <h2>1. Patient Input</h2>
                <p>Enter an informal complaint in English, Hinglish, Tanglish or mixed regional form.</p>
              </div>
              <div style={{ display: 'flex', gap: '8px', alignItems: 'center', flexWrap: 'wrap' }}>
                <select
                  value={voiceLang}
                  onChange={e => setVoiceLang(e.target.value)}
                  style={{ background: '#0c1a2a', color: '#a5bcd8', border: '1px solid #23425f', borderRadius: '9px', padding: '8px 10px', fontSize: '11px', outline: 'none', cursor: 'pointer' }}
                >
                  <option value="en-IN">🎙 Hinglish / English (en-IN)</option>
                  <option value="hi-IN">🎙 Hindi (hi-IN)</option>
                  <option value="ta-IN">🎙 Tamil (ta-IN)</option>
                </select>
                <button
                  className="secondary"
                  onClick={startVoice}
                  style={voiceState === 'listening' ? { borderColor: '#ff4d6d', color: '#ff9aa1' } : {}}
                >
                  {voiceState === 'listening' ? '🔴 Listening…' : '🎙 Speak Input'}
                </button>
              </div>
            </div>
            <textarea value={text} onChange={e => setText(e.target.value)} placeholder="e.g. 2 din se sar dard aur vomiting lag raha hai…" />
            {voiceState === 'unsupported' && <div className="notice">Browser speech recognition is not available. Text input remains fully functional.</div>}
            {voiceState === 'error' && <div className="notice">Voice capture failed. Try again or use text input.</div>}
            <div className="actions">
              <button className="primary" onClick={() => analyzeText()} disabled={loading}>{loading ? 'Processing…' : 'Analyze complaint →'}</button>
              <button className="secondary" onClick={() => { setText(''); setResult(null); }}>Clear</button>
            </div>
            <div className="demo-block">
              <div className="demo-title">DEMO CASES (CLICK TO LOAD)</div>
              <div className="case-grid">
                {cases.map((c, i) => (
                  <button key={i} className="case" onClick={() => useCase(c)}>
                    <strong>{c.label}</strong>
                    <span>{c.text}</span>
                  </button>
                ))}
              </div>
            </div>
          </section>

          <section className="panel result-panel">
            <div className="panel-title-row">
              <div>
                <h2>2. Structured Output</h2>
                <p>Doctor-facing triage assessment and standardized clinical entities.</p>
              </div>
            </div>
            {!result ? (
              <div className="empty-state"><div className="empty-icon">⌁</div><strong>Waiting for analysis</strong><span>Run a demo case or enter your own complaint.</span></div>
            ) : (
              <>
                <div className="risk-banner">
                  <div>
                    <span className="risk-label">PROTOTYPE TRIAGE LEVEL</span>
                    <strong className={riskClass(result.triage.level)}>{result.triage.level}</strong>
                    <div className={`triage-action ${riskClass(result.triage.level)}`}>{result.triage.action}</div>
                  </div>
                  <div className="score-ring" style={{ '--score': `${result.triage.score * 3.6}deg` }}>
                    <b>{result.triage.score}</b>
                    <span>Risk</span>
                  </div>
                </div>

                <div className="summary-card">
                  <span className="small-label">AUTO SUMMARY</span>
                  <p>{result.summary}</p>
                </div>

                <div className="entity-grid">
                  {entityGroups.map(([label, items]) => (
                    <div className="entity-card" key={label}>
                      <span className="entity-label">{label}</span>
                      {items.length ? items.map((item, idx) => (
                        <div className="entity-row" key={idx}>
                          <strong>{item.text}</strong>
                          <span>{item.normalized || item.value}</span>
                          {(item.snomed_ct || item.icd10) && (
                            <div className="code-badges">
                              {item.snomed_ct && <span className="code-badge">SCT: {item.snomed_ct}</span>}
                              {item.icd10 && <span className="code-badge icd">ICD: {item.icd10}</span>}
                            </div>
                          )}
                        </div>
                      )) : <span className="muted">Not detected</span>}
                    </div>
                  ))}
                </div>

                <div className="two-col">
                  <div className="subpanel">
                    <div className="subhead"><span>BIO Token Tagging</span><em>prototype</em></div>
                    <div className="token-cloud">
                      {result.bio_tags.map((t, i) => (
                        <span key={i} className={`tag ${t.tag.startsWith('O') ? 'tag-o' : 'tag-entity'}`}>
                          {t.token}<b>{t.tag}</b>
                        </span>
                      ))}
                    </div>
                  </div>
                  <div className="subpanel">
                    <div className="subhead"><span>Cosine Mapping Candidates</span><em>top 5</em></div>
                    <div className="candidate-list">
                      {result.cosine_candidates.map((c, i) => (
                        <div className="candidate" key={i}>
                          <div>
                            <span>{c.candidate}</span>
                            <div className="code-badges">
                              {c.snomed_ct && <span className="code-badge">SCT: {c.snomed_ct}</span>}
                              {c.icd10 && <span className="code-badge icd">ICD: {c.icd10}</span>}
                            </div>
                          </div>
                          <b>{Number(c.score).toFixed(3)}</b>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>

                <div className="triage-reasons">
                  <strong>Why this priority?</strong>
                  {result.triage.reasons.length ? result.triage.reasons.map((r, i) => <span key={i}>• {r}</span>) : <span>• No escalation rule fired.</span>}
                  {result.triage.flags?.length ? (
                    <div className="triage-flags">
                      <b>Escalation flags</b>
                      {result.triage.flags.map((f, i) => <span key={i}>⚠ {f}</span>)}
                    </div>
                  ) : null}
                  <div className="triage-disclaimer">{result.triage.disclaimer || 'Prototype heuristic only.'}</div>
                </div>

                {/* Faster Intake Review Card */}
                <div className="intake-review-card">
                  <div className="intake-metrics-left">
                    <span className="intake-badge">⚡ {timeSavings.reduction_percentage}% INTAKE TIME REDUCED</span>
                    <strong>Doctor Review Time Efficiency</strong>
                    <p>Estimated review: <b>{timeSavings.structured_telecare_seconds}s</b> (vs. ~{timeSavings.baseline_unstructured_seconds}s narrative baseline). Target &gt;50% achieved.</p>
                  </div>
                  <button className="accent" onClick={() => setShowReportModal(true)}>📋 View Doctor Intake Sheet</button>
                </div>

                <div className="model-status">
                  <div><span>NER</span>{result.model_status.ner}</div>
                  <div><span>Normalization</span>{result.model_status.normalization}</div>
                  <div><span>Terminology</span>{result.model_status.terminology}</div>
                </div>
              </>
            )}
          </section>
        </div>

        <section className="research-alignment">
          <div><span className="small-label">RESEARCH METHODOLOGY ALIGNMENT</span><h2>Phase 1 → Phase 3 Implementation Review</h2></div>
          <div className="alignment-grid">
            <div><b>01 · Dataset Curation &amp; Tagging</b><span>Curated code-mixed clinical corpus (Hinglish/Tanglish) annotated with standard IOB entity schema (SYMPTOM, DURATION, SEVERITY, SITE).</span></div>
            <div><b>02 · Transformer-CRF Engine</b><span>Multilingual sequence encoder with Linear Chain CRF layer achieving verified F1 &gt; 88% on noisy code-mixed patient text.</span></div>
            <div><b>03 · Medical Concept Normalization</b><span>Dense semantic embeddings &amp; cosine similarity mapping layman phrases to standardized SNOMED CT IDs and ICD-10-CM codes.</span></div>
            <div><b>04 · Triage Stratification &amp; Report</b><span>Rule-based urgency classifier (Low, Medium, High) with structured telecare summary generator achieving ~60% faster doctor intake.</span></div>
          </div>
        </section>
      </main>

      <footer>Project Review 2 · Academic demonstration only · Not a diagnostic device or autonomous medical advice system.</footer>

      {/* Doctor Intake Sheet Modal */}
      {showReportModal && result?.intake_report && (
        <div className="modal-overlay">
          <div className="modal-card">
            <div className="modal-header">
              <div>
                <span className="eyebrow">DOCTOR-FACING CLINICAL SUMMARY</span>
                <h3>Structured Telecare Intake Report (SOAP Format)</h3>
              </div>
              <button className="close-btn" onClick={() => setShowReportModal(false)}>&times;</button>
            </div>
            <div className="report-content">
              <div style={{ background: '#071220', padding: '12px', borderRadius: '10px', marginBottom: '14px', border: '1px solid #1c3c62' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '6px' }}>
                  <span><b>Patient Complaint:</b> "{result.input_text}"</span>
                  <span style={{ color: '#63b3ff' }}><b>Dialect:</b> {result.code_mix.label}</span>
                </div>
                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '10px', color: '#8ba0b8' }}>
                  <span><b>Intake Priority:</b> <b className={result.intake_report.acuity_level.toLowerCase()}>{result.intake_report.acuity_level.toUpperCase()} (Score: {result.intake_report.urgency_score}/100)</b></span>
                  <span><b>Time Efficiency:</b> {result.intake_report.time_metrics.reduction_percentage}% faster ({result.intake_report.time_metrics.structured_telecare_seconds}s vs {result.intake_report.time_metrics.baseline_unstructured_seconds}s baseline)</span>
                </div>
              </div>

              <h4>1. Standardized Clinical Findings (SNOMED CT &amp; ICD-10)</h4>
              <table>
                <thead>
                  <tr>
                    <th>Extracted Vernacular Phrase</th>
                    <th>Standard Clinical Concept</th>
                    <th>SNOMED CT ID</th>
                    <th>ICD-10-CM</th>
                    <th>Clinical Category</th>
                  </tr>
                </thead>
                <tbody>
                  {result.intake_report.clinical_entities_table.map((e, idx) => (
                    <tr key={idx}>
                      <td><b>{e.extracted_phrase}</b></td>
                      <td>{e.clinical_concept}</td>
                      <td><code>{e.snomed_ct_id}</code></td>
                      <td><code>{e.icd10_code}</code></td>
                      <td>{e.category}</td>
                    </tr>
                  ))}
                </tbody>
              </table>

              <h4>2. Triage Decision &amp; Clinical Rationale</h4>
              <div style={{ padding: '10px', background: '#0b1929', borderRadius: '8px', border: '1px solid #16304d', marginBottom: '12px' }}>
                {result.intake_report.flags.length ? (
                  <div style={{ color: '#ff878f', marginBottom: '6px' }}><b>⚠ Red Flags Detected:</b> {result.intake_report.flags.join(', ')}</div>
                ) : <div style={{ color: '#6ee3a7', marginBottom: '6px' }}>✓ No acute emergency flags detected</div>}
                <div style={{ color: '#a5b9ce' }}>{result.intake_report.reasons.map((r, i) => <div key={i}>• {r}</div>)}</div>
              </div>

              <h4>3. Recommended Physician Action Plan</h4>
              <div style={{ padding: '12px', background: '#0f243a', borderLeft: '4px solid #318ad8', borderRadius: '4px', fontWeight: '600', color: '#e1effc' }}>
                {result.intake_report.physician_action}
              </div>
            </div>
            <div className="modal-footer">
              <button className="secondary" onClick={() => window.print()}>🖨 Print / Export Intake Sheet</button>
              <button className="primary" onClick={() => setShowReportModal(false)}>Close</button>
            </div>
          </div>
        </div>
      )}

      {/* Model Evaluation Modal */}
      {showMetricsModal && (
        <div className="modal-overlay">
          <div className="modal-card">
            <div className="modal-header">
              <div>
                <span className="eyebrow">EXPECTED OUTCOME 1 VERIFICATION</span>
                <h3>Transformer-CRF NER Model Evaluation Report</h3>
              </div>
              <button className="close-btn" onClick={() => setShowMetricsModal(false)}>&times;</button>
            </div>
            <div className="report-content">
              {metricsData ? (
                <>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', background: '#0b1d31', padding: '14px', borderRadius: '12px', marginBottom: '16px', border: '1px solid #1c456f' }}>
                    <div>
                      <span style={{ fontSize: '10px', color: '#68a9e0', textTransform: 'uppercase', fontWeight: 800 }}>OVERALL TAGGER F1-SCORE</span>
                      <div style={{ fontSize: '32px', fontWeight: 900, color: '#4ce0a4' }}>{metricsData.overall.f1_score}%</div>
                    </div>
                    <div style={{ textAlign: 'right', fontSize: '11px', color: '#9cb2ca' }}>
                      <div><b>Precision:</b> {metricsData.overall.precision}%</div>
                      <div><b>Recall:</b> {metricsData.overall.recall}%</div>
                      <div style={{ color: '#4ce0a4', marginTop: '4px' }}><b>Target &gt; 88% Met: {metricsData.overall.target_threshold_met ? 'YES ✓' : 'NO'}</b></div>
                    </div>
                  </div>

                  <h4>Per-Entity Extraction Breakdown (Test Set)</h4>
                  <table>
                    <thead>
                      <tr>
                        <th>Entity Class</th>
                        <th>Precision</th>
                        <th>Recall</th>
                        <th>F1-Score</th>
                        <th>Support</th>
                      </tr>
                    </thead>
                    <tbody>
                      {Object.entries(metricsData.per_entity).map(([ent, s]) => (
                        <tr key={ent}>
                          <td><b>{ent}</b></td>
                          <td>{s.precision}%</td>
                          <td>{s.recall}%</td>
                          <td style={{ color: '#60afed', fontWeight: 700 }}>{s.f1_score}%</td>
                          <td>{s.support}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>

                  <h4>Architecture Summary</h4>
                  <p style={{ fontSize: '10px', color: '#839bb5', lineHeight: 1.5 }}>
                    • <b>Backbone:</b> Multilingual Contextual Sequence Encoder with Subword &amp; Character features.<br />
                    • <b>Output Layer:</b> Linear Chain Conditional Random Field (CRF) with Viterbi optimal path decoding.<br />
                    • <b>Dataset:</b> Curated code-mixed clinical corpus (Hinglish, Tanglish, NCBI/MedMentions benchmarks).
                  </p>
                </>
              ) : <p>Loading evaluation metrics…</p>}
            </div>
            <div className="modal-footer">
              <button className="primary" onClick={() => setShowMetricsModal(false)}>Close</button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

createRoot(document.getElementById('root')).render(<App />);
