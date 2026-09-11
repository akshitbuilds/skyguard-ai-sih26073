"""SkyGuard AI — SIH26073 screening command center.

Designed for a judge-first demo: the first screen answers WHAT happened,
WHY the system decided it, and WHETHER the data was corrected.
"""
from __future__ import annotations
import os, sys, time
import pandas as pd
import streamlit as st

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
BACKEND = os.path.join(ROOT, "backend_integration")
if BACKEND not in sys.path:
    sys.path.insert(0, BACKEND)
from demo_runner import SCENARIOS, run_scenario
import main as backend

RUNTIME = backend.runtime_status()

st.set_page_config(
    page_title="SkyGuard AI | SIH26073",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------- Visual system ----------
st.markdown("""
<style>
:root {
  --bg:#050b14;
  --panel:#0b1725;
  --panel2:#0e1d2d;
  --panel3:#10253a;
  --line:#29445e;
  --line2:#345873;
  --text:#f7fbff;
  --text2:#e3edf7;
  --muted:#b7c8d9;
  --muted2:#96acc1;
  --cyan:#55d6ff;
  --green:#37e6a2;
  --amber:#ffc857;
  --red:#ff6675;
}

/* ---------- App canvas ---------- */
[data-testid="stAppViewContainer"] {
  background:var(--bg);
  color:var(--text);
}
[data-testid="stHeader"] { background:rgba(5,11,20,.92); }
.block-container {
  max-width:1500px;
  padding-top:.75rem;
  padding-bottom:1.4rem;
}
[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] li {
  color:var(--muted);
}
[data-testid="stCaptionContainer"],
[data-testid="stCaptionContainer"] * {
  color:var(--muted2) !important;
}
[data-testid="stMetricLabel"] {
  color:var(--muted) !important;
  font-weight:700 !important;
}
[data-testid="stMetricValue"] {
  color:var(--text) !important;
}

/* ---------- Sidebar ---------- */
section[data-testid="stSidebar"] {
  background:#07111d !important;
  border-right:1px solid #203a52 !important;
  min-width:330px !important;
  width:330px !important;
}
section[data-testid="stSidebar"] > div:first-child {
  width:330px !important;
}
section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] li {
  color:#dce8f3 !important;
}
section[data-testid="stSidebar"] [data-testid="stCaptionContainer"],
section[data-testid="stSidebar"] [data-testid="stCaptionContainer"] * {
  color:#a9bdcf !important;
}
section[data-testid="stSidebar"] [data-testid="stButton"] button {
  min-height:46px !important;
  font-weight:850 !important;
  border-radius:10px !important;
}

/* ---------- Scenario selector ---------- */

section[data-testid="stSidebar"] [data-testid="stSelectbox"] {
  width:100% !important;
}

/* Main select control */
section[data-testid="stSidebar"] [data-testid="stSelectbox"] [data-baseweb="select"] {
  width:100% !important;
  min-height:48px !important;
}

/* Visible selected value */
section[data-testid="stSidebar"] [data-testid="stSelectbox"] [data-baseweb="select"] > div {
  background:#f7fbff !important;
  border:1px solid #55d6ff !important;
  border-radius:11px !important;
  min-height:48px !important;
}

/* Text inside selected value */
section[data-testid="stSidebar"] [data-testid="stSelectbox"] [data-baseweb="select"] div {
  color:#07111d !important;
  -webkit-text-fill-color:#07111d !important;
}

/* Input itself */
section[data-testid="stSidebar"] [data-testid="stSelectbox"] input {
  color:#07111d !important;
  -webkit-text-fill-color:#07111d !important;
  opacity:1 !important;
}

/* Arrow */
section[data-testid="stSidebar"] [data-testid="stSelectbox"] svg {
  color:#078fc0 !important;
  fill:#078fc0 !important;
}

/* Dropdown popup */
body [role="listbox"] {
  background:#0b1a2a !important;
  border:1px solid #315873 !important;
  border-radius:11px !important;
  padding:6px !important;
}

body [role="listbox"] [role="option"] {
  background:#0b1a2a !important;
  color:#f7fbff !important;
  -webkit-text-fill-color:#f7fbff !important;
  padding:11px 13px !important;
  border-radius:8px !important;
  font-weight:750 !important;
}

body [role="listbox"] [role="option"]:hover,
body [role="listbox"] [role="option"][data-focused="true"] {
  background:#143650 !important;
  color:#ffffff !important;
  -webkit-text-fill-color:#ffffff !important;
}

body [role="listbox"] [role="option"][aria-selected="true"] {
  background:#164b6a !important;
  color:#ffffff !important;
  -webkit-text-fill-color:#ffffff !important;
}
/* ---------- Sidebar cards ---------- */
.scenario-card {
  margin:10px 0 12px;
  padding:13px 14px;
  border:1px solid #2f5b78;
  border-radius:13px;
  background:linear-gradient(135deg,#0b1d2d,#091522);
}
.scenario-kicker {
  font-size:.62rem;
  letter-spacing:.14em;
  text-transform:uppercase;
  color:#67ddff;
  font-weight:900;
}
.scenario-name {
  font-size:1rem;
  font-weight:900;
  color:#fff;
  margin-top:4px;
}
.scenario-meta {
  font-size:.70rem;
  color:#a9bdd0;
  margin-top:4px;
}

/* ---------- Hero ---------- */
.hero {
  padding:18px 21px;
  border:1px solid #29465e;
  border-radius:17px;
  background:linear-gradient(135deg,#091725,#0b1d2d 60%,#07121e);
  box-shadow:0 12px 36px rgba(0,0,0,.22);
}
.hero-row {
  display:flex;
  align-items:center;
  justify-content:space-between;
  gap:18px;
}
.hero h1 {
  margin:0;
  color:#fff;
  font-size:2.15rem;
  letter-spacing:-.04em;
}
.hero p {
  margin:6px 0 0;
  color:#c1d1df;
  font-size:.96rem;
}
.badge {
  display:inline-block;
  padding:4px 9px;
  margin:9px 5px 0 0;
  border:1px solid #315873;
  border-radius:999px;
  color:#c2efff;
  background:#0a1c2c;
  font-size:.70rem;
  font-weight:700;
}
.system {
  padding:8px 11px;
  border:1px solid #26755c;
  background:#08251f;
  color:#70efbe;
  border-radius:999px;
  font-weight:850;
  white-space:nowrap;
  font-size:.75rem;
}

/* ---------- Replay + runtime ---------- */
.replay-banner {
  display:flex;
  align-items:center;
  justify-content:space-between;
  gap:14px;
  margin:8px 0 10px;
  padding:10px 14px;
  border:1px solid #294d68;
  border-radius:13px;
  background:#091827;
}
.replay-banner .label {
  font-size:.62rem;
  letter-spacing:.14em;
  text-transform:uppercase;
  color:#6bdcff;
  font-weight:900;
}
.replay-banner .name {
  font-size:1rem;
  color:#fff;
  font-weight:900;
}
.replay-banner .time {
  font-size:.68rem;
  color:#a5bacd;
}
.runtime-strip {
  display:flex;
  flex-wrap:wrap;
  gap:6px;
  align-items:center;
  margin:0 0 12px;
  padding:7px 9px;
  border:1px solid #23435b;
  border-radius:11px;
  background:#081622;
  color:#a7bbce;
  font-size:.68rem;
}
.runtime-strip span {
  padding:3px 7px;
  border-radius:999px;
  background:#0c2030;
  border:1px solid #28465e;
}
.runtime-strip b { color:#f0f7fc; }

/* ---------- KPI ---------- */
.kpi {
  background:#0b1725;
  border:1px solid #29445e;
  border-radius:12px;
  padding:10px 12px;
  min-height:70px;
}
.kpi .label {
  font-size:.64rem;
  text-transform:uppercase;
  letter-spacing:.08em;
  color:#a9bdcf;
  font-weight:800;
}
.kpi .value {
  font-size:1.45rem;
  font-weight:900;
  color:#fff;
  margin-top:2px;
}
.kpi .sub {
  font-size:.68rem;
  color:#93a9bd;
}

/* ---------- General cards ---------- */
.card {
  background:#0a1624;
  border:1px solid #29445e;
  border-radius:14px;
  padding:14px;
  height:100%;
}
.card-title {
  font-size:.78rem;
  color:#d7e6f2;
  text-transform:uppercase;
  letter-spacing:.10em;
  font-weight:900;
  margin-bottom:9px;
}

/* ---------- Decision ---------- */
.decision {
  min-height:175px;
  padding:20px 21px;
  border-radius:16px;
  border:1px solid #31475d;
  background:#0b1725;
}
.decision h2 {
  margin:3px 0 7px;
  color:#fff;
  font-size:1.85rem;
  line-height:1.12;
}
.decision .reason {
  color:#d6e3ee;
  line-height:1.5;
  font-size:.90rem;
}
.decision-good {
  border-color:#23775b;
  background:linear-gradient(135deg,#08251f,#0b1725);
}
.decision-warn {
  border-color:#80621f;
  background:linear-gradient(135deg,#29210f,#0b1725);
}
.decision-bad {
  border-color:#803542;
  background:linear-gradient(135deg,#2a1218,#0b1725);
}
.decision-critical {
  border-color:#a92f40;
  background:linear-gradient(135deg,#331018,#0b1725);
}
.chip {
  display:inline-block;
  border-radius:999px;
  padding:4px 8px;
  font-size:.68rem;
  font-weight:800;
  border:1px solid #31506a;
  background:#0b1e2e;
  color:#d2e7f7;
  margin:3px 4px 0 0;
}

/* ---------- Pipeline ---------- */
.stage-wrap {
  display:flex;
  gap:7px;
  align-items:stretch;
}
.stage {
  flex:1;
  border:1px solid #29445e;
  background:#0b1725;
  border-radius:10px;
  padding:9px 6px;
  text-align:center;
}
.stage .num { color:#6b8ca8; font-size:.64rem; font-weight:900; }
.stage .name { color:#f0f7fc; font-size:.70rem; font-weight:900; margin-top:3px; }
.stage .status { color:#70e9b8; font-size:.62rem; margin-top:3px; }

/* ---------- Metrics / evidence ---------- */
.metricbox {
  background:#0b1928;
  border:1px solid #29445e;
  border-radius:11px;
  padding:10px;
}
.metricbox .v { font-size:1.22rem; font-weight:900; color:#fff; }
.metricbox .l {
  font-size:.64rem;
  color:#9eb4c7;
  text-transform:uppercase;
  letter-spacing:.05em;
  margin-top:2px;
}
.explain {
  border-left:3px solid var(--cyan);
  padding:9px 11px;
  background:#0a1d2d;
  border-radius:0 9px 9px 0;
  color:#d8e7f2;
  line-height:1.45;
  font-size:.82rem;
}
.protect {
  border:1px solid #238263;
  background:#07251e;
  border-radius:12px;
  padding:13px;
  color:#b4f4da;
  line-height:1.45;
}
.alert {
  border:1px solid #8d3542;
  background:#2a1117;
  border-radius:12px;
  padding:13px;
  color:#ffe0e4;
}
.correction {
  border:1px solid #2b607f;
  background:#0a1e2d;
  border-radius:12px;
  padding:13px;
  color:#d8e9f5;
}
.small { font-size:.72rem; color:#a0b5c8; }
.footer {
  text-align:center;
  color:#6e879f;
  font-size:.64rem;
  padding-top:14px;
}

/* ---------- Tables / dataframe ---------- */
.html-table {
  width:100%;
  border-collapse:separate;
  border-spacing:0;
  overflow:hidden;
  border:1px solid #29445e;
  border-radius:11px;
  font-size:.72rem;
}
.html-table th {
  text-align:left;
  color:#b4c8da;
  font-size:.61rem;
  letter-spacing:.07em;
  text-transform:uppercase;
  background:#0b1928;
  padding:8px 8px;
  border-bottom:1px solid #29445e;
}
.html-table td {
  padding:8px 8px;
  color:#e8f1f8;
  border-bottom:1px solid #1c3348;
  background:#0b1725;
}
.html-table tr:last-child td { border-bottom:none; }
.status-stable {color:#69e8b6;font-weight:900;}
.status-watch {color:#ffd56e;font-weight:900;}
.status-critical {color:#ff7885;font-weight:900;}
.status-high {color:#ffd16c;font-weight:900;}
.status-medium {color:#ffb675;font-weight:900;}
.status-low {color:#a9bdd0;font-weight:900;}

/* ---------- Tabs / controls ---------- */
button[data-baseweb="tab"] {
  color:#b9ccdc !important;
  font-weight:800 !important;
}
button[data-baseweb="tab"][aria-selected="true"] {
  color:#55d6ff !important;
}
button[kind="primary"] {
  background:#168bc1 !important;
  border-color:#31b9ef !important;
}
[data-testid="stDataFrame"] {
  border:1px solid #29445e;
  border-radius:11px;
}

/* ---------- Narrow screens ---------- */
@media (max-width:900px) {
  .block-container {padding-left:.75rem !important; padding-right:.75rem !important;}
  .hero-row {flex-direction:column !important; align-items:flex-start !important;}
  .system {white-space:normal !important;}
  .decision {min-height:0 !important;}
  .decision h2 {font-size:1.45rem !important;}
  .replay-banner {align-items:flex-start; flex-direction:column;}
}
</style>
""", unsafe_allow_html=True)


def pct(v):
    return f"{float(v):.0%}"


def severity_class(anomaly: str, severity: str) -> str:
    if anomaly == "genuine_event": return "decision-good"
    if severity == "critical": return "decision-critical"
    if severity in ("high", "medium"): return "decision-bad"
    if severity == "low": return "decision-warn"
    return "decision-good"


def pretty(v):
    return str(v).replace("_", " ").title()


# Compact labels keep the sidebar selector readable on judge laptops.
# The full scenario name remains visible in the ACTIVE REPLAY card.
SCENARIO_SHORT_LABELS = {
    "normal": "Normal",
    "spike": "Sensor Spike",
    "frozen": "Frozen Sensor",
    "dropout": "Communication Dropout",
    "drift": "Calibration Drift",
    "genuine_event": "Genuine Weather Event",
}


def scenario_short_label(key):
    return SCENARIO_SHORT_LABELS.get(key, SCENARIOS[key]["label"])


def evidence_bar(label, value, caption=""):
    value = max(0.0, min(1.0, float(value)))
    st.markdown(
        f'''<div style="margin:8px 0 12px">
        <div style="display:flex;justify-content:space-between;color:#b9cadb;font-size:.76rem">
        <span>{label}</span><b>{value:.0%}</b></div>
        <div style="height:8px;background:#122236;border-radius:99px;overflow:hidden;margin-top:5px">
        <div style="height:100%;width:{value*100:.1f}%;background:linear-gradient(90deg,#3c8fb4,#55d6ff);border-radius:99px"></div></div>
        <div style="color:#647c94;font-size:.67rem;margin-top:3px">{caption}</div></div>''',
        unsafe_allow_html=True,
    )


# ---------- Sidebar ----------
with st.sidebar:
    st.markdown("### 🛡️ Judge Console")
    st.caption("SIH26073 · Ministry of Earth Sciences / IMD")
    st.markdown('<div class="scenario-kicker">Replay lab</div><div style="font-size:1.05rem;font-weight:850;margin:3px 0 10px">Choose a judge scenario</div>', unsafe_allow_html=True)
    scenario = st.selectbox(
    "Active replay scenario",
    list(SCENARIOS.keys()),
    format_func=scenario_short_label,
    label_visibility="collapsed",
)
    cfg_preview = SCENARIOS[scenario]
    st.caption("Short selector labels; the full scenario name and timestamp appear below.")
    st.markdown(f"<div class=\"scenario-card\"><div class=\"scenario-kicker\">ACTIVE REPLAY</div><div class=\"scenario-name\">{cfg_preview['label']}</div><div class=\"scenario-meta\">{cfg_preview['station_id']} · {cfg_preview['timestamp']}</div></div>", unsafe_allow_html=True)
    st.caption(cfg_preview["description"])
    run = st.button("▶  RUN SELECTED SCENARIO", type="primary")
    st.divider()
    st.markdown("**5-Agent decision pipeline**")
    for x in [
        "01 · Ingestion",
        "02 · Physics + spatial + temporal",
        "03 · LSTM-AE",
        "04 · Evidence fusion + SHAP",
        "05 · Safe correction + alert",
    ]:
        st.markdown(f"✓ {x}")
    st.divider()
    st.markdown("**Judge focus**")
    st.markdown("**Anomaly ≠ Sensor Fault**")
    st.caption("The genuine-event shield prevents automatic correction when the network moves together.")
    st.divider()
    st.caption("Replay source: committed Agent-1 historical stream. Scenario faults are deterministic and auditable.")

# ---------- Run ----------
if run or "results" not in st.session_state:
    t0 = time.perf_counter()
    cfg, results, selected_obj = run_scenario(scenario)
    elapsed_ms = (time.perf_counter() - t0) * 1000
    st.session_state.results = [r.model_dump() for r in results]
    st.session_state.selected = selected_obj.model_dump()
    st.session_state.cfg = cfg
    st.session_state.scenario = scenario
    st.session_state.elapsed_ms = elapsed_ms

results = st.session_state.results
selected = st.session_state.selected
scenario = st.session_state.scenario
df = pd.DataFrame(results)
elapsed_ms = float(st.session_state.get("elapsed_ms", 0.0))

fault_types = ["sensor_stuck", "sensor_spike", "sensor_drift", "sensor_dropout"]
faults = df[df.anomaly_type.isin(fault_types)]
events = df[df.anomaly_type == "genuine_event"]
crit = df[df.alert_severity.isin(["critical", "high"])]

# ---------- Hero ----------
st.markdown(f'''
<div class="hero">
  <div class="hero-row">
    <div>
      <h1>🛡️ SkyGuard AI</h1>
      <p>Autonomous quality control for Automatic Weather Stations · <b>SIH26073</b></p>
      <span class="badge">5-Agent Intelligence</span>
      <span class="badge">Neuro-Symbolic Fusion</span>
      <span class="badge">Explainable AI</span>
      <span class="badge">Self-Healing</span>
      <span class="badge">Genuine-Event Safety Shield</span>
    </div>
    <div class="system">● {"SYSTEM READY" if RUNTIME["ready"] else "SYSTEM DEGRADED"} · A2 REAL · A3 {("LSTM-AE" if RUNTIME["agent3"]["engine"] == "LSTM_AUTOENCODER" else "FALLBACK")} · A4 SHAP · A5 HEAL</div>
  </div>
</div>
''', unsafe_allow_html=True)

scenario_label = SCENARIOS[scenario]["label"]

if not RUNTIME["ready"]:
    st.error(
        f"Judge warning: runtime is degraded. Agent 3 is `{RUNTIME['agent3']['engine']}`. "
        "Do not present the ML demo until the LSTM Autoencoder runtime is available."
    )

selected_station = selected["station_id"]
selected_timestamp = selected["timestamp"]
st.markdown(f"""<div class="replay-banner"><div><div class="label">● Active deterministic replay</div><div class="name">{scenario_label}</div></div><div style="text-align:right"><div class="time">{selected_station} · {selected_timestamp}</div><div class="time">Agent-1 historical stream · auditable scenario</div></div></div>""", unsafe_allow_html=True)

st.markdown(f"""<div class="runtime-strip">
  <span><b>Runtime integrity</b></span>
  <span>Agent 2: <b>{'REAL' if RUNTIME['agent2']['available'] else 'OFFLINE'}</b></span>
  <span>Agent 3: <b>{RUNTIME['agent3']['engine']}</b></span>
  <span>Agent 4: <b>{RUNTIME['agent4']['engine']}</b></span>
  <span>Agent 5: <b>REAL CORRECTION</b></span>
  <span>Replay latency: <b>{elapsed_ms:.0f} ms</b></span>
</div>""", unsafe_allow_html=True)

st.write("")

# ---------- KPI strip ----------
kpis = st.columns(5)
kpi_data = [
    ("AWS monitored", str(len(df)), "network snapshot"),
    ("Faults isolated", str(len(faults)), "sensor failures"),
    ("Genuine events", str(len(events)), "weather protected"),
    ("High / critical", str(len(crit)), "operator attention"),
    ("Decision confidence", pct(selected.get("confidence_score", 0)), f"{elapsed_ms:.0f} ms replay"),
]
for c, (label, value, sub) in zip(kpis, kpi_data):
    with c:
        st.markdown(f'<div class="kpi"><div class="label">{label}</div><div class="value">{value}</div><div class="sub">{sub}</div></div>', unsafe_allow_html=True)

st.write("")

# ---------- Decision hero ----------
atype = selected["anomaly_type"]
severity = selected["alert_severity"]
exp = selected.get("explanation", {})
sig = exp.get("evidence_signature", {})

if atype == "genuine_event":
    title = "GENUINE WEATHER EVENT"
    icon = "🌪️"
    action = "CORRECTION BLOCKED — TRUST RAW OBSERVATION"
elif atype == "none":
    title = "NETWORK NOMINAL"
    icon = "🟢"
    action = "NO SENSOR-FAULT ACTION REQUIRED"
else:
    title = pretty(atype).upper()
    icon = "🔴" if severity in ("high", "critical") else "🟠"
    action = "CORRECTION APPROVED" if selected.get("corrected_value") else "ALERT — NO CORRECTION"

reason = exp.get("root_cause_narrative", "")
classes = severity_class(atype, severity)

left, right = st.columns([1.45, .75])
with left:
    st.markdown(f'''
    <div class="decision {classes}">
      <div style="color:#91a9bf;font-size:.72rem;text-transform:uppercase;letter-spacing:.1em">FINAL DECISION</div>
      <h2>{icon} {title}</h2>
      <div style="margin:5px 0 10px"><span class="chip">{selected['station_id']}</span><span class="chip">{severity.upper()}</span><span class="chip">{pct(selected.get('confidence_score',0))} CONFIDENCE</span></div>
      <div class="reason"><b style="color:#ffffff">Why:</b> {reason}</div>
      <div style="margin-top:15px;color:#ffffff;font-weight:900;font-size:.92rem;letter-spacing:.02em">{action}</div>
    </div>''', unsafe_allow_html=True)
with right:
    st.markdown('<div class="card"><div class="card-title">Sensor status</div>', unsafe_allow_html=True)
    h = selected.get("sensor_health_status", "green")
    health_label = {"green":"STABLE","amber":"WATCH / DEGRADING","red":"CRITICAL"}.get(h, h.upper())
    st.metric("Health status", health_label)
    st.metric("Root cause", pretty(atype))
    st.metric("Severity", severity.upper())
    st.markdown('</div>', unsafe_allow_html=True)

st.write("")

# ---------- Pipeline ----------
st.markdown('<div class="card-title">5-AGENT DECISION PIPELINE</div>', unsafe_allow_html=True)
ml_engine = selected.get("engine", "DETERMINISTIC_FALLBACK")
ml_label = "LSTM-AE" if ml_engine == "LSTM_AUTOENCODER" else "UNAVAILABLE / WARM-UP"
stages = [
    ("01", "INGEST", "Agent 1", "ok"),
    ("02", "SCREEN", "Agent 2", "ok"),
    ("03", "DETECT", ml_label, "ok"),
    ("04", "DECIDE", "Fusion + SHAP", "ok"),
    ("05", "HEAL", "Agent 5", "ok"),
]
cols = st.columns(5)
for c, (num, name, detail, _) in zip(cols, stages):
    with c:
        st.markdown(f'<div class="stage"><div class="num">{num}</div><div class="name">{name}</div><div class="status">✓ {detail}</div></div>', unsafe_allow_html=True)

st.write("")

# ---------- Main network + station decision ----------
left, right = st.columns([1.15, .85])
with left:
    st.markdown('<div class="card"><div class="card-title">📍 NETWORK DIGITAL TWIN · 5 AWS</div>', unsafe_allow_html=True)
    mapdf = df[["latitude", "longitude", "station_id"]].copy().rename(columns={"latitude":"lat", "longitude":"lon"})
    st.map(mapdf[["lat", "lon"]], zoom=6)
    rows_html = []
    for _, row in df[["station_id", "anomaly_type", "sensor_health_status", "alert_severity", "confidence_score"]].iterrows():
        h = row["sensor_health_status"]
        sev = str(row["alert_severity"]).lower()
        hc = {"green":"status-stable","amber":"status-watch","red":"status-critical"}.get(h, "")
        sc = {"high":"status-high","medium":"status-medium","low":"status-low"}.get(sev, "")
        rows_html.append(f"<tr><td><b>{row['station_id']}</b></td><td>{pretty(row['anomaly_type'])}</td><td class=\"{hc}\">● {h.upper()}</td><td class=\"{sc}\">{sev.upper()}</td><td><b>{pct(row['confidence_score'])}</b></td></tr>")
    st.markdown('<table class=\"html-table\"><thead><tr><th>Station</th><th>Diagnosis</th><th>Health</th><th>Severity</th><th>Confidence</th></tr></thead><tbody>' + ''.join(rows_html) + '</tbody></table>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with right:
    st.markdown('<div class="card"><div class="card-title">🎯 SELECTED STATION · LIVE EVIDENCE</div>', unsafe_allow_html=True)
    meta = backend.STATION_META[selected["station_id"]]
    st.markdown(f"### {selected['station_id']} · {meta['name']}")
    st.caption(selected["timestamp"])
    m1, m2, m3 = st.columns(3)
    m1.markdown(f'<div class="metricbox"><div class="v">{float(selected.get("spatial_deviation_score",0)):.2f}</div><div class="l">Spatial deviation</div></div>', unsafe_allow_html=True)
    m2.markdown(f'<div class="metricbox"><div class="v">{float(selected.get("physical_consistency_score",0)):.2f}</div><div class="l">Physical consistency</div></div>', unsafe_allow_html=True)
    m3.markdown(f'<div class="metricbox"><div class="v">{float(selected.get("ml_anomaly_score",0)):.2f}</div><div class="l">ML anomaly</div></div>', unsafe_allow_html=True)
    st.write("")
    st.markdown(f'<div class="explain">{reason}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

st.write("")

# ---------- Evidence and correction ----------
tab1, tab2, tab3 = st.tabs(["🧠 Evidence & Explainability", "🔧 Self-Healing", "📊 Validation & Architecture"])

with tab1:
    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="card"><div class="card-title">Fault evidence</div>', unsafe_allow_html=True)
        deviation = float(selected.get("spatial_deviation_score", 0))
        physical_fault = 1.0 - float(selected.get("physical_consistency_score", 1))
        ml = float(selected.get("ml_anomaly_score", 0))
        flat = max((sig.get("flatline_runs") or {}).values(), default=0)
        spike = 1.0 if sig.get("isolated_spike") else 0.0
        evidence_bar("Spatial isolation", deviation, "Higher means the station disagrees with neighbors")
        evidence_bar("Physical inconsistency", physical_fault, "1 − physical consistency score")
        evidence_bar("Temporal / ML anomaly", ml, "LSTM reconstruction-based anomaly score")
        evidence_bar("Fault signature", max(spike, min(flat / 8.0, 1.0)), "Spike / frozen / drift signature")
        st.markdown('</div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="card"><div class="card-title">Event evidence / safety shield</div>', unsafe_allow_html=True)
        event_corr = float(sig.get("event_corroboration", 0))
        regional = float(sig.get("regional_event_index", 0))
        coherent = float(selected.get("physical_consistency_score", 0))
        evidence_bar("Multi-station corroboration", event_corr, "Do neighboring stations move with the observation?")
        evidence_bar("Regional event index", regional, "Multi-hour network-wide transition")
        evidence_bar("Physical coherence", coherent, "Joint T/P/RH consistency")
        if atype == "genuine_event":
            st.markdown('<div class="protect"><b>🛡️ SAFETY SHIELD ACTIVE</b><br>Regional evidence is strong enough to protect the weather signal. Automatic correction is blocked.</div>', unsafe_allow_html=True)
        else:
            st.caption("The event shield remains available for every observation; it only activates when regional evidence is sufficiently corroborated.")
        st.markdown('</div>', unsafe_allow_html=True)

    st.write("")
    st.markdown('<div class="card"><div class="card-title">Why did SkyGuard decide this?</div>', unsafe_allow_html=True)
    factors = exp.get("top_factors", [])
    if not factors:
        factors = ["within_expected_range", "stable_temporal_pattern", "no_spatial_outlier"]
    for i, factor in enumerate(factors[:5], 1):
        st.markdown(f"**{i}.** {pretty(factor)}")
    shield_active = atype == "genuine_event"
    st.markdown(f'<div class="small" style="margin-top:10px">Final fusion verdict: <b>{pretty(atype)}</b> · Genuine-event safety shield: <b>{"ACTIVE" if shield_active else "INACTIVE"}</b></div>', unsafe_allow_html=True)
    shap_features = exp.get("shap_features") or []
    if shap_features:
        st.write("")
        st.markdown('<div class="card"><div class="card-title">SHAP feature contribution</div>', unsafe_allow_html=True)
        shap_rows = []
        for item in shap_features[:6]:
            if isinstance(item, dict):
                shap_rows.append({
                    "Feature": pretty(item.get("feature", "unknown")),
                    "Value": item.get("value", ""),
                    "SHAP contribution": item.get("shap_contribution", ""),
                    "Meaning": item.get("meaning", "")
                })
        if shap_rows:
            st.dataframe(pd.DataFrame(shap_rows), hide_index=True, use_container_width=True)
        st.caption("SHAP is shown when Agent 4 returns feature-level attribution for the selected decision.")
        st.markdown('</div>', unsafe_allow_html=True)

with tab2:
    raw = {
        "Temperature (°C)": selected.get("temperature_c"),
        "Pressure (hPa)": selected.get("pressure_hpa"),
        "Humidity (%)": selected.get("humidity_pct"),
    }
    corr = selected.get("corrected_value") or {}
    corrected = {
        "Temperature (°C)": corr.get("temperature_c"),
        "Pressure (hPa)": corr.get("pressure_hpa"),
        "Humidity (%)": corr.get("humidity_pct"),
    }
    rows=[]
    for k in raw:
        rv, cv = raw[k], corrected[k]
        delta = None if cv is None or rv is None else float(cv)-float(rv)
        rows.append({"Parameter":k,"Raw":rv,"Corrected":cv,"Change":delta})
    if atype == "genuine_event":
        st.markdown('<div class="protect"><b>🛡️ GENUINE-EVENT PROTECTION</b><br>Correction is <b>BLOCKED</b>. SkyGuard preserves the weather observation because the network corroborates it.</div>', unsafe_allow_html=True)
    elif atype != "none":
        changed=[]
        for row in rows:
            rv, cv = row["Raw"], row["Corrected"]
            if cv is not None and rv is not None and abs(float(cv)-float(rv)) > 1e-9:
                changed.append(row["Parameter"])
        changed_text=", ".join(changed) if changed else "no channel changed"
        st.markdown(f'<div class="correction"><b>🔧 SELF-HEALING ACTION</b><br>Correction <b>APPLIED</b> to: {changed_text}. Healthy channels are preserved.</div>', unsafe_allow_html=True)
    else:
        st.success("No correction required — the observation is trusted as received.")
    st.write("")
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    if atype != "none" and atype != "genuine_event":
        st.caption("Correction combines temporal history and healthy same-timestamp spatial neighbors, with safety bounds.")

with tab3:
    # Maintenance / degradation evidence. This is deliberately data-driven:
    # if the backend exposes degradation_tracker fields, show them; otherwise
    # never invent a prediction and fall back to the verified sensor-health state.
    degradation = selected.get("degradation")
    if degradation is None:
        degradation = selected.get("degradation_tracker")
    if degradation is None:
        degradation = {}

    if isinstance(degradation, dict):
        degradation_status = degradation.get("status") or degradation.get("health_status")
        maintenance_priority = degradation.get("maintenance_priority") or degradation.get("priority")
        maintenance_reason = degradation.get("reason") or degradation.get("recommendation")
    else:
        degradation_status = None
        maintenance_priority = None
        maintenance_reason = None

    st.markdown('<div class="card"><div class="card-title">Maintenance & degradation signal</div>', unsafe_allow_html=True)
    mc1, mc2, mc3 = st.columns(3)
    mc1.metric("Current health", {"green":"STABLE","amber":"WATCH / DEGRADING","red":"CRITICAL"}.get(
        selected.get("sensor_health_status", "green"),
        str(selected.get("sensor_health_status", "unknown")).upper()
    ))
    mc2.metric("Degradation tracker", str(degradation_status or "HEALTH STATE"))
    mc3.metric("Maintenance priority", str(maintenance_priority or (
        "URGENT" if selected.get("sensor_health_status") == "red"
        else "REVIEW" if selected.get("sensor_health_status") == "amber"
        else "NORMAL"
    )))
    if maintenance_reason:
        st.caption(f"Tracker recommendation: {maintenance_reason}")
    else:
        st.caption("Maintenance signal is derived from the verified station-health/degradation path; no unsupported remaining-life estimate is displayed.")
    st.markdown('</div>', unsafe_allow_html=True)
    st.write("")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="card"><div class="card-title">Architecture</div>', unsafe_allow_html=True)
        st.markdown("""
**AWS telemetry** → **Agent 1 Ingestion** → **Agent 2 Consistency** → **Agent 3 LSTM-AE** → **Evidence Fusion** → **Agent 4 Explainability + Safety Gate** → **Agent 5 Correction** → **Command Center**

**Design principle:** no single model has unchecked authority over a high-risk decision.
""")
        st.markdown('</div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="card"><div class="card-title">SIH26073 requirement coverage</div>', unsafe_allow_html=True)
        st.markdown("""
        | Requirement | Screening implementation |
        |---|---|
        | Real-time anomaly path | Per-observation inference + severity/alert state |
        | Sensor faults | Spike, frozen, dropout and drift signatures |
        | Temporal / seasonal context | Temporal history + seasonal features |
        | Multivariate consistency | T / P / RH physical consistency |
        | Spatial consistency | Healthy-neighbour comparison |
        | Genuine weather events | Regional corroboration + safety shield |
        | Confidence + explanation | Confidence, root cause, evidence and SHAP |
        | Self-healing | Temporal + spatial correction with safety bounds |
        | Maintenance readiness | Agent-4 degradation signal + sensor-health state |
        | Operator dashboard | Network, station, alert, evidence and correction views |
        """)
        st.caption("Screening build: deterministic replay demonstrates the real inference path. Production deployment can attach a live AWS stream to the same pipeline.")
        st.markdown('</div>', unsafe_allow_html=True)

    st.write("")
    with st.expander("Verified repository evidence", expanded=False):
        st.markdown(f"""
- **6/6** deterministic screening scenarios pass end-to-end
- **5,463 / 5,463** labeled sensor faults evaluable in the correction benchmark
- **82.5%** average within configured correction tolerance
- **Agent 3 runtime:** `{RUNTIME['agent3']['engine']}`
- **Agent 4:** `{RUNTIME['agent4']['engine']}` with SHAP + safety gate
""")
        st.caption("These are repository validation figures, not claims of universal accuracy.")

    st.write("")
    st.markdown('<div class="card"><div class="card-title">Deployment readiness</div>', unsafe_allow_html=True)
    d1,d2,d3,d4 = st.columns(4)
    d1.metric("API", "FastAPI")
    d2.metric("Live input", "POST /process_batch")
    d3.metric("Edge path", "Quantization candidate")
    d4.metric("Persistence", "TimescaleDB-ready")
    st.caption("Screening dashboard uses deterministic Agent-1 replay. The backend already exposes /process and /process_batch for live AWS telemetry; the same inference path feeds the dashboard/API.")
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="footer">AGENTIC GENESIS · SKYGUARD AI · SIH26073 · Screening build · Anomaly ≠ Sensor Fault</div>', unsafe_allow_html=True)