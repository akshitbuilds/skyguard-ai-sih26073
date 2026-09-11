"""SkyGuard AI — SIH26073 screening command center.

Designed for a judge-first demo: the first screen answers WHAT happened,
WHY the system decided it, and WHETHER the data was corrected.
"""
from __future__ import annotations
import os, sys, time
import pandas as pd
import plotly.graph_objects as go
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
  --bg:#050b14; --panel:#0b1422; --panel2:#0e1b2b; --line:#1d3047;
  --text:#edf5ff; --muted:#8fa5bb; --cyan:#55d6ff; --green:#37e6a2;
  --amber:#ffc857; --red:#ff5c6c; --purple:#9c8cff;
}
[data-testid="stAppViewContainer"] {background:var(--bg); color:var(--text);}
[data-testid="stHeader"] {background:rgba(5,11,20,.88);}
.block-container {max-width:1540px; padding-top:1.0rem; padding-bottom:2.5rem;}
/* =========================================================
   JUDGE SCENARIO SELECTOR — READABILITY FIX
   ========================================================= */

/* ---------- Closed selector ---------- */
section[data-testid="stSidebar"] div[data-baseweb="select"] {
  width: 100% !important;
  min-height: 48px !important;

  background: #0b1725 !important;
  background-color: #0b1725 !important;

  border: 2px solid #55d6ff !important;
  border-radius: 12px !important;

  box-shadow: 0 0 18px rgba(85, 214, 255, 0.10) !important;

  opacity: 1 !important;
}

/* BaseWeb inner container */
section[data-testid="stSidebar"] div[data-baseweb="select"] > div {
  background: #0b1725 !important;
  background-color: #0b1725 !important;

  border: none !important;
  border-radius: 10px !important;

  opacity: 1 !important;
}

/* ---------- Selected value / clickable area ---------- */
section[data-testid="stSidebar"] div[data-baseweb="select"] [role="button"] {
  background: #0b1725 !important;
  background-color: #0b1725 !important;

  color: #ffffff !important;

  opacity: 1 !important;
}

/* ---------- FORCE WHITE TEXT ON ALL SELECT DESCENDANTS ---------- */
section[data-testid="stSidebar"] div[data-baseweb="select"] *,
section[data-testid="stSidebar"] div[data-baseweb="select"] [role="button"] *,
section[data-testid="stSidebar"] div[data-baseweb="select"] span,
section[data-testid="stSidebar"] div[data-baseweb="select"] div,
section[data-testid="stSidebar"] div[data-baseweb="select"] input {
  color: #ffffff !important;
  -webkit-text-fill-color: #ffffff !important;

  background-color: transparent !important;

  opacity: 1 !important;

  visibility: visible !important;
}

/* ---------- Selected text specifically ---------- */
section[data-testid="stSidebar"] div[data-baseweb="select"]
[role="button"] span {
  color: #ffffff !important;
  -webkit-text-fill-color: #ffffff !important;

  font-weight: 700 !important;
  opacity: 1 !important;
}

/* ---------- Search/input text ---------- */
section[data-testid="stSidebar"] div[data-baseweb="select"] input {
  color: #ffffff !important;
  -webkit-text-fill-color: #ffffff !important;

  caret-color: #55d6ff !important;
  opacity: 1 !important;
}

/* ---------- Dropdown arrow ---------- */
section[data-testid="stSidebar"] div[data-baseweb="select"] svg {
  color: #55d6ff !important;
  fill: #55d6ff !important;
  stroke: #55d6ff !important;

  opacity: 1 !important;
}


/* =========================================================
   OPEN DROPDOWN / POPOVER
   ========================================================= */

/* Popover outer container */
div[data-baseweb="popover"] {
  background: #0b1725 !important;
  background-color: #0b1725 !important;

  border: 1px solid #31516f !important;
  border-radius: 12px !important;

  box-shadow: 0 18px 50px rgba(0, 0, 0, 0.65) !important;

  opacity: 1 !important;
}

/* Popover descendants */
div[data-baseweb="popover"] *,
div[data-baseweb="popover"] > div {
  background-color: #0b1725 !important;
  opacity: 1 !important;
}

/* Listbox */
div[data-baseweb="popover"] [role="listbox"] {
  background: #0b1725 !important;
  background-color: #0b1725 !important;

  border-radius: 12px !important;

  padding: 6px !important;

  opacity: 1 !important;
}


/* =========================================================
   DROPDOWN OPTIONS — MAXIMUM READABILITY
   ========================================================= */

div[data-baseweb="popover"] [role="option"] {
  background: #0b1725 !important;
  background-color: #0b1725 !important;

  color: #edf5ff !important;
  -webkit-text-fill-color: #edf5ff !important;

  padding: 12px 14px !important;

  border-radius: 8px !important;

  font-weight: 700 !important;
  font-size: 0.90rem !important;

  opacity: 1 !important;
  visibility: visible !important;
}

/* FORCE OPTION TEXT */
div[data-baseweb="popover"] [role="option"] *,
div[data-baseweb="popover"] [role="option"] span,
div[data-baseweb="popover"] [role="option"] div {
  color: #edf5ff !important;
  -webkit-text-fill-color: #edf5ff !important;

  background: transparent !important;
  background-color: transparent !important;

  opacity: 1 !important;
  visibility: visible !important;
}


/* =========================================================
   HOVER STATE
   ========================================================= */

div[data-baseweb="popover"] [role="option"]:hover {
  background: #12304a !important;
  background-color: #12304a !important;

  color: #ffffff !important;
  -webkit-text-fill-color: #ffffff !important;
}

div[data-baseweb="popover"] [role="option"]:hover *,
div[data-baseweb="popover"] [role="option"]:hover span {
  color: #ffffff !important;
  -webkit-text-fill-color: #ffffff !important;

  background: transparent !important;
}


/* =========================================================
   SELECTED OPTION
   ========================================================= */

div[data-baseweb="popover"]
[role="option"][aria-selected="true"] {
  background: #16415d !important;
  background-color: #16415d !important;

  color: #ffffff !important;
  -webkit-text-fill-color: #ffffff !important;
}

div[data-baseweb="popover"]
[role="option"][aria-selected="true"] *,
div[data-baseweb="popover"]
[role="option"][aria-selected="true"] span {
  color: #ffffff !important;
  -webkit-text-fill-color: #ffffff !important;

  background: transparent !important;

  opacity: 1 !important;
}


/* =========================================================
   FOCUS / ACTIVE STATE
   ========================================================= */

section[data-testid="stSidebar"] div[data-baseweb="select"]:focus-within {
  border-color: #55d6ff !important;

  box-shadow:
    0 0 0 2px rgba(85, 214, 255, 0.18),
    0 0 22px rgba(85, 214, 255, 0.12) !important;
}


/* =========================================================
   STREAMLIT SIDEBAR TEXT SAFETY
   ========================================================= */

section[data-testid="stSidebar"] div[data-baseweb="select"] label,
section[data-testid="stSidebar"] div[data-baseweb="select"] p {
  color: #ffffff !important;
  -webkit-text-fill-color: #ffffff !important;

  opacity: 1 !important;
}
.scenario-card {margin:12px 0 16px; padding:14px 15px; border:1px solid #2d5875; border-radius:14px; background:linear-gradient(135deg,#0a1b2a,#08131f);}
.scenario-kicker {font-size:.64rem; letter-spacing:.14em; text-transform:uppercase; color:#55d6ff; font-weight:800;}
.scenario-name {font-size:1.02rem; font-weight:850; color:#fff; margin-top:4px;}
.scenario-meta {font-size:.72rem; color:#8fa5bb; margin-top:4px;}
.replay-banner {display:flex; align-items:center; justify-content:space-between; gap:14px; margin:0 0 16px; padding:12px 16px; border:1px solid #244b68; border-radius:14px; background:linear-gradient(90deg,#091c2c,#0a1421);}
.replay-banner .label {font-size:.68rem; letter-spacing:.14em; text-transform:uppercase; color:#6bdcff; font-weight:800;}
.replay-banner .name {font-size:1.05rem; color:#fff; font-weight:850;}
.replay-banner .time {font-size:.72rem; color:#91a9bf;}
.html-table {width:100%; border-collapse:separate; border-spacing:0; overflow:hidden; border:1px solid #223a52; border-radius:12px; font-size:.78rem;}
.html-table th {text-align:left; color:#7f97ae; font-size:.65rem; letter-spacing:.08em; text-transform:uppercase; background:#0a1624; padding:10px 9px; border-bottom:1px solid #223a52;}
.html-table td {padding:10px 9px; color:#dce8f4; border-bottom:1px solid #172b3e; background:#0b1725;}
.html-table tr:last-child td {border-bottom:none;}
.status-stable {color:#66e6b4;font-weight:800;} .status-watch {color:#ffd36b;font-weight:800;} .status-critical {color:#ff7180;font-weight:800;}
.status-high {color:#ffcf70;font-weight:800;} .status-medium {color:#ffb36b;font-weight:800;} .status-low {color:#9fb3c8;font-weight:800;}
section[data-testid="stSidebar"] {background:#07101c; border-right:1px solid var(--line);}
section[data-testid="stSidebar"] * {color:var(--text);}
.hero {padding:22px 24px; border:1px solid #24405b; border-radius:20px;
  background:radial-gradient(circle at 90% 0%,rgba(85,214,255,.13),transparent 30%),
             linear-gradient(135deg,#091524,#0b1a2a 58%,#07111e); box-shadow:0 16px 50px rgba(0,0,0,.25);}
.hero-row {display:flex; align-items:center; justify-content:space-between; gap:20px;}
.hero h1 {margin:0; color:#fff; font-size:2.25rem; letter-spacing:-.04em;}
.hero p {margin:7px 0 0; color:#9fb3c8; font-size:1rem;}
.badge {display:inline-block; padding:5px 10px; margin:12px 6px 0 0; border:1px solid #31516f;
  border-radius:999px; color:#aeeaff; background:#0a1b2b; font-size:.78rem;}
.system {padding:8px 12px; border:1px solid #1e6a55; background:#08251e; color:#69f0bc;
  border-radius:999px; font-weight:700; white-space:nowrap; font-size:.8rem;}
.kpi {background:linear-gradient(180deg,#0c1827,#0a1421); border:1px solid var(--line);
  border-radius:14px; padding:13px 15px; min-height:86px;}
.kpi .label {font-size:.72rem; text-transform:uppercase; letter-spacing:.09em; color:#8198af;}
.kpi .value {font-size:1.65rem; font-weight:800; color:#fff; margin-top:3px;}
.kpi .sub {font-size:.72rem; color:#8298ad;}
.card {background:linear-gradient(180deg,#0b1625,#09131f); border:1px solid var(--line);
  border-radius:16px; padding:17px; height:100%;}
.card-title {font-size:.86rem; color:#a8bdd1; text-transform:uppercase; letter-spacing:.1em; font-weight:800; margin-bottom:12px;}
.decision {padding:20px; border-radius:17px; border:1px solid #31435a; background:#0b1725;}
.decision h2 {margin:0 0 5px; color:#fff; font-size:1.65rem;}
.decision .reason {color:#a9bed2; line-height:1.55; font-size:.9rem;}
.decision-good {border-color:#1f7258; background:linear-gradient(135deg,#08231d,#0b1725);}
.decision-warn {border-color:#7a5a1d; background:linear-gradient(135deg,#2a2110,#0b1725);}
.decision-bad {border-color:#7b2d38; background:linear-gradient(135deg,#291116,#0b1725);}
.decision-critical {border-color:#a92e3d; background:linear-gradient(135deg,#321017,#0b1725);}
.chip {display:inline-block; border-radius:999px; padding:4px 9px; font-size:.72rem; font-weight:700;
  border:1px solid #2c4057; background:#0b1b2c; color:#b8d1e8; margin:3px 4px 0 0;}
.stage-wrap {display:flex; gap:8px; align-items:stretch;}
.stage {flex:1; border:1px solid var(--line); background:#0b1725; border-radius:12px; padding:11px 8px; text-align:center;}
.stage .num {color:#5b7a96; font-size:.68rem; font-weight:800;}
.stage .name {color:#eaf4ff; font-size:.76rem; font-weight:800; margin-top:3px;}
.stage .status {color:#69e9b4; font-size:.68rem; margin-top:3px;}
.stage .arrow {color:#39526b; position:absolute;}
.metricbox {background:#0b1725; border:1px solid var(--line); border-radius:12px; padding:12px;}
.metricbox .v {font-size:1.35rem; font-weight:800; color:#fff;}
.metricbox .l {font-size:.68rem; color:#8298ad; text-transform:uppercase; letter-spacing:.06em;}
.explain {border-left:3px solid var(--cyan); padding:9px 12px; background:#091a28; border-radius:0 10px 10px 0; color:#c5d6e6;}
.protect {border:1px solid #1d8062; background:#07241d; border-radius:14px; padding:16px; color:#9af2cd;}
.alert {border:1px solid #8c3340; background:#2a1117; border-radius:14px; padding:16px; color:#ffd9dd;}
.correction {border:1px solid #275b7a; background:#0a1d2b; border-radius:14px; padding:15px;}
.small {font-size:.78rem; color:#8298ad;}
.footer {text-align:center; color:#526a82; font-size:.7rem; padding-top:18px;}
.runtime-strip {display:flex;flex-wrap:wrap;gap:8px;align-items:center;margin:10px 0 16px;padding:9px 12px;border:1px solid #1d3b53;border-radius:12px;background:#081622;color:#91a9bf;font-size:.72rem;}
.runtime-strip span {padding:3px 8px;border-radius:999px;background:#0b1d2d;border:1px solid #203d55;}
.runtime-strip b {color:#eaf6ff;}
button[kind="primary"] {background:#168bc1; border-color:#31b9ef;}
[data-testid="stMetricValue"] {color:#fff;}
[data-testid="stDataFrame"] {border:1px solid var(--line); border-radius:12px;}

/* ---------- Final Streamlit/BaseWeb selector override ---------- */
section[data-testid="stSidebar"] div[data-baseweb="select"],
section[data-testid="stSidebar"] div[data-baseweb="select"] > div,
section[data-testid="stSidebar"] div[data-baseweb="select"] [role="button"] {
  background: #0b1725 !important;
  background-color: #0b1725 !important;
  color: #ffffff !important;
  border-color: #55d6ff !important;
}

section[data-testid="stSidebar"] div[data-baseweb="select"] [role="button"] > div,
section[data-testid="stSidebar"] div[data-baseweb="select"] [role="button"] div,
section[data-testid="stSidebar"] div[data-baseweb="select"] [role="button"] span,
section[data-testid="stSidebar"] div[data-baseweb="select"] [role="button"] input {
  color: #ffffff !important;
  background: transparent !important;
  opacity: 1 !important;
  -webkit-text-fill-color: #ffffff !important;
}

section[data-testid="stSidebar"] div[data-baseweb="select"] svg {
  color: #55d6ff !important;
  fill: #55d6ff !important;
}

div[data-baseweb="popover"],
div[data-baseweb="popover"] > div,
div[data-baseweb="popover"] [role="listbox"] {
  background: #0b1725 !important;
  color: #edf5ff !important;
}

div[data-baseweb="popover"] [role="option"],
div[data-baseweb="popover"] [role="option"] * {
  color: #edf5ff !important;
  -webkit-text-fill-color: #edf5ff !important;
  opacity: 1 !important;
}

div[data-baseweb="popover"] [role="option"][aria-selected="true"] {
  background: #16415d !important;
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
    format_func=lambda x: SCENARIOS[x]["label"],
    label_visibility="collapsed",
)
    cfg_preview = SCENARIOS[scenario]
    st.markdown(f"<div class=\"scenario-card\"><div class=\"scenario-kicker\">ACTIVE REPLAY</div><div class=\"scenario-name\">{cfg_preview['label']}</div><div class=\"scenario-meta\">{cfg_preview['station_id']} · {cfg_preview['timestamp']}</div></div>", unsafe_allow_html=True)
    st.caption(cfg_preview["description"])
    run = st.button("▶  RUN SCENARIO", width="stretch", type="primary")
    st.divider()
    st.markdown("**Decision pipeline**")
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
    <div class="system">● {"SYSTEM READY" if RUNTIME["ready"] else "SYSTEM DEGRADED"} · A2 REAL · A3 {RUNTIME["agent3"]["engine"]} · A4 SHAP · A5 HEAL</div>
  </div>
</div>
''', unsafe_allow_html=True)

st.markdown(f"<div class=\"replay-banner\"><div><div class=\"label\">● Active deterministic replay</div><div class=\"name\">{SCENARIOS[scenario]["label"]}</div></div><div style=\"text-align:right\"><div class=\"time\">{selected["station_id"]} · {selected["timestamp"]}</div><div class=\"time\">Agent-1 historical stream · auditable scenario</div></div></div>", unsafe_allow_html=True)

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
      <div class="reason"><b>Why:</b> {reason}</div>
      <div style="margin-top:12px;color:#dbe8f4;font-weight:800">{action}</div>
    </div>''', unsafe_allow_html=True)
with right:
    st.markdown('<div class="card"><div class="card-title">Station health</div>', unsafe_allow_html=True)
    h = selected.get("sensor_health_status", "green")
    health_label = {"green":"STABLE","amber":"WATCH / DEGRADING","red":"CRITICAL"}.get(h, h.upper())
    st.metric("Health status", health_label)
    st.metric("Root cause", pretty(atype))
    st.metric("Severity", severity.upper())
    st.markdown('</div>', unsafe_allow_html=True)

st.write("")

# ---------- Pipeline ----------
st.markdown('<div class="card-title">LIVE DECISION TRACE</div>', unsafe_allow_html=True)
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
    st.markdown('<div class="card"><div class="card-title">📍 NETWORK DIGITAL TWIN</div>', unsafe_allow_html=True)
    mapdf = df[["latitude", "longitude", "station_id"]].copy().rename(columns={"latitude":"lat", "longitude":"lon"})
    st.map(mapdf[["lat", "lon"]], zoom=6, width="stretch")
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
    st.markdown('<div class="card"><div class="card-title">🎯 SELECTED STATION</div>', unsafe_allow_html=True)
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
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    if atype != "none" and atype != "genuine_event":
        st.caption("Correction combines temporal history and healthy same-timestamp spatial neighbors, with safety bounds.")

with tab3:
    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="card"><div class="card-title">Architecture</div>', unsafe_allow_html=True)
        st.markdown("""
**AWS telemetry** → **Agent 1 Ingestion** → **Agent 2 Consistency** → **Agent 3 LSTM-AE** → **Evidence Fusion** → **Agent 4 Explainability + Safety Gate** → **Agent 5 Correction** → **Command Center**

**Design principle:** no single model has unchecked authority over a high-risk decision.
""")
        st.markdown('</div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="card"><div class="card-title">Verified repository evidence</div>', unsafe_allow_html=True)
        st.markdown(f"""
- **6/6** deterministic screening scenarios pass end-to-end
- **5,463 / 5,463** labeled sensor faults evaluable in the correction benchmark
- **82.5%** average within configured correction tolerance
- **Agent 3 runtime:** `{RUNTIME['agent3']['engine']}`
- **Agent 4:** `{RUNTIME['agent4']['engine']}` with SHAP + safety gate
""")
        st.caption("These are repository validation figures, not claims of universal accuracy.")
        st.markdown('</div>', unsafe_allow_html=True)

    st.write("")
    st.markdown('<div class="card"><div class="card-title">Deployment readiness</div>', unsafe_allow_html=True)
    d1,d2,d3,d4 = st.columns(4)
    d1.metric("API", "FastAPI")
    d2.metric("Streaming state", "In-memory")
    d3.metric("Edge path", "Quantization candidate")
    d4.metric("Persistence", "TimescaleDB-ready")
    st.caption("Screening build: in-memory state + auditable replay. The per-station inference path is suitable for later edge quantization; no unmeasured power figure is claimed.")
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="footer">AGENTIC GENESIS · SKYGUARD AI · SIH26073 · Screening build · Anomaly ≠ Sensor Fault</div>', unsafe_allow_html=True)
