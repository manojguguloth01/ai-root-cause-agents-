import pandas as pd
import altair as alt
import streamlit as st
from core import (SERVICES, DEPENDS_ON, SCENARIOS, generate_data, ensure_minute,
                  detect_anomalies, find_root_cause, decide, explain,
                  save_feedback, load_feedback, evaluate)

st.set_page_config(page_title="AI Root Cause & Decision Agents", page_icon="🧠", layout="wide")

CSS = """
<style>
#MainMenu, footer {visibility: hidden;}
header[data-testid="stHeader"] {background: transparent;}
.block-container {padding-top: 2.6rem; padding-bottom: 2rem; max-width: 1280px;}
.hero {background: linear-gradient(120deg,#4f46e5 0%,#7c3aed 55%,#db2777 100%); border-radius: 20px;
       padding: 30px 36px; margin-bottom: 18px; box-shadow: 0 12px 30px rgba(79,70,229,.28);}
.hero h1 {margin: 0; font-size: 2.2rem; font-weight: 800; color: #ffffff; letter-spacing: -.01em;}
.hero p {margin: 8px 0 0; font-size: 1.05rem; color: #ede9fe;}
.about {background: #ffffff; border: 1px solid #e5e7eb; border-radius: 16px; padding: 16px 18px; height: 100%;
        box-shadow: 0 2px 10px rgba(0,0,0,.04);}
.about .ic {font-size: 1.5rem;}
.about h4 {margin: 6px 0 4px; color: #111827; font-size: 1rem;}
.about p {margin: 0; color: #4b5563; font-size: .9rem; line-height: 1.45;}
.pipe {display: flex; gap: 8px; flex-wrap: wrap; margin: 14px 0 6px;}
.pipe .st {flex: 1; min-width: 130px; text-align: center; background: #eef2ff; color: #3730a3; border-radius: 12px;
           padding: 9px 6px; font-size: .82rem; font-weight: 600; border: 1px solid #c7d2fe;}
.kpi {background: #ffffff; border: 1px solid #e5e7eb; border-radius: 16px; padding: 14px 16px;
      box-shadow: 0 2px 10px rgba(0,0,0,.05);}
.kpi-l {color: #6b7280; font-size: .78rem; font-weight: 600; text-transform: uppercase; letter-spacing: .05em;}
.kpi-v {color: #111827; font-size: 1.8rem; font-weight: 800; line-height: 1.2;}
.kpi-s {color: #6b7280; font-size: .8rem;}
.explain {background: #eff6ff; border-left: 5px solid #3b82f6; border-radius: 10px; padding: 12px 16px;
          color: #1e3a8a; font-size: .92rem; line-height: 1.5; margin: 6px 0 14px;}
.explain b {color: #1e40af;}
.banner-bad {background: #fef2f2; border: 1px solid #fecaca; color: #991b1b; border-radius: 14px;
             padding: 14px 18px; font-size: 1rem; margin: 14px 0;}
.banner-ok {background: #ecfdf5; border: 1px solid #a7f3d0; color: #065f46; border-radius: 14px;
            padding: 14px 18px; font-size: 1rem; margin: 14px 0;}
.rec {background: #ffffff; border: 1px solid #e5e7eb; border-radius: 14px; padding: 12px 16px;
      box-shadow: 0 2px 8px rgba(0,0,0,.04);}
.rec-t {color: #111827; font-weight: 600; margin-top: 6px;}
.badge {color: #ffffff; font-size: .72rem; font-weight: 700; padding: 3px 10px; border-radius: 999px;}
.score {color: #6b7280; font-size: .8rem; margin-left: 8px;}
.chip {display: inline-block; background: #f3f4f6; color: #111827; border-radius: 999px; padding: 6px 14px;
       font-weight: 600; font-size: .88rem; margin: 3px 0;}
.chip.root {background: #fee2e2; color: #991b1b;}
.arrow {color: #9ca3af; margin: 0 4px;}
.story {background: #faf5ff; border: 1px solid #e9d5ff; color: #581c87; border-radius: 14px;
        padding: 16px 18px; line-height: 1.6; font-size: .98rem;}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# ---------------- small UI helpers ----------------
def kpi(label, value, sub="", color="#4f46e5"):
    return (f'<div class="kpi" style="border-top:4px solid {color}"><div class="kpi-l">{label}</div>'
            f'<div class="kpi-v">{value}</div><div class="kpi-s">{sub}</div></div>')


def explainer(title, text):
    st.markdown(f'<div class="explain"><b>{title}</b><br>{text}</div>', unsafe_allow_html=True)


def line_chart(data, metric, height=300, threshold=None):
    d = data[["minute", "service", metric]]
    base = alt.Chart(d).mark_line(strokeWidth=2.2).encode(
        x=alt.X("minute:Q", title="Minutes since start"),
        y=alt.Y(f"{metric}:Q", title=metric),
        color=alt.Color("service:N", scale=alt.Scale(scheme="category10"), title="Service"),
        tooltip=["service", "minute", metric])
    if threshold is not None:
        rule = alt.Chart(pd.DataFrame({"y": [threshold]})).mark_rule(
            color="#ef4444", strokeDash=[6, 4], strokeWidth=2).encode(y="y:Q")
        base = base + rule
    return base.properties(height=height, width="container")


# ---------------- Hero + what it does ----------------
st.markdown(
    '<div class="hero"><h1>AI Root Cause &amp; Decision Agents</h1>'
    '<p>When something breaks, this system finds <b>why</b> it broke and tells you <b>what to do next</b>.</p></div>',
    unsafe_allow_html=True)

a, b, c = st.columns(3)
a.markdown('<div class="about"><div class="ic">🚨</div><h4>The problem</h4><p>Modern systems produce thousands of '
           'logs and metrics. When one service fails, others fail too, so engineers waste hours finding the '
           'real cause by hand.</p></div>', unsafe_allow_html=True)
b.markdown('<div class="about"><div class="ic">🤖</div><h4>What the AI does</h4><p>Machine learning spots unusual '
           'behaviour. A <b>Root Cause Agent</b> separates the true origin from its side effects. A '
           '<b>Decision Agent</b> picks the best fix.</p></div>', unsafe_allow_html=True)
c.markdown('<div class="about"><div class="ic">✅</div><h4>What you get</h4><p>The failing service, the evidence, '
           'how the failure spread, ranked recommended actions, and a learning loop that improves with your '
           'feedback.</p></div>', unsafe_allow_html=True)

st.markdown(
    '<div class="pipe"><div class="st">1 · Collect data</div><div class="st">2 · Clean &amp; prepare</div>'
    '<div class="st">3 · Detect anomalies</div><div class="st">4 · Root Cause Agent</div>'
    '<div class="st">5 · Decision Agent</div><div class="st">6 · Feedback &amp; learning</div></div>',
    unsafe_allow_html=True)

# ---------------- Sidebar ----------------
with st.sidebar:
    st.markdown("### Control panel")
    mode = st.radio("Data source", ["Simulated incident", "Upload CSV"])
    if mode == "Simulated incident":
        scenario = st.selectbox("Incident scenario", list(SCENARIOS.keys()))
        seed = st.number_input("Random seed", 0, 999, 42)
        st.caption("Each scenario injects a real failure into one service and lets it spread to the others.")
    else:
        up = st.file_uploader("CSV with columns: timestamp, service, cpu, memory, latency_ms, error_rate, event",
                              type="csv")
    st.markdown("---")
    api_key = st.text_input("Anthropic API key (optional)", type="password",
                            help="If given, the explanation is written by an LLM. Without it a built-in explanation is used.")
    st.markdown("---")
    st.markdown("**How to use**\n\n1. Pick a scenario\n2. Read the result cards\n3. Open each tab to see how the AI reached it\n4. Give feedback on the recommendations")


@st.cache_data
def run_simulated(scn, sd):
    return detect_anomalies(generate_data(scn, seed=int(sd)))


if mode == "Simulated incident":
    det = run_simulated(scenario, seed)
else:
    if up is None:
        st.info("Upload a CSV in the sidebar, or switch to 'Simulated incident'.")
        st.stop()
    try:
        det = detect_anomalies(ensure_minute(pd.read_csv(up)))
    except Exception as e:
        st.error(f"Could not read this file ({type(e).__name__}). Required columns: timestamp, service, cpu, "
                 "memory, latency_ms, error_rate, event.")
        st.stop()

rc = find_root_cause(det)

# ---------------- Result banner + KPIs ----------------
if rc is None:
    st.markdown('<div class="banner-ok">✅ <b>All clear.</b> No sustained anomaly was found.</div>',
                unsafe_allow_html=True)
else:
    st.markdown(f'<div class="banner-bad">🚨 <b>Incident detected.</b> Root cause: the <b>{rc["root"]}</b> service. '
                f'It started at {rc["start_time"]:%H:%M} and spread to {len(rc["onset"]) - 1} other service(s).</div>',
                unsafe_allow_html=True)

k = st.columns(5)
k[0].markdown(kpi("Records analysed", f"{len(det):,}", "logs + metrics"), unsafe_allow_html=True)
k[1].markdown(kpi("Anomalies found", int(det["anomaly"].sum()), "abnormal readings", "#f59e0b"), unsafe_allow_html=True)
k[2].markdown(kpi("Services affected", int(det[det.sustained].service.nunique()), f"of {det.service.nunique()}", "#ef4444"),
              unsafe_allow_html=True)
k[3].markdown(kpi("Root cause", rc["root"] if rc else "none", rc["event"] if rc else "", "#7c3aed"),
              unsafe_allow_html=True)
k[4].markdown(kpi("Confidence", f"{rc['confidence'] * 100:.0f}%" if rc else "-", "how much it explains", "#10b981"),
              unsafe_allow_html=True)
st.write("")

# ---------------- Tabs ----------------
t1, t2, t3, t4, t5 = st.tabs(["📥 Data", "🔍 Anomalies", "🎯 Root Cause", "💡 Decisions", "📊 Evaluation"])

with t1:
    explainer("What this step does",
              "The system collects logs and metrics (CPU, memory, response time, error rate and log events) from every "
              "service, removes duplicates and missing values, and sorts them by time. The chart shows how each "
              "service behaved. Look for the lines that suddenly jump.")
    metric = st.selectbox("Metric to plot", ["latency_ms", "error_rate", "cpu", "memory"])
    st.altair_chart(line_chart(det, metric))
    with st.expander("View raw records"):
        st.dataframe(det[["timestamp", "service", "cpu", "memory", "latency_ms", "error_rate", "event"]].head(300))

with t2:
    explainer("What this step does",
              "An <b>Isolation Forest</b> (machine learning) learns what normal looks like for each service, and a "
              "<b>robust z-score</b> measures how far a reading is from normal. A minute is flagged only when both "
              "agree, and an incident is confirmed when it lasts <b>3 minutes in a row</b>, which filters out random "
              "blips. The dashed red line is the threshold.")
    st.altair_chart(line_chart(det, "max_z", threshold=4))
    with st.expander("View flagged records"):
        st.dataframe(det[det.anomaly][["timestamp", "service", "top_metric", "max_z", "event"]].head(200))

with t3:
    explainer("What this step does",
              "Services depend on each other, so one failure causes many alarms. The <b>Root Cause Agent</b> looks "
              "for anomalous services whose own dependencies are healthy. Those are the true origin, and the rest "
              "are symptoms. Confidence shows how much of the damage the chosen service explains.")
    if rc is None:
        st.success("No sustained anomaly found.")
    else:
        left, right = st.columns([3, 2])
        with left:
            st.markdown("**Service dependency map**")
            dot = ["digraph G {", 'rankdir=LR; bgcolor="transparent";',
                   'node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=14, '
                   'margin="0.25,0.12", color="#e5e7eb"];', 'edge [color="#9ca3af"];']
            for s in SERVICES:
                if s == rc["root"]:
                    fill, font = "#ef4444", "white"
                elif s in rc["onset"]:
                    fill, font = "#fbbf24", "#111827"
                else:
                    fill, font = "#34d399", "#064e3b"
                dot.append(f'{s} [fillcolor="{fill}", fontcolor="{font}"];')
            for s, deps in DEPENDS_ON.items():
                for d in deps:
                    dot.append(f"{s} -> {d};")
            dot.append("}")
            st.graphviz_chart("\n".join(dot))
            st.caption("🔴 root cause   🟡 affected by it   🟢 healthy   ·   an arrow means 'calls'")
        with right:
            st.markdown("**Evidence**")
            st.markdown(f"- **{rc['metric']}** jumped from **{rc['baseline']}** to **{rc['peak']}**")
            st.markdown(f"- Key log event: `{rc['event']}`")
            st.markdown(f"- First abnormal at **{rc['start_time']:%H:%M}**")
            st.markdown(f"- Confidence: **{rc['confidence'] * 100:.0f}%**")

        st.markdown("**How the failure spread**")
        t0 = rc["onset"][rc["chain"][0]]
        parts = []
        for s in rc["chain"]:
            cls = "chip root" if s == rc["root"] else "chip"
            parts.append(f'<span class="{cls}">{s} · +{rc["onset"][s] - t0} min</span>')
        st.markdown('<span class="arrow">➜</span>'.join(parts), unsafe_allow_html=True)

with t4:
    explainer("What this step does",
              "The <b>Decision Agent</b> takes the root cause and its log event, looks up known fixes, and ranks "
              "them. When you press <b>Helpful</b> or <b>Not helpful</b>, the score changes, so the system learns "
              "from experience and ranks better next time.")
    if rc is None:
        st.info("Nothing to decide.")
    else:
        dec = decide(rc)
        st.markdown("**Plain-English explanation**")
        st.markdown(f'<div class="story">{explain(rc, dec, api_key or None)}</div>', unsafe_allow_html=True)
        st.write("")
        st.markdown("**Recommended actions**")
        for i, r in dec.iterrows():
            col = "#ef4444" if r["priority"] == "Immediate" else "#6366f1"
            boost = f" · learned {r['learned_from_feedback']:+g}" if r["learned_from_feedback"] else ""
            x, y, z = st.columns([7, 1.2, 1.6])
            x.markdown(f'<div class="rec"><span class="badge" style="background:{col}">{r["priority"]}</span>'
                       f'<span class="score">score {r["score"]:g}{boost}</span>'
                       f'<div class="rec-t">{r["action"]}</div></div>', unsafe_allow_html=True)
            if y.button("👍 Helpful", key=f"y{i}"):
                save_feedback(rc["event"], r["action"], 1)
                st.rerun()
            if z.button("👎 Not helpful", key=f"n{i}"):
                save_feedback(rc["event"], r["action"], 0)
                st.rerun()
        st.caption(f"{len(load_feedback())} feedback entries learned so far.")

with t5:
    explainer("What this step does",
              "To check the system works, it is run on many simulated incidents where the true cause is already "
              "known. We measure how often it names the right service (accuracy) and how fast it answers (response "
              "time). Results come from simulated data, not production systems.")
    n = st.slider("Runs per scenario", 2, 10, 5)
    if st.button("▶ Run evaluation"):
        with st.spinner("Running the full pipeline on simulated incidents..."):
            res = evaluate(n)
        e1, e2, e3 = st.columns(3)
        e1.markdown(kpi("Root-cause accuracy", f"{res['correct'].mean() * 100:.0f}%", "correct / total", "#10b981"),
                    unsafe_allow_html=True)
        e2.markdown(kpi("Avg response time", f"{res['response_time_s'].mean():.2f} s", "per incident", "#4f46e5"),
                    unsafe_allow_html=True)
        e3.markdown(kpi("Incidents tested", len(res), f"{res['scenario'].nunique()} scenarios", "#f59e0b"),
                    unsafe_allow_html=True)
        st.write("")
        st.dataframe(res)
