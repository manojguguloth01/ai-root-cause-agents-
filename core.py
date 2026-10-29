"""
AI Root Cause and Decision Agents - core engine
Pipeline: Data -> Preprocessing -> Anomaly Detection -> Root Cause Agent
          -> Decision Agent -> Recommendations -> Feedback
"""
import os
import time
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

# ----------------------------------------------------------------------------
# System model: which service calls (depends on) which
# ----------------------------------------------------------------------------
SERVICES = ["web", "api", "auth", "db", "cache"]
DEPENDS_ON = {
    "web": ["api"],
    "api": ["auth", "db", "cache"],
    "auth": ["db"],
    "db": [],
    "cache": [],
}
METRICS = ["cpu", "memory", "latency_ms", "error_rate"]
T0 = 120  # minute at which the incident starts

# (service, start offset in minutes, metric, delta, log event)
SCENARIOS = {
    "DB connection pool exhaustion": [
        ("db", 0, "latency_ms", 450, "DB_CONNECTION_POOL_EXHAUSTED"),
        ("db", 0, "cpu", 35, None),
        ("auth", 3, "latency_ms", 250, "AUTH_TIMEOUT"),
        ("api", 5, "latency_ms", 400, "UPSTREAM_TIMEOUT"),
        ("api", 5, "error_rate", 8, None),
        ("web", 8, "error_rate", 10, "HTTP_503"),
    ],
    "Cache memory leak": [
        ("cache", 0, "memory", 40, "OOM_WARNING"),
        ("api", 10, "latency_ms", 220, "CACHE_MISS_SPIKE"),
        ("web", 14, "latency_ms", 180, "SLOW_RESPONSE"),
    ],
    "Auth token failure": [
        ("auth", 0, "error_rate", 25, "AUTH_TOKEN_EXPIRED"),
        ("api", 2, "error_rate", 14, "HTTP_401_UNAUTHORIZED"),
        ("web", 4, "error_rate", 9, "LOGIN_FAILURES"),
    ],
}
TRUE_ROOT = {
    "DB connection pool exhaustion": "db",
    "Cache memory leak": "cache",
    "Auth token failure": "auth",
}

# ----------------------------------------------------------------------------
# 1. Data collection (simulated logs + metrics)
# ----------------------------------------------------------------------------
def generate_data(scenario, seed=42, minutes=240):
    rng = np.random.default_rng(seed)
    start = pd.Timestamp("2026-10-30 09:00")
    frames = []
    for s in SERVICES:
        d = pd.DataFrame({
            "minute": np.arange(minutes),
            "timestamp": [start + pd.Timedelta(minutes=int(m)) for m in range(minutes)],
            "service": s,
            "cpu": rng.normal(35, 4, minutes),
            "memory": rng.normal(50, 3, minutes),
            "latency_ms": rng.normal(120, 12, minutes),
            "error_rate": np.abs(rng.normal(0.5, 0.2, minutes)),
            "event": "OK",
        })
        d.loc[rng.random(minutes) < 0.02, "event"] = "INFO_GC_PAUSE"  # harmless noise
        frames.append(d)
    df = pd.concat(frames, ignore_index=True)

    for svc, off, metric, delta, ev in SCENARIOS[scenario]:
        a, b = T0 + off, T0 + off + 45
        mask = (df.service == svc) & (df.minute >= a) & (df.minute < b)
        ramp = np.clip((df.loc[mask, "minute"] - a + 1) / 4, 0, 1)
        df.loc[mask, metric] += delta * ramp
        if ev:
            df.loc[mask, "event"] = ev
    return df


def ensure_minute(df):
    """Make an uploaded CSV compatible with the pipeline."""
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    if "minute" not in df.columns:
        df["minute"] = ((df["timestamp"] - df["timestamp"].min()).dt.total_seconds() // 60).astype(int)
    return df


# ----------------------------------------------------------------------------
# 2. Preprocessing + 3. AI-based anomaly detection
# ----------------------------------------------------------------------------
def preprocess(df):
    df = df.drop_duplicates().dropna(subset=METRICS + ["service"]).copy()
    return df.sort_values(["service", "minute"]).reset_index(drop=True)


def detect_anomalies(df):
    """Isolation Forest (ML) + robust z-score (statistical) per service."""
    out = []
    for s, g in preprocess(df).groupby("service"):
        g = g.copy()
        X = g[METRICS].values
        med = np.median(X, axis=0)
        mad = np.median(np.abs(X - med), axis=0) * 1.4826 + 1e-9
        z = np.abs((X - med) / mad)
        g["max_z"] = z.max(axis=1)
        g["top_metric"] = [METRICS[i] for i in z.argmax(axis=1)]
        iso = IsolationForest(n_estimators=60, contamination=0.3, random_state=0).fit(X)
        g["iso_flag"] = iso.predict(X) == -1
        g["anomaly"] = g["iso_flag"] & (g["max_z"] > 4)
        # sustained = 3 consecutive anomalous minutes (filters random spikes)
        g["sustained"] = g["anomaly"].astype(int).rolling(3).sum().fillna(0) == 3
        out.append(g)
    return pd.concat(out, ignore_index=True)


# ----------------------------------------------------------------------------
# 4. Root Cause Agent
# ----------------------------------------------------------------------------
def dependents_of(service):
    """All services that (transitively) depend on `service`."""
    found, stack = set(), [service]
    while stack:
        cur = stack.pop()
        for s, deps in DEPENDS_ON.items():
            if cur in deps and s not in found:
                found.add(s)
                stack.append(s)
    return found


def find_root_cause(det):
    onset = {}
    for s, g in det.groupby("service"):
        sus = g[g["sustained"]]
        if len(sus):
            onset[s] = int(sus["minute"].min()) - 2
    if not onset:
        return None

    # Candidate root = anomalous service none of whose dependencies are anomalous
    cands = [s for s in onset if not any(d in onset for d in DEPENDS_ON.get(s, []))]
    if not cands:
        cands = list(onset)

    def explained(s):
        return len([x for x in dependents_of(s) if x in onset])

    root = sorted(cands, key=lambda s: (-explained(s), onset[s]))[0]
    frac = 1.0 if len(onset) == 1 else explained(root) / (len(onset) - 1)
    confidence = round(0.6 * frac + 0.4 * (1.0 if len(cands) == 1 else 0.5), 2)

    g = det[(det.service == root) & det["anomaly"]]
    base = det[(det.service == root) & (~det["anomaly"])]
    metric = g["top_metric"].mode().iloc[0]
    events = g[~g["event"].isin(["OK", "INFO_GC_PAUSE"])]["event"]
    event = events.mode().iloc[0] if len(events) else "NONE"

    return {
        "root": root,
        "onset": onset,
        "chain": sorted(onset, key=onset.get),
        "metric": metric,
        "baseline": round(float(base[metric].median()), 1),
        "peak": round(float(g[metric].max()), 1),
        "event": event,
        "confidence": confidence,
        "start_time": det[(det.service == root) & (det.minute == onset[root])]["timestamp"].iloc[0],
    }


# ----------------------------------------------------------------------------
# 5. Decision Agent (knowledge base + learning from feedback)
# ----------------------------------------------------------------------------
KB = {
    "DB_CONNECTION_POOL_EXHAUSTED": [
        ("Increase DB connection pool size and recycle stale connections", 3),
        ("Kill long-running queries and add missing indexes", 2),
        ("Add retry with exponential backoff in the API layer", 1),
    ],
    "OOM_WARNING": [
        ("Restart cache service and set a max-memory eviction policy", 3),
        ("Profile for memory leak and roll back the latest deployment", 2),
        ("Add a memory alert at 80% threshold", 1),
    ],
    "AUTH_TOKEN_EXPIRED": [
        ("Rotate / refresh signing keys and token secrets", 3),
        ("Verify clock sync (NTP) between auth service and clients", 2),
        ("Add a token-refresh failure alert", 1),
    ],
}
DEFAULT_KB = {
    "latency_ms": [("Scale out the service and inspect slow dependencies", 3), ("Review recent deployments", 2)],
    "cpu": [("Scale horizontally and profile hot code paths", 3), ("Check for runaway jobs", 2)],
    "memory": [("Restart service and check for memory leaks", 3), ("Increase memory limit temporarily", 2)],
    "error_rate": [("Roll back the latest change and inspect error logs", 3), ("Enable circuit breaker", 2)],
}
FEEDBACK_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "feedback.csv")


def load_feedback():
    if os.path.exists(FEEDBACK_FILE):
        return pd.read_csv(FEEDBACK_FILE)
    return pd.DataFrame(columns=["event", "action", "helpful"])


def save_feedback(event, action, helpful):
    fb = pd.concat([load_feedback(), pd.DataFrame([{"event": event, "action": action, "helpful": int(helpful)}])])
    fb.to_csv(FEEDBACK_FILE, index=False)


def decide(rc):
    options = KB.get(rc["event"]) or DEFAULT_KB[rc["metric"]]
    fb = load_feedback()
    rows = []
    for action, base in options:
        f = fb[(fb.event == rc["event"]) & (fb.action == action)]
        learned = 0.5 * (int((f.helpful == 1).sum()) - int((f.helpful == 0).sum()))
        rows.append({"action": action, "base": base, "learned_from_feedback": learned, "score": base + learned})
    out = pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)
    out["priority"] = ["Immediate"] + ["Next"] * (len(out) - 1) if len(out) else []
    return out


# ----------------------------------------------------------------------------
# 6. Explanation (template; optional LLM)
# ----------------------------------------------------------------------------
def explain(rc, decisions, api_key=None):
    chain = " -> ".join(f"{s} (t+{rc['onset'][s] - rc['onset'][rc['chain'][0]]} min)" for s in rc["chain"])
    base_text = (
        f"The root cause is the '{rc['root']}' service. Its {rc['metric']} rose from a baseline of "
        f"{rc['baseline']} to {rc['peak']}, with log event {rc['event']}. Failures then propagated: {chain}. "
        f"Recommended first action: {decisions.iloc[0]['action']}. Confidence: {rc['confidence']}."
    )
    if not api_key:
        return base_text
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)
        msg = client.messages.create(
            model="claude-sonnet-5-5",
            max_tokens=400,
            messages=[{"role": "user", "content":
                       "You are an SRE assistant. Explain this incident for a non-expert in 4 sentences, "
                       "using only these facts:\n" + base_text}],
        )
        return msg.content[0].text
    except Exception as e:  # never break the demo
        return base_text + f"\n\n(LLM unavailable, showing built-in explanation: {type(e).__name__})"


# ----------------------------------------------------------------------------
# 7. Evaluation (synopsis section 9: accuracy, detection rate, response time)
# ----------------------------------------------------------------------------
def evaluate(runs_per_scenario=5):
    rows = []
    for scn, truth in TRUE_ROOT.items():
        for seed in range(runs_per_scenario):
            t = time.time()
            det = detect_anomalies(generate_data(scn, seed=seed))
            rc = find_root_cause(det)
            dt = time.time() - t
            rows.append({
                "scenario": scn, "seed": seed, "true_root": truth,
                "predicted_root": rc["root"] if rc else None,
                "correct": bool(rc and rc["root"] == truth),
                "response_time_s": round(dt, 2),
            })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    res = evaluate(5)
    print(res.groupby("scenario")[["correct", "response_time_s"]].mean())
    print("Overall accuracy:", res["correct"].mean())
