"""
LogGuard - Real-Time Log Anomaly Detection Using Mealy Machines
Course: Theory of Computation (CSE 2nd Year)
Architecture: Streamlit + Pandas + Plotly + Pure Python Mealy Automaton
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import random
import time
import streamlit.components.v1 as components

# ==============================================================================
# 1. FORMAL MEALY MACHINE DEFINITION & TRANSITION TABLE
# ==============================================================================
# Formal Model: M = (Q, Sigma, Delta, delta, lambda, q0)
# Q (States):
#   S0 = Normal (0 consecutive failed logins)
#   S1 = Suspicious Stage 1 (1 failed login attempt)
#   S2 = Suspicious Stage 2 (2 failed login attempts)
#   S3 = Security Alert State (3 or more failed login attempts)
#
# Sigma (Input Alphabet):
#   LOGIN_SUCCESS, LOGIN_FAIL, LOGOUT, ACCESS_GRANTED, ACCESS_DENIED, SYSTEM_ERROR
#
# Delta (Output Alphabet):
#   NORMAL, WARNING, SECURITY_ALERT, RESET, INFO
#
# q0 = S0 (Initial State)

STATES = {
    "S0": {"name": "Normal State", "desc": "Baseline safe state (0 failed logins)", "color": "#10B981"},
    "S1": {"name": "Warning State 1", "desc": "First failed login attempt recorded", "color": "#F59E0B"},
    "S2": {"name": "Warning State 2", "desc": "Second consecutive failed login", "color": "#F97316"},
    "S3": {"name": "Security Alert", "desc": "Threshold reached: Brute force alert triggered", "color": "#EF4444"}
}

INPUT_ALPHABET = [
    "LOGIN_SUCCESS",
    "LOGIN_FAIL",
    "LOGOUT",
    "ACCESS_GRANTED",
    "ACCESS_DENIED",
    "SYSTEM_ERROR"
]

OUTPUT_ALPHABET = [
    "NORMAL",
    "WARNING",
    "SECURITY_ALERT",
    "RESET",
    "INFO"
]

# Transition Table: (Current_State, Input_Symbol) -> (Next_State, Output_Symbol)
# Total combinations: 4 states * 6 inputs = 24 total deterministic transitions
TRANSITION_TABLE = {
    # State S0: Normal baseline
    ("S0", "LOGIN_FAIL"):     ("S1", "WARNING"),
    ("S0", "LOGIN_SUCCESS"):  ("S0", "NORMAL"),
    ("S0", "LOGOUT"):         ("S0", "INFO"),
    ("S0", "ACCESS_GRANTED"): ("S0", "INFO"),
    ("S0", "ACCESS_DENIED"):  ("S0", "INFO"),
    ("S0", "SYSTEM_ERROR"):   ("S0", "INFO"),

    # State S1: One failure recorded
    ("S1", "LOGIN_FAIL"):     ("S2", "WARNING"),
    ("S1", "LOGIN_SUCCESS"):  ("S0", "NORMAL"),
    ("S1", "LOGOUT"):         ("S1", "INFO"),
    ("S1", "ACCESS_GRANTED"): ("S1", "INFO"),
    ("S1", "ACCESS_DENIED"):  ("S1", "INFO"),
    ("S1", "SYSTEM_ERROR"):   ("S1", "INFO"),

    # State S2: Two failures recorded (Near threshold)
    ("S2", "LOGIN_FAIL"):     ("S3", "SECURITY_ALERT"),
    ("S2", "LOGIN_SUCCESS"):  ("S0", "NORMAL"),
    ("S2", "LOGOUT"):         ("S2", "INFO"),
    ("S2", "ACCESS_GRANTED"): ("S2", "INFO"),
    ("S2", "ACCESS_DENIED"):  ("S2", "INFO"),
    ("S2", "SYSTEM_ERROR"):   ("S2", "INFO"),

    # State S3: Security Alert Active (Threshold met / brute-force detected)
    ("S3", "LOGIN_FAIL"):     ("S3", "SECURITY_ALERT"),
    ("S3", "LOGIN_SUCCESS"):  ("S0", "RESET"),
    ("S3", "LOGOUT"):         ("S3", "INFO"),
    ("S3", "ACCESS_GRANTED"): ("S3", "INFO"),
    ("S3", "ACCESS_DENIED"):  ("S3", "INFO"),
    ("S3", "SYSTEM_ERROR"):   ("S3", "INFO"),
}

def execute_mealy_transition(current_state: str, input_event: str):
    """
    Core Mealy Machine transition function:
    Computes delta(current_state, input_event) and lambda(current_state, input_event).
    Returns (prev_state, input_event, next_state, output_symbol).
    """
    key = (current_state, input_event)
    if key in TRANSITION_TABLE:
        next_state, output = TRANSITION_TABLE[key]
    else:
        # Graceful handling for unspecified symbols
        next_state = current_state
        output = "INFO"
    return current_state, input_event, next_state, output


# ==============================================================================
# 2. SESSION STATE MANAGEMENT & SIMULATION HELPERS
# ==============================================================================
SAMPLE_USERS = ["alex.t", "sarah.c", "david.m", "elena.r", "admin_sys", "sec_ops", "guest_user"]
SAMPLE_IPS = ["198.51.100.14", "198.51.100.42", "203.0.113.88", "203.0.113.105", "192.0.2.10", "192.0.2.45"]

def init_session_state():
    """Initializes all necessary session variables if not present."""
    if "machine_state" not in st.session_state:
        st.session_state.machine_state = "S0"
    
    if "event_history" not in st.session_state:
        # Pre-seed with realistic initial baseline trace
        st.session_state.event_history = []
        # Load sample logs if available
        try:
            sample_df = pd.read_csv("logs/sample_logs.csv")
            for idx, row in sample_df.iterrows():
                st.session_state.event_history.append({
                    "event_num": idx + 1,
                    "timestamp": row["timestamp"],
                    "username": row["username"],
                    "ip_address": row["ip_address"],
                    "input_event": row["event_type"],
                    "prev_state": "S0" if idx == 0 else "S1",
                    "next_state": row["machine_state"],
                    "output": row["generated_output"]
                })
            if st.session_state.event_history:
                st.session_state.machine_state = st.session_state.event_history[-1]["next_state"]
        except Exception:
            pass

    if "demo_scenario" not in st.session_state:
        st.session_state.demo_scenario = "scenario_2"
    if "demo_step" not in st.session_state:
        st.session_state.demo_step = 0
    if "demo_history" not in st.session_state:
        st.session_state.demo_history = []
    if "demo_state" not in st.session_state:
        st.session_state.demo_state = "S0"

def log_event_to_session(input_event: str, username: str = None, ip_address: str = None):
    """Processes an event through the Mealy Machine and stores it in session state."""
    curr_s = st.session_state.machine_state
    prev_s, inp, next_s, out = execute_mealy_transition(curr_s, input_event)
    st.session_state.machine_state = next_s

    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    evt_num = len(st.session_state.event_history) + 1
    user = username if username else random.choice(SAMPLE_USERS)
    ip = ip_address if ip_address else random.choice(SAMPLE_IPS)

    entry = {
        "event_num": evt_num,
        "timestamp": now_str,
        "username": user,
        "ip_address": ip,
        "input_event": inp,
        "prev_state": prev_s,
        "next_state": next_s,
        "output": out
    }
    st.session_state.event_history.append(entry)
    return entry

def reset_all_session():
    """Resets the Mealy machine and event history."""
    st.session_state.machine_state = "S0"
    st.session_state.event_history = []
    st.session_state.demo_step = 0
    st.session_state.demo_history = []
    st.session_state.demo_state = "S0"


# ==============================================================================
# 3. STREAMLIT PAGE CONFIG & CUSTOM CSS (SIEM DARK THEME)
# ==============================================================================
st.set_page_config(
    page_title="LogGuard | Mealy Machine SIEM",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

init_session_state()

# High-contrast, clean SIEM stylesheet (Navy / Slate / Cyan / Amber / Crimson)
st.markdown("""
<style>
    /* Dark Theme Colors */
    :root {
        --bg-main: #0B1120;
        --bg-card: #0F172A;
        --bg-surface: #1E293B;
        --border-color: #334155;
        --text-primary: #F8FAFC;
        --text-secondary: #94A3B8;
        --accent-cyan: #38BDF8;
        --color-normal: #10B981;
        --color-warning: #F59E0B;
        --color-alert: #EF4444;
        --color-reset: #38BDF8;
        --color-info: #64748B;
    }

    /* Main Container Padding */
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
        background-color: var(--bg-main);
    }

    /* Top Wordmark */
    .siem-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        background-color: var(--bg-card);
        border: 1px solid var(--border-color);
        border-radius: 8px;
        padding: 16px 24px;
        margin-bottom: 24px;
    }
    .siem-title {
        font-family: 'JetBrains Mono', 'Consolas', monospace;
        font-size: 1.5rem;
        font-weight: 700;
        letter-spacing: 1px;
        color: #F8FAFC;
        margin: 0;
    }
    .siem-subtitle {
        font-size: 0.85rem;
        color: var(--accent-cyan);
        margin: 4px 0 0 0;
        font-weight: 500;
    }
    .siem-badge {
        font-family: monospace;
        background: #1E293B;
        border: 1px solid var(--border-color);
        color: #38BDF8;
        padding: 4px 10px;
        border-radius: 4px;
        font-size: 0.78rem;
    }

    /* Metric Cards */
    .metric-card {
        background-color: var(--bg-card);
        border: 1px solid var(--border-color);
        border-radius: 8px;
        padding: 16px 20px;
        position: relative;
    }
    .metric-card-title {
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: var(--text-secondary);
        margin-bottom: 6px;
    }
    .metric-card-val {
        font-size: 1.75rem;
        font-weight: 700;
        color: var(--text-primary);
        font-family: 'JetBrains Mono', monospace;
    }
    .metric-card-sub {
        font-size: 0.75rem;
        color: var(--text-secondary);
        margin-top: 4px;
    }

    /* Status Pills */
    .badge-pill {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 4px;
        font-weight: 600;
        font-size: 0.78rem;
        font-family: monospace;
        letter-spacing: 0.5px;
    }
    .pill-normal { background: #064E3B; color: #34D399; border: 1px solid #059669; }
    .pill-warning { background: #78350F; color: #FCD34D; border: 1px solid #D97706; }
    .pill-alert { background: #7F1D1D; color: #FCA5A5; border: 1px solid #DC2626; }
    .pill-reset { background: #0C4A6E; color: #7DD3FC; border: 1px solid #0284C7; }
    .pill-info { background: #1E293B; color: #94A3B8; border: 1px solid #475569; }

    /* State Boxes */
    .state-pipeline {
        display: flex;
        gap: 12px;
        margin: 16px 0;
    }
    .state-node-box {
        flex: 1;
        background: var(--bg-surface);
        border: 1px solid var(--border-color);
        border-radius: 6px;
        padding: 12px;
        text-align: center;
        transition: all 0.2s ease;
    }
    .state-node-active {
        border-color: #38BDF8;
        background: #0C4A6E;
        box-shadow: 0 0 12px rgba(56, 189, 248, 0.3);
    }

    /* Step Pipeline */
    .scenario-step {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 10px 14px;
        border-radius: 6px;
        margin-bottom: 8px;
        background: #1E293B;
        border: 1px solid #334155;
        font-family: monospace;
    }
    .scenario-step.active {
        border-color: #38BDF8;
        background: #0f2744;
        box-shadow: 0 0 8px rgba(56, 189, 248, 0.2);
    }
    .scenario-step.done {
        opacity: 0.65;
        border-color: #10B981;
    }

    /* Theory Cards */
    .theory-box {
        background: #0F172A;
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 20px;
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)


# ==============================================================================
# 4. SIDEBAR NAVIGATION & PERSISTENT STATUS
# ==============================================================================
with st.sidebar:
    st.markdown("""
    <div style="padding: 10px 0 16px 0; border-bottom: 1px solid #334155;">
        <div style="font-family: 'JetBrains Mono', monospace; font-size: 1.4rem; font-weight: 800; color: #F8FAFC; letter-spacing: 1px;">
            🛡️ LOGGUARD
        </div>
        <div style="font-size: 0.75rem; color: #38BDF8; font-weight: 500; margin-top: 2px;">
            Mealy Automaton SIEM Engine
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<p style='font-size: 0.75rem; color: #64748B; margin: 12px 0 4px 0; text-transform: uppercase;'>Navigation</p>", unsafe_allow_html=True)
    page = st.radio(
        label="Select Module",
        options=[
            "Dashboard",
            "Live Simulation",
            "Guided Demonstration",
            "Automata Visualizer",
            "Log Explorer",
            "Analytics",
            "About Project / TOC Theory"
        ],
        label_visibility="collapsed"
    )

    st.markdown("<hr style='border-color: #334155; margin: 20px 0 15px 0;'>", unsafe_allow_html=True)
    
    # Active State Widget in Sidebar
    curr_s = st.session_state.machine_state
    state_meta = STATES[curr_s]
    st.markdown(f"""
    <div style="background: #0F172A; border: 1px solid #334155; border-radius: 6px; padding: 12px; margin-bottom: 12px;">
        <div style="font-size: 0.7rem; color: #94A3B8; text-transform: uppercase; letter-spacing: 0.5px;">Active Machine State</div>
        <div style="font-size: 1.3rem; font-weight: 700; color: {state_meta['color']}; font-family: monospace; margin: 4px 0;">
            {curr_s} &middot; {state_meta['name']}
        </div>
        <div style="font-size: 0.75rem; color: #94A3B8;">{state_meta['desc']}</div>
    </div>
    """, unsafe_allow_html=True)

    if st.button("Reset Mealy Machine", use_container_width=True):
        reset_all_session()
        st.rerun()

    st.markdown("""
    <div style="font-size: 0.72rem; color: #64748B; margin-top: 24px; text-align: center;">
        CSE 2nd Year &bull; Theory of Computation<br>
        Module I &bull; Finite State Automata
    </div>
    """, unsafe_allow_html=True)


# ==============================================================================
# PAGE 1: DASHBOARD
# ==============================================================================
if page == "Dashboard":
    # Top Bar Header
    st.markdown("""
    <div class="siem-header">
        <div>
            <div class="siem-title">LOGGUARD &middot; SECURITY MONITORING</div>
            <div class="siem-subtitle">Real-Time Log Anomaly Detection via Deterministic Mealy Automaton</div>
        </div>
        <div class="siem-badge">MODEL: M = (Q, &Sigma;, &Delta;, &delta;, &lambda;, q<sub>0</sub>)</div>
    </div>
    """, unsafe_allow_html=True)

    # Calculate real-time stats from session history
    events = st.session_state.event_history
    total_events = len(events)
    normal_events = sum(1 for e in events if e["output"] in ["NORMAL", "INFO"])
    warning_events = sum(1 for e in events if e["output"] == "WARNING")
    alert_events = sum(1 for e in events if e["output"] == "SECURITY_ALERT")
    curr_s = st.session_state.machine_state

    # 5 Compact Metric Cards
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.markdown(f"""
        <div class="metric-card" style="border-top: 3px solid #38BDF8;">
            <div class="metric-card-title">Total Events</div>
            <div class="metric-card-val">{total_events}</div>
            <div class="metric-card-sub">Processed in Session</div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="metric-card" style="border-top: 3px solid #10B981;">
            <div class="metric-card-title">Normal / Info</div>
            <div class="metric-card-val">{normal_events}</div>
            <div class="metric-card-sub">{round(normal_events/max(1, total_events)*100, 1)}% of Traffic</div>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div class="metric-card" style="border-top: 3px solid #F59E0B;">
            <div class="metric-card-title">Warnings</div>
            <div class="metric-card-val">{warning_events}</div>
            <div class="metric-card-sub">Failed Login Flags</div>
        </div>
        """, unsafe_allow_html=True)
    with c4:
        st.markdown(f"""
        <div class="metric-card" style="border-top: 3px solid #EF4444;">
            <div class="metric-card-title">Security Alerts</div>
            <div class="metric-card-val">{alert_events}</div>
            <div class="metric-card-sub">Threshold Triggers</div>
        </div>
        """, unsafe_allow_html=True)
    with c5:
        state_color = STATES[curr_s]["color"]
        st.markdown(f"""
        <div class="metric-card" style="border-top: 3px solid {state_color};">
            <div class="metric-card-title">Machine State</div>
            <div class="metric-card-val" style="color: {state_color};">{curr_s}</div>
            <div class="metric-card-sub">{STATES[curr_s]['name']}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)

    # 2 Charts Row
    chart_col1, chart_col2 = st.columns([1, 1])

    with chart_col1:
        st.markdown("<p style='font-size: 0.9rem; font-weight: 600; color: #F8FAFC; margin-bottom: 8px;'>Generated Output Distribution</p>", unsafe_allow_html=True)
        if total_events > 0:
            df_hist = pd.DataFrame(events)
            out_counts = df_hist["output"].value_counts().reset_index()
            out_counts.columns = ["Output", "Count"]

            # Map colors to output types
            color_map = {
                "NORMAL": "#10B981",
                "WARNING": "#F59E0B",
                "SECURITY_ALERT": "#EF4444",
                "RESET": "#38BDF8",
                "INFO": "#64748B"
            }
            colors = [color_map.get(o, "#38BDF8") for o in out_counts["Output"]]

            fig_donut = go.Figure(data=[go.Pie(
                labels=out_counts["Output"],
                values=out_counts["Count"],
                hole=0.55,
                marker=dict(colors=colors, line=dict(color='#0F172A', width=2)),
                textinfo='label+percent',
                textfont=dict(color='#F8FAFC', size=11)
            )])
            fig_donut.update_layout(
                paper_bgcolor='#0F172A',
                plot_bgcolor='#0F172A',
                margin=dict(t=15, b=15, l=15, r=15),
                height=260,
                showlegend=False
            )
            st.plotly_chart(fig_donut, use_container_width=True)
        else:
            st.info("No events logged yet. Inject events in the Live Simulation tab.")

    with chart_col2:
        st.markdown("<p style='font-size: 0.9rem; font-weight: 600; color: #F8FAFC; margin-bottom: 8px;'>Input Symbol Frequency (Sigma)</p>", unsafe_allow_html=True)
        if total_events > 0:
            df_hist = pd.DataFrame(events)
            inp_counts = df_hist["input_event"].value_counts().reset_index()
            inp_counts.columns = ["Input Symbol", "Count"]

            fig_bar = px.bar(
                inp_counts,
                x="Input Symbol",
                y="Count",
                text="Count",
                color_discrete_sequence=["#38BDF8"]
            )
            fig_bar.update_layout(
                paper_bgcolor='#0F172A',
                plot_bgcolor='#0F172A',
                margin=dict(t=15, b=15, l=15, r=15),
                height=260,
                xaxis=dict(
                    showgrid=False,
                    tickfont=dict(color='#94A3B8', size=10),
                    title=None
                ),
                yaxis=dict(
                    showgrid=True,
                    gridcolor='#1E293B',
                    tickfont=dict(color='#94A3B8', size=10),
                    title=None
                ),
                font=dict(color='#F8FAFC')
            )
            st.plotly_chart(fig_bar, use_container_width=True)
        else:
            st.info("Awaiting event stream data...")

    st.markdown("<div style='height: 15px;'></div>", unsafe_allow_html=True)

    # Machine State Indicator & Latest Result
    st.markdown("<p style='font-size: 0.9rem; font-weight: 600; color: #F8FAFC; margin-bottom: 8px;'>Current Automata State Pipeline</p>", unsafe_allow_html=True)
    
    col_p0, col_p1, col_p2, col_p3 = st.columns(4)
    with col_p0:
        act = (curr_s == "S0")
        b_col = "#38BDF8" if act else "#334155"
        bg = "#0C4A6E" if act else "#1E293B"
        st.markdown(f"<div style='background: {bg}; border: 1px solid {b_col}; border-radius: 6px; padding: 12px; text-align: center;'><div style='font-weight: 700; color: #10B981; font-family: monospace;'>S0 {'&#9679;' if act else ''}</div><div style='font-size: 0.75rem; color: #94A3B8;'>Normal (0 Fails)</div></div>", unsafe_allow_html=True)
    with col_p1:
        act = (curr_s == "S1")
        b_col = "#38BDF8" if act else "#334155"
        bg = "#0C4A6E" if act else "#1E293B"
        st.markdown(f"<div style='background: {bg}; border: 1px solid {b_col}; border-radius: 6px; padding: 12px; text-align: center;'><div style='font-weight: 700; color: #F59E0B; font-family: monospace;'>S1 {'&#9679;' if act else ''}</div><div style='font-size: 0.75rem; color: #94A3B8;'>Warning 1 (1 Fail)</div></div>", unsafe_allow_html=True)
    with col_p2:
        act = (curr_s == "S2")
        b_col = "#38BDF8" if act else "#334155"
        bg = "#0C4A6E" if act else "#1E293B"
        st.markdown(f"<div style='background: {bg}; border: 1px solid {b_col}; border-radius: 6px; padding: 12px; text-align: center;'><div style='font-weight: 700; color: #F97316; font-family: monospace;'>S2 {'&#9679;' if act else ''}</div><div style='font-size: 0.75rem; color: #94A3B8;'>Warning 2 (2 Fails)</div></div>", unsafe_allow_html=True)
    with col_p3:
        act = (curr_s == "S3")
        b_col = "#38BDF8" if act else "#334155"
        bg = "#0C4A6E" if act else "#1E293B"
        st.markdown(f"<div style='background: {bg}; border: 1px solid {b_col}; border-radius: 6px; padding: 12px; text-align: center;'><div style='font-weight: 700; color: #EF4444; font-family: monospace;'>S3 {'&#9679;' if act else ''}</div><div style='font-size: 0.75rem; color: #94A3B8;'>Alert (3+ Fails)</div></div>", unsafe_allow_html=True)

    # Recent Log Events Table
    st.markdown("<p style='font-size: 0.9rem; font-weight: 600; color: #F8FAFC; margin-top: 15px; margin-bottom: 8px;'>Recent Log Events & Mealy Output Trace (Last 10 Records)</p>", unsafe_allow_html=True)
    if events:
        recent = events[-10:][::-1]
        display_rows = []
        for r in recent:
            out = r["output"]
            pill_cls = "pill-normal" if out == "NORMAL" else ("pill-warning" if out == "WARNING" else ("pill-alert" if out == "SECURITY_ALERT" else ("pill-reset" if out == "RESET" else "pill-info")))
            display_rows.append({
                "Event #": r["event_num"],
                "Timestamp": r["timestamp"],
                "User": r["username"],
                "IP Address": r["ip_address"],
                "Input Symbol (&Sigma;)": r["input_event"],
                "Transition (&delta;)": f"{r['prev_state']} &rarr; {r['next_state']}",
                "Output Symbol (&Delta;)": f"<span class='badge-pill {pill_cls}'>{out}</span>"
            })
        df_recent = pd.DataFrame(display_rows)
        st.write(df_recent.to_html(escape=False, index=False), unsafe_allow_html=True)
    else:
        st.info("No events in history. Use Live Simulation or Guided Demonstration to run events.")


# ==============================================================================
# PAGE 2: LIVE SIMULATION
# ==============================================================================
elif page == "Live Simulation":
    st.markdown("""
    <div class="siem-header">
        <div>
            <div class="siem-title">LIVE SIMULATION ENGINE</div>
            <div class="siem-subtitle">Real-Time Event Injection & Interactive Mealy State Transitions</div>
        </div>
        <div class="siem-badge">INTERACTIVE TESTBENCH</div>
    </div>
    """, unsafe_allow_html=True)

    # SECTION A: SIMULATION CONTROLS
    st.markdown("<div style='background: #0F172A; border: 1px solid #334155; border-radius: 8px; padding: 18px; margin-bottom: 20px;'>", unsafe_allow_html=True)
    st.markdown("<div style='font-size: 0.8rem; font-weight: 700; color: #38BDF8; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 12px;'>SECTION A: SIMULATION CONTROLS</div>", unsafe_allow_html=True)
    
    ctrl_col1, ctrl_col2, ctrl_col3, ctrl_col4 = st.columns([1, 1, 1, 1])
    with ctrl_col1:
        if st.button("Process Next Random Event", use_container_width=True, type="primary"):
            random_evt = random.choices(
                INPUT_ALPHABET,
                weights=[0.35, 0.25, 0.15, 0.10, 0.10, 0.05],
                k=1
            )[0]
            log_event_to_session(random_evt)
            st.rerun()

    with ctrl_col2:
        if st.button("Run Batch (5 Random Events)", use_container_width=True):
            for _ in range(5):
                evt = random.choices(
                    INPUT_ALPHABET,
                    weights=[0.30, 0.30, 0.15, 0.10, 0.10, 0.05],
                    k=1
                )[0]
                log_event_to_session(evt)
            st.rerun()

    with ctrl_col3:
        if st.button("Simulate Attack Wave (3 Fails)", use_container_width=True):
            for _ in range(3):
                log_event_to_session("LOGIN_FAIL", username="attacker_bot", ip_address="203.0.113.88")
            st.rerun()

    with ctrl_col4:
        if st.button("Reset Simulation", use_container_width=True):
            reset_all_session()
            st.rerun()

    st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
    st.markdown("<div style='font-size: 0.75rem; color: #94A3B8; margin-bottom: 8px;'>MANUAL EVENT INJECTION (Input Alphabet &Sigma;):</div>", unsafe_allow_html=True)
    
    # Dedicated Manual Event Injection Buttons
    btn_c1, btn_c2, btn_c3, btn_c4, btn_c5, btn_c6 = st.columns(6)
    with btn_c1:
        if st.button("+ LOGIN_SUCCESS", use_container_width=True):
            log_event_to_session("LOGIN_SUCCESS")
            st.rerun()
    with btn_c2:
        if st.button("+ LOGIN_FAIL", use_container_width=True):
            log_event_to_session("LOGIN_FAIL")
            st.rerun()
    with btn_c3:
        if st.button("+ LOGOUT", use_container_width=True):
            log_event_to_session("LOGOUT")
            st.rerun()
    with btn_c4:
        if st.button("+ ACCESS_DENIED", use_container_width=True):
            log_event_to_session("ACCESS_DENIED")
            st.rerun()
    with btn_c5:
        if st.button("+ ACCESS_GRANTED", use_container_width=True):
            log_event_to_session("ACCESS_GRANTED")
            st.rerun()
    with btn_c6:
        if st.button("+ SYSTEM_ERROR", use_container_width=True):
            log_event_to_session("SYSTEM_ERROR")
            st.rerun()
    
    st.markdown("</div>", unsafe_allow_html=True)

    # SECTION B: CURRENT MACHINE STATE VISUALIZATION
    curr_s = st.session_state.machine_state
    state_meta = STATES[curr_s]
    last_event = st.session_state.event_history[-1] if st.session_state.event_history else None

    st.markdown("<div style='background: #0F172A; border: 1px solid #334155; border-radius: 8px; padding: 18px; margin-bottom: 20px;'>", unsafe_allow_html=True)
    st.markdown("<div style='font-size: 0.8rem; font-weight: 700; color: #38BDF8; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 12px;'>SECTION B: CURRENT MACHINE STATE INSPECTION</div>", unsafe_allow_html=True)

    ms_c1, ms_c2, ms_c3, ms_c4 = st.columns(4)
    with ms_c1:
        st.markdown(f"""
        <div style="background: #1E293B; border-left: 4px solid {state_meta['color']}; padding: 12px 16px; border-radius: 4px;">
            <div style="font-size: 0.72rem; color: #94A3B8; text-transform: uppercase;">Current State (q)</div>
            <div style="font-size: 1.6rem; font-weight: 700; color: {state_meta['color']}; font-family: monospace;">{curr_s}</div>
            <div style="font-size: 0.78rem; color: #F8FAFC;">{state_meta['name']}</div>
        </div>
        """, unsafe_allow_html=True)
    with ms_c2:
        last_inp = last_event["input_event"] if last_event else "NONE"
        st.markdown(f"""
        <div style="background: #1E293B; border-left: 4px solid #38BDF8; padding: 12px 16px; border-radius: 4px;">
            <div style="font-size: 0.72rem; color: #94A3B8; text-transform: uppercase;">Last Input Event (&Sigma;)</div>
            <div style="font-size: 1.25rem; font-weight: 700; color: #38BDF8; font-family: monospace; margin-top: 4px;">{last_inp}</div>
            <div style="font-size: 0.78rem; color: #94A3B8;">Applied to &delta;(q, &sigma;)</div>
        </div>
        """, unsafe_allow_html=True)
    with ms_c3:
        last_out = last_event["output"] if last_event else "NONE"
        out_col = "#10B981" if last_out in ["NORMAL", "RESET"] else ("#F59E0B" if last_out == "WARNING" else ("#EF4444" if last_out == "SECURITY_ALERT" else "#94A3B8"))
        st.markdown(f"""
        <div style="background: #1E293B; border-left: 4px solid {out_col}; padding: 12px 16px; border-radius: 4px;">
            <div style="font-size: 0.72rem; color: #94A3B8; text-transform: uppercase;">Generated Output (&Delta;)</div>
            <div style="font-size: 1.25rem; font-weight: 700; color: {out_col}; font-family: monospace; margin-top: 4px;">{last_out}</div>
            <div style="font-size: 0.78rem; color: #94A3B8;">Produced by &lambda;(q, &sigma;)</div>
        </div>
        """, unsafe_allow_html=True)
    with ms_c4:
        prev_s = last_event["prev_state"] if last_event else "S0"
        st.markdown(f"""
        <div style="background: #1E293B; border-left: 4px solid #64748B; padding: 12px 16px; border-radius: 4px;">
            <div style="font-size: 0.72rem; color: #94A3B8; text-transform: uppercase;">Latest State Transition</div>
            <div style="font-size: 1.25rem; font-weight: 700; color: #F8FAFC; font-family: monospace; margin-top: 4px;">{prev_s} &rarr; {curr_s}</div>
            <div style="font-size: 0.78rem; color: #94A3B8;">q<sub>prev</sub> &rarr; q<sub>next</sub></div>
        </div>
        """, unsafe_allow_html=True)

    # Visual State Pipeline
    ls_p0, ls_p1, ls_p2, ls_p3 = st.columns(4)
    with ls_p0:
        act = (curr_s == "S0")
        b_col = "#38BDF8" if act else "#334155"
        bg = "#0C4A6E" if act else "#1E293B"
        st.markdown(f"<div style='background: {bg}; border: 1px solid {b_col}; border-radius: 6px; padding: 12px; text-align: center;'><div style='font-weight: 700; color: #10B981; font-family: monospace;'>S0 {'&#9679;' if act else ''}</div><div style='font-size: 0.75rem; color: #94A3B8;'>NORMAL (0 Fails)</div></div>", unsafe_allow_html=True)
    with ls_p1:
        act = (curr_s == "S1")
        b_col = "#38BDF8" if act else "#334155"
        bg = "#0C4A6E" if act else "#1E293B"
        st.markdown(f"<div style='background: {bg}; border: 1px solid {b_col}; border-radius: 6px; padding: 12px; text-align: center;'><div style='font-weight: 700; color: #F59E0B; font-family: monospace;'>S1 {'&#9679;' if act else ''}</div><div style='font-size: 0.75rem; color: #94A3B8;'>WARNING 1 (1 Fail)</div></div>", unsafe_allow_html=True)
    with ls_p2:
        act = (curr_s == "S2")
        b_col = "#38BDF8" if act else "#334155"
        bg = "#0C4A6E" if act else "#1E293B"
        st.markdown(f"<div style='background: {bg}; border: 1px solid {b_col}; border-radius: 6px; padding: 12px; text-align: center;'><div style='font-weight: 700; color: #F97316; font-family: monospace;'>S2 {'&#9679;' if act else ''}</div><div style='font-size: 0.75rem; color: #94A3B8;'>WARNING 2 (2 Fails)</div></div>", unsafe_allow_html=True)
    with ls_p3:
        act = (curr_s == "S3")
        b_col = "#38BDF8" if act else "#334155"
        bg = "#0C4A6E" if act else "#1E293B"
        st.markdown(f"<div style='background: {bg}; border: 1px solid {b_col}; border-radius: 6px; padding: 12px; text-align: center;'><div style='font-weight: 700; color: #EF4444; font-family: monospace;'>S3 {'&#9679;' if act else ''}</div><div style='font-size: 0.75rem; color: #94A3B8;'>SECURITY ALERT (3+ Fails)</div></div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # SECTION C: EVENT EXECUTION TRACE
    st.markdown("<div style='background: #0F172A; border: 1px solid #334155; border-radius: 8px; padding: 18px;'>", unsafe_allow_html=True)
    st.markdown("<div style='font-size: 0.8rem; font-weight: 700; color: #38BDF8; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 12px;'>SECTION C: LIVE EVENT EXECUTION TRACE</div>", unsafe_allow_html=True)

    if st.session_state.event_history:
        # Show all events reversed
        display_rows = []
        for idx, r in enumerate(reversed(st.session_state.event_history)):
            out = r["output"]
            pill_cls = "pill-normal" if out == "NORMAL" else ("pill-warning" if out == "WARNING" else ("pill-alert" if out == "SECURITY_ALERT" else ("pill-reset" if out == "RESET" else "pill-info")))
            is_latest = (idx == 0)
            indicator = " <span style='color: #38BDF8; font-weight:bold; font-size: 0.7rem;'>[LATEST]</span>" if is_latest else ""
            display_rows.append({
                "Event #": f"{r['event_num']}{indicator}",
                "Timestamp": r["timestamp"],
                "User": r["username"],
                "Input Symbol (&Sigma;)": r["input_event"],
                "Prev State": r["prev_state"],
                "Next State": r["next_state"],
                "Generated Output (&Delta;)": f"<span class='badge-pill {pill_cls}'>{out}</span>"
            })
        df_trace = pd.DataFrame(display_rows)
        st.write(df_trace.to_html(escape=False, index=False), unsafe_allow_html=True)
    else:
        st.info("No events recorded yet. Click one of the buttons above to inject events into the Mealy Machine.")
    
    st.markdown("</div>", unsafe_allow_html=True)


# ==============================================================================
# PAGE 3: GUIDED DEMONSTRATION (PRESENTATION MODE)
# ==============================================================================
elif page == "Guided Demonstration":
    st.markdown("""
    <div class="siem-header">
        <div>
            <div class="siem-title">GUIDED MEALY MACHINE DEMONSTRATION</div>
            <div class="siem-subtitle">Step-by-Step Formal Automaton Execution Mode for Viva & Presentations</div>
        </div>
        <div class="siem-badge">PRESENTATION MODE</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="background: #0F172A; border: 1px solid #334155; border-radius: 8px; padding: 16px; margin-bottom: 20px;">
        <span style="color: #38BDF8; font-weight: 600;">Presentation Guide:</span>
        This mode runs deterministic, step-by-step test sequences required to prove your Mealy Machine implementation to professors.
        You can advance one event at a time using <b>Next Step</b> and explain the formal transition functions: 
        <code style="color: #38BDF8;">&delta;(q, &sigma;) &rarr; q'</code> and <code style="color: #38BDF8;">&lambda;(q, &sigma;) &rarr; &Delta;</code>.
    </div>
    """, unsafe_allow_html=True)

    # Scenarios Definitions
    SCENARIOS = {
        "scenario_1": {
            "title": "Scenario 1: Normal Login Sequence",
            "desc": "Demonstrates legitimate user behavior maintaining S0 state.",
            "sequence": ["LOGIN_SUCCESS", "LOGOUT", "LOGIN_SUCCESS"],
            "expected_states": ["S0", "S0", "S0"],
            "expected_outputs": ["NORMAL", "INFO", "NORMAL"],
            "objective": "Prove that normal credentials do not trigger alerts and retain baseline state S0."
        },
        "scenario_2": {
            "title": "Scenario 2: Suspicious Login Activity (Brute Force Detection)",
            "desc": "Demonstrates state progression S0 -> S1 -> S2 -> S3 upon 3 consecutive failures.",
            "sequence": ["LOGIN_FAIL", "LOGIN_FAIL", "LOGIN_FAIL"],
            "expected_states": ["S1", "S2", "S3"],
            "expected_outputs": ["WARNING", "WARNING", "SECURITY_ALERT"],
            "objective": "Verify the anomaly detection threshold: exactly 3 failed attempts triggers the SECURITY_ALERT state S3."
        },
        "scenario_3": {
            "title": "Scenario 3: Alert Reset via Legitimate Authentication",
            "desc": "Demonstrates recovery from S3 alert state back to S0 upon valid login.",
            "sequence": ["LOGIN_FAIL", "LOGIN_FAIL", "LOGIN_FAIL", "LOGIN_SUCCESS"],
            "expected_states": ["S1", "S2", "S3", "S0"],
            "expected_outputs": ["WARNING", "WARNING", "SECURITY_ALERT", "RESET"],
            "objective": "Verify that when in alert state S3, a valid LOGIN_SUCCESS transitions to S0 with output RESET."
        }
    }

    # Scenario Selection
    scen_col1, scen_col2 = st.columns([2, 1])
    with scen_col1:
        chosen_scen = st.selectbox(
            "Select Scenario to Demonstrate:",
            options=["scenario_1", "scenario_2", "scenario_3"],
            format_func=lambda x: SCENARIOS[x]["title"]
        )
    with scen_col2:
        st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
        if st.button("Reset Scenario", use_container_width=True):
            st.session_state.demo_scenario = chosen_scen
            st.session_state.demo_step = 0
            st.session_state.demo_history = []
            st.session_state.demo_state = "S0"
            st.rerun()

    # If scenario changed, reset state
    if st.session_state.demo_scenario != chosen_scen:
        st.session_state.demo_scenario = chosen_scen
        st.session_state.demo_step = 0
        st.session_state.demo_history = []
        st.session_state.demo_state = "S0"

    curr_scen_data = SCENARIOS[chosen_scen]
    total_steps = len(curr_scen_data["sequence"])
    current_step_idx = st.session_state.demo_step

    st.markdown(f"""
    <div style="background: #1E293B; border-left: 4px solid #38BDF8; padding: 12px 18px; border-radius: 4px; margin-bottom: 18px;">
        <div style="font-size: 0.82rem; color: #94A3B8; text-transform: uppercase;">Objective</div>
        <div style="font-size: 0.95rem; color: #F8FAFC; margin-top: 2px;">{curr_scen_data['objective']}</div>
    </div>
    """, unsafe_allow_html=True)

    # Step Controls Row
    btn_s1, btn_s2 = st.columns([1, 1])
    with btn_s1:
        next_disabled = (current_step_idx >= total_steps)
        if st.button("Next Step ⏭", disabled=next_disabled, use_container_width=True, type="primary"):
            inp_sym = curr_scen_data["sequence"][current_step_idx]
            prev_s = st.session_state.demo_state
            p_s, i_s, n_s, o_s = execute_mealy_transition(prev_s, inp_sym)
            
            st.session_state.demo_state = n_s
            st.session_state.demo_step += 1
            st.session_state.demo_history.append({
                "step": st.session_state.demo_step,
                "input": inp_sym,
                "prev_state": p_s,
                "next_state": n_s,
                "output": o_s
            })
            # Also log to main history for consistency
            log_event_to_session(inp_sym, username="demo_user", ip_address="198.51.100.99")
            st.rerun()

    with btn_s2:
        if st.button("Run Full Scenario ⚡", disabled=(current_step_idx >= total_steps), use_container_width=True):
            while st.session_state.demo_step < total_steps:
                idx = st.session_state.demo_step
                inp_sym = curr_scen_data["sequence"][idx]
                prev_s = st.session_state.demo_state
                p_s, i_s, n_s, o_s = execute_mealy_transition(prev_s, inp_sym)
                st.session_state.demo_state = n_s
                st.session_state.demo_step += 1
                st.session_state.demo_history.append({
                    "step": st.session_state.demo_step,
                    "input": inp_sym,
                    "prev_state": p_s,
                    "next_state": n_s,
                    "output": o_s
                })
                log_event_to_session(inp_sym, username="demo_user", ip_address="198.51.100.99")
            st.rerun()

    st.markdown("<div style='height: 15px;'></div>", unsafe_allow_html=True)

    # Sequence Visual Pipeline
    st.markdown("<p style='font-size: 0.85rem; font-weight: 600; color: #F8FAFC;'>INPUT SEQUENCE PROGRESSION:</p>", unsafe_allow_html=True)
    
    step_cols = st.columns(len(curr_scen_data["sequence"]))
    for idx, (sym, col) in enumerate(zip(curr_scen_data["sequence"], step_cols)):
        is_done = idx < current_step_idx
        is_curr = (idx == current_step_idx)
        
        bg_col = "#0C4A6E" if is_curr else ("#064E3B" if is_done else "#1E293B")
        border_col = "#38BDF8" if is_curr else ("#10B981" if is_done else "#334155")
        status_txt = "COMPLETED" if is_done else ("CURRENT STEP" if is_curr else "PENDING")
        status_col = "#34D399" if is_done else ("#38BDF8" if is_curr else "#94A3B8")
        badge_icon = "&#10003; " if is_done else ("&#9654; " if is_curr else "&#9679; ")

        card_html = (
            f"<div style='background: {bg_col}; border: 1px solid {border_col}; border-radius: 6px; padding: 14px; font-family: monospace; min-height: 95px;'>"
            f"<div style='font-size: 0.68rem; color: {status_col}; font-weight: 700;'>{badge_icon}STEP {idx+1} &bull; {status_txt}</div>"
            f"<div style='font-size: 0.95rem; font-weight: 700; color: #F8FAFC; margin-top: 6px;'>{sym}</div>"
            f"<div style='font-size: 0.72rem; color: #94A3B8; margin-top: 6px;'>Expected: <b style='color: #F8FAFC;'>{curr_scen_data['expected_outputs'][idx]}</b></div>"
            f"</div>"
        )
        with col:
            st.markdown(card_html, unsafe_allow_html=True)

    # Detailed State Inspector for the latest step
    if st.session_state.demo_history:
        latest = st.session_state.demo_history[-1]
        st.markdown("<div style='background: #0F172A; border: 1px solid #334155; border-radius: 8px; padding: 18px; margin-bottom: 20px;'>", unsafe_allow_html=True)
        st.markdown(f"<div style='font-size: 0.8rem; font-weight: 700; color: #38BDF8; text-transform: uppercase;'>TRANSITION APPLIED (STEP {latest['step']} OF {total_steps})</div>", unsafe_allow_html=True)

        ins_c1, ins_c2 = st.columns([1, 1])
        with ins_c1:
            st.markdown(f"""
            <div style="font-family: monospace; font-size: 0.95rem; line-height: 2; color: #F8FAFC;">
                &bull; Previous State (q): <span style="color: #38BDF8; font-weight: bold;">{latest['prev_state']}</span><br>
                &bull; Input Symbol (&sigma;): <span style="color: #FBBF24; font-weight: bold;">{latest['input']}</span><br>
                &bull; Next State (q'): <span style="color: {STATES[latest['next_state']]['color']}; font-weight: bold;">{latest['next_state']}</span><br>
                &bull; Generated Output (y): <span style="color: #EF4444 if latest['output']=='SECURITY_ALERT' else '#10B981'; font-weight: bold;">{latest['output']}</span>
            </div>
            """, unsafe_allow_html=True)
        with ins_c2:
            st.markdown(f"""
            <div style="background: #1E293B; border-radius: 6px; padding: 12px; font-family: monospace; font-size: 0.88rem;">
                <div style="color: #94A3B8; font-size: 0.75rem; margin-bottom: 4px;">FORMAL TRANSITION EQUATIONS:</div>
                <div style="color: #38BDF8;">&delta;({latest['prev_state']}, {latest['input']}) = {latest['next_state']}</div>
                <div style="color: #34D399;">&lambda;({latest['prev_state']}, {latest['input']}) = {latest['output']}</div>
                <div style="color: #94A3B8; font-size: 0.75rem; margin-top: 8px; border-top: 1px solid #334155; padding-top: 6px;">
                    Transition Table Mapping: ({latest['prev_state']}, {latest['input']}) &rarr; ({latest['next_state']}, {latest['output']})
                </div>
            </div>
            """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    if current_step_idx >= total_steps:
        st.success(f"Scenario Complete! All {total_steps} events successfully processed by the Mealy Machine. Click 'Reset Scenario' to test again.")


# ==============================================================================
# PAGE 4: AUTOMATA VISUALIZER
# ==============================================================================
elif page == "Automata Visualizer":
    st.markdown("""
    <div class="siem-header">
        <div>
            <div class="siem-title">AUTOMATA VISUALIZER & STATE TRANSITION DIAGRAM</div>
            <div class="siem-subtitle">Theoretical Representation of Mealy Machine M = (Q, &Sigma;, &Delta;, &delta;, &lambda;, q<sub>0</sub>)</div>
        </div>
        <div class="siem-badge">FORMAL TOC GRAPH</div>
    </div>
    """, unsafe_allow_html=True)

    curr_s = st.session_state.machine_state

    # Visual State Graph using crisp, styled SVG & Plotly interactive tabs
    st.markdown("<p style='font-size: 0.9rem; font-weight: 600; color: #F8FAFC; margin-bottom: 10px;'>Mealy State Machine & Transition Diagram</p>", unsafe_allow_html=True)

    tab_svg, tab_plotly = st.tabs(["State Transition Diagram (SVG Automata)", "Interactive Network Explorer (Plotly)"])

    with tab_svg:
        s0_active = (curr_s == "S0")
        s1_active = (curr_s == "S1")
        s2_active = (curr_s == "S2")
        s3_active = (curr_s == "S3")

        svg_html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8"/>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ background: #0B1120; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; display: flex; justify-content: center; }}
  .svg-container {{ background: #0F172A; border: 1px solid #334155; border-radius: 8px; padding: 14px; width: 100%; max-width: 880px; text-align: center; }}
  .node-title {{ font-family: 'JetBrains Mono', Consolas, monospace; font-size: 17px; font-weight: bold; fill: #F8FAFC; text-anchor: middle; }}
  .node-sub {{ font-size: 11px; font-weight: 600; text-anchor: middle; }}
  .trans-label {{ font-family: 'JetBrains Mono', Consolas, monospace; font-size: 10px; font-weight: 600; text-anchor: middle; }}
</style>
</head>
<body>
<div class="svg-container">
    <svg width="850" height="340" viewBox="0 0 850 340" xmlns="http://www.w3.org/2000/svg">
        <defs>
            <filter id="glow-active" x="-30%" y="-30%" width="160%" height="160%">
                <feGaussianBlur stdDeviation="5" result="blur" />
                <feMerge>
                    <feMergeNode in="blur"/>
                    <feMergeNode in="SourceGraphic"/>
                </feMerge>
            </filter>
            <marker id="arr-cyan" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
                <path d="M 0 1 L 9 5 L 0 9 z" fill="#38BDF8" />
            </marker>
            <marker id="arr-green" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
                <path d="M 0 1 L 9 5 L 0 9 z" fill="#10B981" />
            </marker>
            <marker id="arr-amber" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
                <path d="M 0 1 L 9 5 L 0 9 z" fill="#F59E0B" />
            </marker>
            <marker id="arr-red" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
                <path d="M 0 1 L 9 5 L 0 9 z" fill="#EF4444" />
            </marker>
        </defs>

        <!-- Start Arrow into S0 -->
        <path d="M 25 155 L 75 155" stroke="#94A3B8" stroke-width="2.5" marker-end="url(#arr-cyan)" />
        <text x="35" y="145" fill="#94A3B8" font-size="12" font-family="monospace">start</text>

        <!-- State S0 -->
        <circle cx="115" cy="155" r="38" fill="#1E293B" stroke="{'#38BDF8' if s0_active else '#10B981'}" stroke-width="{'5' if s0_active else '2.5'}" {'filter="url(#glow-active)"' if s0_active else ''} />
        <text x="115" y="161" class="node-title">S0</text>
        <text x="115" y="210" class="node-sub" fill="#10B981">{'● ACTIVE (Normal)' if s0_active else 'Normal'}</text>

        <!-- S0 self loop: LOGIN_SUCCESS / NORMAL -->
        <path d="M 100 120 C 75 60, 155 60, 130 120" fill="none" stroke="#10B981" stroke-width="2" marker-end="url(#arr-green)" />
        <text x="115" y="52" fill="#34D399" class="trans-label">LOGIN_SUCCESS / NORMAL</text>

        <!-- Transition S0 -> S1: LOGIN_FAIL / WARNING -->
        <path d="M 153 155 L 297 155" fill="none" stroke="#F59E0B" stroke-width="2.5" marker-end="url(#arr-amber)" />
        <text x="225" y="143" fill="#FCD34D" class="trans-label">LOGIN_FAIL / WARNING</text>

        <!-- State S1 -->
        <circle cx="335" cy="155" r="38" fill="#1E293B" stroke="{'#38BDF8' if s1_active else '#F59E0B'}" stroke-width="{'5' if s1_active else '2.5'}" {'filter="url(#glow-active)"' if s1_active else ''} />
        <text x="335" y="161" class="node-title">S1</text>
        <text x="335" y="210" class="node-sub" fill="#F59E0B">{'● ACTIVE (Warn 1)' if s1_active else 'Warning 1'}</text>

        <!-- Transition S1 -> S2: LOGIN_FAIL / WARNING -->
        <path d="M 373 155 L 517 155" fill="none" stroke="#F97316" stroke-width="2.5" marker-end="url(#arr-amber)" />
        <text x="445" y="143" fill="#FCD34D" class="trans-label">LOGIN_FAIL / WARNING</text>

        <!-- State S2 -->
        <circle cx="555" cy="155" r="38" fill="#1E293B" stroke="{'#38BDF8' if s2_active else '#F97316'}" stroke-width="{'5' if s2_active else '2.5'}" {'filter="url(#glow-active)"' if s2_active else ''} />
        <text x="555" y="161" class="node-title">S2</text>
        <text x="555" y="210" class="node-sub" fill="#F97316">{'● ACTIVE (Warn 2)' if s2_active else 'Warning 2'}</text>

        <!-- Transition S2 -> S3: LOGIN_FAIL / SECURITY_ALERT -->
        <path d="M 593 155 L 737 155" fill="none" stroke="#EF4444" stroke-width="2.5" marker-end="url(#arr-red)" />
        <text x="665" y="143" fill="#FCA5A5" class="trans-label">LOGIN_FAIL / ALERT</text>

        <!-- State S3: Double Outer Ring representing Security Alert State -->
        <circle cx="775" cy="155" r="44" fill="none" stroke="#EF4444" stroke-width="1.5" stroke-dasharray="3" />
        <circle cx="775" cy="155" r="38" fill="#1E293B" stroke="{'#38BDF8' if s3_active else '#EF4444'}" stroke-width="{'5' if s3_active else '2.5'}" {'filter="url(#glow-active)"' if s3_active else ''} />
        <text x="775" y="161" class="node-title">S3</text>
        <text x="775" y="210" class="node-sub" fill="#EF4444">{'● ACTIVE (Alert)' if s3_active else 'Security Alert'}</text>

        <!-- S3 self loop: LOGIN_FAIL / SECURITY_ALERT -->
        <path d="M 760 120 C 735 60, 815 60, 790 120" fill="none" stroke="#EF4444" stroke-width="2" marker-end="url(#arr-red)" />
        <text x="775" y="52" fill="#FCA5A5" class="trans-label">LOGIN_FAIL / ALERT</text>

        <!-- Curved Return Transitions to S0 on LOGIN_SUCCESS -->
        <!-- S1 -> S0 -->
        <path d="M 320 190 C 265 235, 185 235, 130 190" fill="none" stroke="#10B981" stroke-width="1.8" stroke-dasharray="4" marker-end="url(#arr-green)" />
        <text x="225" y="235" fill="#34D399" class="trans-label">S1: LOGIN_SUCCESS / NORMAL</text>

        <!-- S2 -> S0 -->
        <path d="M 540 190 C 425 275, 245 275, 125 195" fill="none" stroke="#10B981" stroke-width="1.8" stroke-dasharray="4" marker-end="url(#arr-green)" />
        <text x="345" y="275" fill="#34D399" class="trans-label">S2: LOGIN_SUCCESS / NORMAL</text>

        <!-- S3 -> S0: LOGIN_SUCCESS / RESET -->
        <path d="M 760 190 C 615 325, 275 325, 120 195" fill="none" stroke="#38BDF8" stroke-width="2.5" marker-end="url(#arr-cyan)" />
        <text x="440" y="320" fill="#38BDF8" class="trans-label" font-size="11px">S3 &rarr; S0: LOGIN_SUCCESS / RESET (Alert Remediated)</text>
    </svg>
</div>
</body>
</html>"""
        components.html(svg_html, height=365, scrolling=False)

    with tab_plotly:
        # Interactive Plotly Network Graph
        node_x = [1, 2.5, 4, 5.5]
        node_y = [2, 2, 2, 2]
        node_names = ["S0 (Normal)", "S1 (Warning 1)", "S2 (Warning 2)", "S3 (Alert)"]
        node_colors = [
            "#38BDF8" if curr_s == "S0" else "#10B981",
            "#38BDF8" if curr_s == "S1" else "#F59E0B",
            "#38BDF8" if curr_s == "S2" else "#F97316",
            "#38BDF8" if curr_s == "S3" else "#EF4444"
        ]
        node_sizes = [55 if curr_s == "S0" else 45, 55 if curr_s == "S1" else 45, 55 if curr_s == "S2" else 45, 55 if curr_s == "S3" else 45]

        fig_net = go.Figure()

        # Add nodes
        fig_net.add_trace(go.Scatter(
            x=node_x,
            y=node_y,
            mode='markers+text',
            marker=dict(size=node_sizes, color=node_colors, line=dict(width=2, color='#FFFFFF')),
            text=["<b>S0</b>", "<b>S1</b>", "<b>S2</b>", "<b>S3</b>"],
            textposition="middle center",
            textfont=dict(color="#0B1120", size=14, family="monospace"),
            hoverinfo='text',
            hovertext=[
                "<b>State S0: Normal</b><br>0 failed logins<br>Input: LOGIN_FAIL -> S1 (WARNING)<br>Input: LOGIN_SUCCESS -> S0 (NORMAL)",
                "<b>State S1: Warning 1</b><br>1 failed login<br>Input: LOGIN_FAIL -> S2 (WARNING)<br>Input: LOGIN_SUCCESS -> S0 (NORMAL)",
                "<b>State S2: Warning 2</b><br>2 failed logins<br>Input: LOGIN_FAIL -> S3 (SECURITY_ALERT)<br>Input: LOGIN_SUCCESS -> S0 (NORMAL)",
                "<b>State S3: Security Alert</b><br>3+ failed logins<br>Input: LOGIN_FAIL -> S3 (SECURITY_ALERT)<br>Input: LOGIN_SUCCESS -> S0 (RESET)"
            ]
        ))

        # Add annotations for transitions
        annotations = [
            dict(x=1.75, y=2.08, text="LOGIN_FAIL / WARNING &rarr;", showarrow=False, font=dict(color="#FCD34D", size=10, family="monospace")),
            dict(x=3.25, y=2.08, text="LOGIN_FAIL / WARNING &rarr;", showarrow=False, font=dict(color="#FCD34D", size=10, family="monospace")),
            dict(x=4.75, y=2.08, text="LOGIN_FAIL / ALERT &rarr;", showarrow=False, font=dict(color="#FCA5A5", size=10, family="monospace")),
            dict(x=3.25, y=1.70, text="&larr; LOGIN_SUCCESS / RESET (from S3)", showarrow=False, font=dict(color="#38BDF8", size=11, family="monospace")),
        ]

        fig_net.update_layout(
            paper_bgcolor='#0F172A',
            plot_bgcolor='#0F172A',
            height=300,
            margin=dict(t=20, b=20, l=30, r=30),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[0.4, 6.1]),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[1.5, 2.5]),
            annotations=annotations,
            showlegend=False
        )
        st.plotly_chart(fig_net, use_container_width=True)

    # Formal 6-Tuple Definition Section
    st.markdown("<p style='font-size: 0.9rem; font-weight: 600; color: #F8FAFC; margin-bottom: 10px;'>Formal 6-Tuple Definition: M = (Q, &Sigma;, &Delta;, &delta;, &lambda;, q<sub>0</sub>)</p>", unsafe_allow_html=True)
    
    tup_c1, tup_c2, tup_c3 = st.columns(3)
    with tup_c1:
        st.markdown("""
        <div style="background: #0F172A; border: 1px solid #334155; border-radius: 6px; padding: 14px; font-family: monospace; font-size: 0.82rem;">
            <div style="color: #38BDF8; font-weight: bold; margin-bottom: 6px;">Q (States)</div>
            Q = {S0, S1, S2, S3}<br>
            <span style="color: #94A3B8; font-size: 0.75rem;">S0: Normal | S1: Warn 1<br>S2: Warn 2 | S3: Security Alert</span>
            <div style="margin-top: 10px; color: #38BDF8; font-weight: bold;">q<sub>0</sub> (Initial State)</div>
            q<sub>0</sub> = S0 &isin; Q
        </div>
        """, unsafe_allow_html=True)
    with tup_c2:
        st.markdown("""
        <div style="background: #0F172A; border: 1px solid #334155; border-radius: 6px; padding: 14px; font-family: monospace; font-size: 0.82rem;">
            <div style="color: #38BDF8; font-weight: bold; margin-bottom: 6px;">&Sigma; (Input Alphabet)</div>
            &Sigma; = {<br>
            &nbsp;&nbsp;LOGIN_SUCCESS, LOGIN_FAIL,<br>
            &nbsp;&nbsp;LOGOUT, ACCESS_GRANTED,<br>
            &nbsp;&nbsp;ACCESS_DENIED, SYSTEM_ERROR<br>
            }
        </div>
        """, unsafe_allow_html=True)
    with tup_c3:
        st.markdown("""
        <div style="background: #0F172A; border: 1px solid #334155; border-radius: 6px; padding: 14px; font-family: monospace; font-size: 0.82rem;">
            <div style="color: #38BDF8; font-weight: bold; margin-bottom: 6px;">&Delta; (Output Alphabet)</div>
            &Delta; = {<br>
            &nbsp;&nbsp;NORMAL, WARNING,<br>
            &nbsp;&nbsp;SECURITY_ALERT, RESET,<br>
            &nbsp;&nbsp;INFO<br>
            }
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)

    # Transition Table (Complete 24 Entries)
    st.markdown("<p style='font-size: 0.9rem; font-weight: 600; color: #F8FAFC; margin-bottom: 10px;'>Complete State Transition & Output Table</p>", unsafe_allow_html=True)

    table_data = []
    for (state, inp), (next_state, out) in sorted(TRANSITION_TABLE.items()):
        pill_cls = "pill-normal" if out == "NORMAL" else ("pill-warning" if out == "WARNING" else ("pill-alert" if out == "SECURITY_ALERT" else ("pill-reset" if out == "RESET" else "pill-info")))
        is_active = (state == curr_s)
        active_tag = "<span style='color: #38BDF8; font-size: 0.7rem;'> [ACTIVE]</span>" if is_active else ""
        table_data.append({
            "Current State (Q)": f"{state}{active_tag}",
            "Input Symbol (&Sigma;)": inp,
            "Next State (&delta;)": next_state,
            "Generated Output (&lambda;)": f"<span class='badge-pill {pill_cls}'>{out}</span>",
            "Security Policy Rule": "Alert threshold triggered" if out == "SECURITY_ALERT" else ("Session restored to baseline" if out == "RESET" else ("Suspicion level increased" if out == "WARNING" else ("Baseline operational traffic" if out == "NORMAL" else "Neutral session event")))
        })

    df_trans = pd.DataFrame(table_data)
    st.write(df_trans.to_html(escape=False, index=False), unsafe_allow_html=True)


# ==============================================================================
# PAGE 5: LOG EXPLORER
# ==============================================================================
elif page == "Log Explorer":
    st.markdown("""
    <div class="siem-header">
        <div>
            <div class="siem-title">LOG EXPLORER & FORENSIC VIEWER</div>
            <div class="siem-subtitle">Inspect, Search, Filter, and Export Log Audit Traces</div>
        </div>
        <div class="siem-badge">CSV EXPORT SUPPORTED</div>
    </div>
    """, unsafe_allow_html=True)

    events = st.session_state.event_history
    if not events:
        st.info("No logs generated yet. Use Live Simulation to generate events.")
    else:
        df_logs = pd.DataFrame(events)

        # Filter Controls
        f_col1, f_col2, f_col3 = st.columns([1.5, 1, 1])
        with f_col1:
            search_query = st.text_input("Search Username or IP:", placeholder="e.g. alex.t, 203.0.113...")
        with f_col2:
            all_events = sorted(df_logs["input_event"].unique())
            sel_events = st.multiselect("Filter by Event Type:", options=all_events, default=all_events)
        with f_col3:
            all_outputs = sorted(df_logs["output"].unique())
            sel_outputs = st.multiselect("Filter by Output Symbol:", options=all_outputs, default=all_outputs)

        # Apply Filters
        filtered_df = df_logs.copy()
        if search_query:
            filtered_df = filtered_df[
                filtered_df["username"].str.contains(search_query, case=False, na=False) |
                filtered_df["ip_address"].str.contains(search_query, case=False, na=False)
            ]
        if sel_events:
            filtered_df = filtered_df[filtered_df["input_event"].isin(sel_events)]
        if sel_outputs:
            filtered_df = filtered_df[filtered_df["output"].isin(sel_outputs)]

        st.markdown(f"<div style='font-size: 0.8rem; color: #94A3B8; margin-bottom: 10px;'>Showing <b>{len(filtered_df)}</b> of <b>{len(df_logs)}</b> records</div>", unsafe_allow_html=True)

        # Display Table with Badge Styling
        display_data = []
        for _, row in filtered_df.iterrows():
            out = row["output"]
            pill_cls = "pill-normal" if out == "NORMAL" else ("pill-warning" if out == "WARNING" else ("pill-alert" if out == "SECURITY_ALERT" else ("pill-reset" if out == "RESET" else "pill-info")))
            display_data.append({
                "Event #": row["event_num"],
                "Timestamp": row["timestamp"],
                "Username": row["username"],
                "IP Address": row["ip_address"],
                "Input Event": row["input_event"],
                "Transition": f"{row['prev_state']} &rarr; {row['next_state']}",
                "Mealy Output": f"<span class='badge-pill {pill_cls}'>{out}</span>"
            })
        
        st.write(pd.DataFrame(display_data).to_html(escape=False, index=False), unsafe_allow_html=True)

        st.markdown("<div style='height: 15px;'></div>", unsafe_allow_html=True)
        # CSV Export
        csv_bytes = filtered_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Filtered Logs (CSV)",
            data=csv_bytes,
            file_name=f"logguard_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=False
        )


# ==============================================================================
# PAGE 6: ANALYTICS
# ==============================================================================
elif page == "Analytics":
    st.markdown("""
    <div class="siem-header">
        <div>
            <div class="siem-title">SECURITY ANALYTICS & FREQUENCY METRICS</div>
            <div class="siem-subtitle">Statistical Verification of Automata Behavior Across Session Traffic</div>
        </div>
        <div class="siem-badge">EMPIRICAL DATA</div>
    </div>
    """, unsafe_allow_html=True)

    events = st.session_state.event_history
    if not events:
        st.info("No session logs available for analytics. Please run events in Live Simulation.")
    else:
        df = pd.DataFrame(events)
        total = len(df)
        fails = sum(1 for e in events if e["input_event"] == "LOGIN_FAIL")
        alerts = sum(1 for e in events if e["output"] == "SECURITY_ALERT")
        warns = sum(1 for e in events if e["output"] == "WARNING")
        resets = sum(1 for e in events if e["output"] == "RESET")

        # Top Metric Cards
        a1, a2, a3, a4 = st.columns(4)
        with a1:
            st.markdown(f"""
            <div class="metric-card" style="border-top: 3px solid #38BDF8;">
                <div class="metric-card-title">Processed Events</div>
                <div class="metric-card-val">{total}</div>
                <div class="metric-card-sub">Total Stream Volume</div>
            </div>
            """, unsafe_allow_html=True)
        with a2:
            st.markdown(f"""
            <div class="metric-card" style="border-top: 3px solid #F59E0B;">
                <div class="metric-card-title">Failed Login Rate</div>
                <div class="metric-card-val">{fails}</div>
                <div class="metric-card-sub">{round(fails/total*100, 1)}% of all inputs</div>
            </div>
            """, unsafe_allow_html=True)
        with a3:
            st.markdown(f"""
            <div class="metric-card" style="border-top: 3px solid #EF4444;">
                <div class="metric-card-title">Security Alert Triggers</div>
                <div class="metric-card-val">{alerts}</div>
                <div class="metric-card-sub">Transitions to S3</div>
            </div>
            """, unsafe_allow_html=True)
        with a4:
            st.markdown(f"""
            <div class="metric-card" style="border-top: 3px solid #10B981;">
                <div class="metric-card-title">Alert Recovery (Resets)</div>
                <div class="metric-card-val">{resets}</div>
                <div class="metric-card-sub">Restorations S3 &rarr; S0</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)

        # Analytics Charts Row 1
        an_col1, an_col2 = st.columns(2)
        with an_col1:
            st.markdown("<p style='font-size: 0.9rem; font-weight: 600; color: #F8FAFC;'>Output Severity Breakdown</p>", unsafe_allow_html=True)
            out_df = df["output"].value_counts().reset_index()
            out_df.columns = ["Output", "Count"]
            colors = {
                "NORMAL": "#10B981",
                "WARNING": "#F59E0B",
                "SECURITY_ALERT": "#EF4444",
                "RESET": "#38BDF8",
                "INFO": "#64748B"
            }
            fig1 = px.bar(
                out_df,
                x="Output",
                y="Count",
                text="Count",
                color="Output",
                color_discrete_map=colors
            )
            fig1.update_layout(
                paper_bgcolor='#0F172A',
                plot_bgcolor='#0F172A',
                font=dict(color='#F8FAFC'),
                margin=dict(t=15, b=15, l=15, r=15),
                height=260,
                showlegend=False,
                xaxis=dict(showgrid=False, title=None),
                yaxis=dict(gridcolor='#1E293B', title=None)
            )
            st.plotly_chart(fig1, use_container_width=True)

        with an_col2:
            st.markdown("<p style='font-size: 0.9rem; font-weight: 600; color: #F8FAFC;'>State Residence Frequency</p>", unsafe_allow_html=True)
            state_df = df["next_state"].value_counts().reset_index()
            state_df.columns = ["State", "Count"]
            state_colors = {
                "S0": "#10B981",
                "S1": "#F59E0B",
                "S2": "#F97316",
                "S3": "#EF4444"
            }
            fig2 = px.pie(
                state_df,
                names="State",
                values="Count",
                color="State",
                color_discrete_map=state_colors,
                hole=0.5
            )
            fig2.update_layout(
                paper_bgcolor='#0F172A',
                plot_bgcolor='#0F172A',
                font=dict(color='#F8FAFC'),
                margin=dict(t=15, b=15, l=15, r=15),
                height=260,
                showlegend=True
            )
            st.plotly_chart(fig2, use_container_width=True)

        # Transition Frequency Matrix
        st.markdown("<p style='font-size: 0.9rem; font-weight: 600; color: #F8FAFC; margin-top: 15px;'>State Transition Matrix (q<sub>prev</sub> &rarr; q<sub>next</sub>)</p>", unsafe_allow_html=True)
        matrix = pd.crosstab(
            df["prev_state"],
            df["next_state"],
            rownames=["From State"],
            colnames=["To State"],
            dropna=False
        ).reindex(index=["S0", "S1", "S2", "S3"], columns=["S0", "S1", "S2", "S3"], fill_value=0)

        fig_heat = px.imshow(
            matrix,
            text_auto=True,
            color_continuous_scale="Blues",
            aspect="auto"
        )
        fig_heat.update_layout(
            paper_bgcolor='#0F172A',
            plot_bgcolor='#0F172A',
            font=dict(color='#F8FAFC'),
            margin=dict(t=20, b=20, l=20, r=20),
            height=280,
            coloraxis_showscale=False
        )
        st.plotly_chart(fig_heat, use_container_width=True)


# ==============================================================================
# PAGE 7: ABOUT PROJECT & TOC THEORY (VIVA GUIDE)
# ==============================================================================
elif page == "About Project / TOC Theory":
    st.markdown("""
    <div class="siem-header">
        <div>
            <div class="siem-title">THEORY OF COMPUTATION & VIVA GUIDE</div>
            <div class="siem-subtitle">Theoretical Foundations, Mathematical Modeling & Module I CSE Syllabus Alignment</div>
        </div>
        <div class="siem-badge">ACADEMIC DOCUMENTATION</div>
    </div>
    """, unsafe_allow_html=True)

    # 1. Core Definitions
    st.markdown("""
    <div class="theory-box">
        <h4 style="color: #38BDF8; margin-top: 0;">1. Introduction to Finite Automata with Output</h4>
        <p style="color: #94A3B8; font-size: 0.9rem; line-height: 1.6;">
            A <b>Finite Automaton (FA)</b> is a mathematical abstraction of a system that transitions among a finite set of states 
            in response to external inputs. In standard Theory of Computation (TOC), automata are classified into two major categories:
        </p>
        <ul style="color: #F8FAFC; font-size: 0.88rem; line-height: 1.6;">
            <li><b>Finite Automata without Output:</b> DFAs and NFAs, primarily utilized as language recognizers or string acceptors.</li>
            <li><b>Finite Automata with Output:</b> <b>Mealy Machines</b> and <b>Moore Machines</b>, utilized as transducers that process input sequences and generate corresponding output sequences.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    # 2. Formal Mealy Definition
    st.markdown("""
    <div class="theory-box">
        <h4 style="color: #38BDF8; margin-top: 0;">2. Formal 6-Tuple Definition of a Mealy Machine</h4>
        <p style="color: #94A3B8; font-size: 0.9rem;">
            A <b>Mealy Machine</b> is formally specified as a 6-tuple:
        </p>
        <div style="background: #1E293B; border-left: 4px solid #38BDF8; padding: 12px 18px; border-radius: 4px; font-family: monospace; color: #F8FAFC; margin: 12px 0;">
            M = (Q, &Sigma;, &Delta;, &delta;, &lambda;, q<sub>0</sub>)
        </div>
        <table style="width: 100%; color: #F8FAFC; font-size: 0.88rem; border-collapse: collapse; margin-top: 10px;">
            <tr style="border-bottom: 1px solid #334155;">
                <th style="text-align: left; padding: 8px; color: #38BDF8;">Tuple Symbol</th>
                <th style="text-align: left; padding: 8px; color: #38BDF8;">Theoretical Meaning</th>
                <th style="text-align: left; padding: 8px; color: #38BDF8;">Concrete LogGuard Implementation</th>
            </tr>
            <tr style="border-bottom: 1px solid #1E293B;">
                <td style="padding: 8px; font-family: monospace;">Q</td>
                <td style="padding: 8px;">Finite non-empty set of internal states</td>
                <td style="padding: 8px; font-family: monospace;">{S0, S1, S2, S3} (Failure levels)</td>
            </tr>
            <tr style="border-bottom: 1px solid #1E293B;">
                <td style="padding: 8px; font-family: monospace;">&Sigma;</td>
                <td style="padding: 8px;">Finite non-empty input alphabet</td>
                <td style="padding: 8px; font-family: monospace;">{LOGIN_SUCCESS, LOGIN_FAIL, LOGOUT, ...}</td>
            </tr>
            <tr style="border-bottom: 1px solid #1E293B;">
                <td style="padding: 8px; font-family: monospace;">&Delta;</td>
                <td style="padding: 8px;">Finite non-empty output alphabet</td>
                <td style="padding: 8px; font-family: monospace;">{NORMAL, WARNING, SECURITY_ALERT, RESET, INFO}</td>
            </tr>
            <tr style="border-bottom: 1px solid #1E293B;">
                <td style="padding: 8px; font-family: monospace;">&delta;</td>
                <td style="padding: 8px;">State transition function: Q &times; &Sigma; &rarr; Q</td>
                <td style="padding: 8px; font-family: monospace;">TRANSITION_TABLE[(state, input)][0]</td>
            </tr>
            <tr style="border-bottom: 1px solid #1E293B;">
                <td style="padding: 8px; font-family: monospace;">&lambda;</td>
                <td style="padding: 8px;">Output function: Q &times; &Sigma; &rarr; &Delta;</td>
                <td style="padding: 8px; font-family: monospace;">TRANSITION_TABLE[(state, input)][1]</td>
            </tr>
            <tr>
                <td style="padding: 8px; font-family: monospace;">q<sub>0</sub></td>
                <td style="padding: 8px;">Initial start state (q<sub>0</sub> &isin; Q)</td>
                <td style="padding: 8px; font-family: monospace;">S0 (Normal State)</td>
            </tr>
        </table>
    </div>
    """, unsafe_allow_html=True)

    # 3. Mealy vs Moore Comparison
    st.markdown("""
    <div class="theory-box">
        <h4 style="color: #38BDF8; margin-top: 0;">3. Theoretical Comparison: Mealy Machine vs. Moore Machine</h4>
        <table style="width: 100%; color: #F8FAFC; font-size: 0.88rem; border-collapse: collapse;">
            <tr style="border-bottom: 1px solid #334155;">
                <th style="text-align: left; padding: 8px; color: #38BDF8;">Characteristic</th>
                <th style="text-align: left; padding: 8px; color: #38BDF8;">Mealy Machine (Used in LogGuard)</th>
                <th style="text-align: left; padding: 8px; color: #38BDF8;">Moore Machine</th>
            </tr>
            <tr style="border-bottom: 1px solid #1E293B;">
                <td style="padding: 8px; font-weight: bold;">Output Dependency</td>
                <td style="padding: 8px; color: #34D399;">Depends on BOTH current state and current input symbol: &lambda;(q, &sigma;)</td>
                <td style="padding: 8px; color: #94A3B8;">Depends ONLY on current state: &lambda;(q)</td>
            </tr>
            <tr style="border-bottom: 1px solid #1E293B;">
                <td style="padding: 8px; font-weight: bold;">Response Latency</td>
                <td style="padding: 8px; color: #34D399;">Immediate response in the same cycle as input arrival</td>
                <td style="padding: 8px; color: #94A3B8;">Output changes only after state transition completes</td>
            </tr>
            <tr style="border-bottom: 1px solid #1E293B;">
                <td style="padding: 8px; font-weight: bold;">Number of States</td>
                <td style="padding: 8px; color: #34D399;">Requires fewer states for equivalent transductions</td>
                <td style="padding: 8px; color: #94A3B8;">Typically requires more states</td>
            </tr>
            <tr>
                <td style="padding: 8px; font-weight: bold;">SIEM Suitability</td>
                <td style="padding: 8px; color: #34D399;">Ideal: Differentiates legitimate recovery (RESET) vs normal activity (NORMAL) instantly</td>
                <td style="padding: 8px; color: #94A3B8;">Requires separate intermediate states to differentiate outputs</td>
            </tr>
        </table>
    </div>
    """, unsafe_allow_html=True)

    # 4. Viva Questions and Answers
    st.markdown("""
    <div class="theory-box">
        <h4 style="color: #38BDF8; margin-top: 0;">4. Frequently Asked Viva Questions & Model Answers</h4>
        
        <div style="margin-bottom: 14px;">
            <div style="font-weight: 700; color: #FCD34D; font-size: 0.9rem;">Q1: Why is LogGuard built using a Mealy Machine instead of Machine Learning?</div>
            <div style="color: #94A3B8; font-size: 0.85rem; margin-top: 4px;">
                <b>Answer:</b> Finite Automata provide <b>deterministic execution</b> with strictly bounded <b>O(1) time complexity</b> per event 
                and <b>O(1) space complexity</b>. Unlike machine learning models, automata have zero inference latency, require no training datasets, 
                and produce 100% explainable, mathematically verifiable decisions without false-positive black-box hallucinations.
            </div>
        </div>

        <div style="margin-bottom: 14px;">
            <div style="font-weight: 700; color: #FCD34D; font-size: 0.9rem;">Q2: What is the significance of the RESET output?</div>
            <div style="color: #94A3B8; font-size: 0.85rem; margin-top: 4px;">
                <b>Answer:</b> In a Mealy Machine, the output depends on both state and input: &lambda;(S3, LOGIN_SUCCESS) = RESET, 
                whereas &lambda;(S0, LOGIN_SUCCESS) = NORMAL. This enables LogGuard to inform the SIEM administrator that an active security incident 
                has been de-escalated and remediated, rather than merely logging a routine baseline login.
            </div>
        </div>

        <div style="margin-bottom: 14px;">
            <div style="font-weight: 700; color: #FCD34D; font-size: 0.9rem;">Q3: How does the application guarantee complete transition coverage?</div>
            <div style="color: #94A3B8; font-size: 0.85rem; margin-top: 4px;">
                <b>Answer:</b> The transition table explicitly enumerates all 24 Cartesian product pairs |Q| &times; |&Sigma;| = 4 &times; 6 = 24. 
                Any event that does not escalate failed logins (such as LOGOUT or ACCESS_GRANTED) preserves the current state with an INFO output, 
                ensuring the automaton is completely defined with no undefined transitions.
            </div>
        </div>

        <div>
            <div style="font-weight: 700; color: #FCD34D; font-size: 0.9rem;">Q4: What TOC syllabus module does this project cover?</div>
            <div style="color: #94A3B8; font-size: 0.85rem; margin-top: 4px;">
                <b>Answer:</b> This project strictly implements <b>Module I: Regular Languages and Finite Automata</b> (specifically Finite Automata with Output: Mealy and Moore Machines). It does not implement context-free grammars, pushdown automata (Module II/III), or Turing Machines (Module IV/V), maintaining academic clarity and honesty.
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
