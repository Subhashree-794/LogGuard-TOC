# LogGuard – Real-Time Log Anomaly Detection Using Mealy Machines

> **Second-Year CSE Theory of Computation (TOC) Project**  
> **Course Module:** Module I &ndash; Regular Languages & Finite State Automata  
> **Topic:** Finite Automata with Output (Mealy Machine Transducer)

---

## 1. Project Overview

**LogGuard** is a simulation-based cybersecurity monitoring platform (inspired by Security Information and Event Management &ndash; SIEM systems) that demonstrates the practical, real-world utility of **Mealy Machines** in processing system log streams and identifying unauthorized access patterns (such as brute-force authentication attacks).

Unlike non-deterministic or compute-heavy machine learning heuristics, LogGuard is built strictly on **Formal Language and Automata Theory**. By representing authentication states and transitions within a deterministic finite transducer, it achieves:
- **$O(1)$ Time Complexity:** Instantaneous, single-cycle transition per log entry.
- **$O(1)$ Space Complexity:** Fixed, bounded memory requirements regardless of traffic volume.
- **100% Explainable Security:** Every alert corresponds directly to a mathematically verifiable sequence of state transitions $\delta(q, \sigma)$ and output functions $\lambda(q, \sigma)$.

---

## 2. Project Objectives

1. **Practical Transducer Modeling:** Apply formal finite automata with output to a concrete computer engineering problem (log anomaly detection).
2. **Deterministic Intrusion Detection:** Formally identify when a sequence of three consecutive failed login attempts breaches the safe threshold.
3. **State Remediation / Recovery:** Implement formal de-escalation from security alert states back to safe baseline operation upon legitimate authentication.
4. **Academic Viva Readiness:** Provide a step-by-step presentation testbench where computer science students can explain formal transitions ($\delta$ and $\lambda$) one event at a time to an examiner.

---

## 3. Technology Stack

- **Language:** Python 3.10+
- **Frontend Dashboard:** Streamlit
- **Data Manipulation:** Pandas
- **Visualization:** Plotly & SVG Vector Graphics
- **Architecture:** Zero external database / Zero cloud APIs / Pure deterministic finite automaton

---

## 4. Project Structure

```text
LogGuard/
│
├── app.py                # Complete Streamlit application with formal Mealy Machine logic
├── requirements.txt      # Python dependencies (Streamlit, Pandas, Plotly)
├── README.md             # Detailed documentation, mathematical theory, and viva guide
└── logs/
    └── sample_logs.csv   # Pre-seeded RFC 5737-compliant sample audit records
```

---

## 5. Formal Mealy Machine Specification

A **Mealy Machine** is a 6-tuple transducer defined as:

$$M = (Q, \Sigma, \Delta, \delta, \lambda, q_0)$$

### 5.1 The 6 Tuples in LogGuard

1. **$Q$ (States):** The finite set of internal operational states:
   - **$S_0$ (Normal):** Baseline safe state (0 failed login attempts recorded).
   - **$S_1$ (Warning State 1):** Suspicious state &ndash; 1 failed login attempt recorded.
   - **$S_2$ (Warning State 2):** Elevated suspicion &ndash; 2 consecutive failed logins recorded.
   - **$S_3$ (Security Alert):** Threshold breached &ndash; 3 or more failed logins (active brute force).

2. **$\Sigma$ (Input Alphabet):** The finite set of log event symbols:
   $$\Sigma = \{\text{LOGIN\_SUCCESS}, \text{LOGIN\_FAIL}, \text{LOGOUT}, \text{ACCESS\_GRANTED}, \text{ACCESS\_DENIED}, \text{SYSTEM\_ERROR}\}$$

3. **$\Delta$ (Output Alphabet):** The finite set of generated diagnostic alerts:
   $$\Delta = \{\text{NORMAL}, \text{WARNING}, \text{SECURITY\_ALERT}, \text{RESET}, \text{INFO}\}$$

4. **$q_0$ (Initial State):**
   $$q_0 = S_0 \in Q$$

5. **$\delta$ (State Transition Function):**
   $$\delta: Q \times \Sigma \to Q$$

6. **$\lambda$ (Output Function):**
   $$\lambda: Q \times \Sigma \to \Delta$$

---

### 5.2 Transition & Output Table (All 24 Cartesian Product Pairs)

The transition function and output function are completely defined for all $|Q| \times |\Sigma| = 4 \times 6 = 24$ pairs:

| Current State ($q$) | Input Symbol ($\sigma$) | Next State ($\delta(q, \sigma)$) | Output Symbol ($\lambda(q, \sigma)$) | Security Policy Meaning |
|:---:|:---:|:---:|:---:|:---|
| **$S_0$** | `LOGIN_FAIL` | **$S_1$** | `WARNING` | First failure logged; advance to warning state 1 |
| **$S_0$** | `LOGIN_SUCCESS` | **$S_0$** | `NORMAL` | Routine legitimate login; retain normal state |
| **$S_0$** | `LOGOUT` | **$S_0$** | `INFO` | Neutral session termination |
| **$S_0$** | `ACCESS_GRANTED` | **$S_0$** | `INFO` | Legitimate authorization event |
| **$S_0$** | `ACCESS_DENIED` | **$S_0$** | `INFO` | Neutral authorization check event |
| **$S_0$** | `SYSTEM_ERROR` | **$S_0$** | `INFO` | Internal service fault event |
| **$S_1$** | `LOGIN_FAIL` | **$S_2$** | `WARNING` | Second failure logged; advance to warning state 2 |
| **$S_1$** | `LOGIN_SUCCESS` | **$S_0$** | `NORMAL` | Legitimate login restores state to $S_0$ |
| **$S_1$** | `LOGOUT` | **$S_1$** | `INFO` | Preserves login failure count |
| **$S_1$** | `ACCESS_GRANTED` | **$S_1$** | `INFO` | Preserves login failure count |
| **$S_1$** | `ACCESS_DENIED` | **$S_1$** | `INFO` | Preserves login failure count |
| **$S_1$** | `SYSTEM_ERROR` | **$S_1$** | `INFO` | Preserves login failure count |
| **$S_2$** | `LOGIN_FAIL` | **$S_3$** | `SECURITY_ALERT` | **Threshold reached! 3rd consecutive failure triggers alert** |
| **$S_2$** | `LOGIN_SUCCESS` | **$S_0$** | `NORMAL` | Legitimate login restores state to $S_0$ |
| **$S_2$** | `LOGOUT` | **$S_2$** | `INFO` | Preserves login failure count |
| **$S_2$** | `ACCESS_GRANTED` | **$S_2$** | `INFO` | Preserves login failure count |
| **$S_2$** | `ACCESS_DENIED` | **$S_2$** | `INFO` | Preserves login failure count |
| **$S_2$** | `SYSTEM_ERROR` | **$S_2$** | `INFO` | Preserves login failure count |
| **$S_3$** | `LOGIN_FAIL` | **$S_3$** | `SECURITY_ALERT` | Persistent attack; remains in security alert state |
| **$S_3$** | `LOGIN_SUCCESS` | **$S_0$** | `RESET` | **Remediation! Valid authentication resets security alert** |
| **$S_3$** | `LOGOUT` | **$S_3$** | `INFO` | Preserves active alert state |
| **$S_3$** | `ACCESS_GRANTED` | **$S_3$** | `INFO` | Preserves active alert state |
| **$S_3$** | `ACCESS_DENIED` | **$S_3$** | `INFO` | Preserves active alert state |
| **$S_3$** | `SYSTEM_ERROR` | **$S_3$** | `INFO` | Preserves active alert state |

---

## 6. Installation & Execution

### 6.1 Prerequisites
- Python 3.10 or higher installed.
- VS Code, PyCharm, or any terminal environment.

### 6.2 Setup Instructions

1. Clone or open the project folder in VS Code:
   ```bash
   cd c:\Users\SUBHASHREE\Desktop\LogGuard
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Launch the Streamlit application:
   ```bash
   streamlit run app.py
   ```

4. The dashboard will automatically open in your default browser at `http://localhost:8501`.

---

## 7. Guided Demonstration (Presentation Guide for Viva)

LogGuard includes a dedicated **Guided Demonstration** tab designed specifically for classroom presentation and viva examination.

### Scenario 1: Normal User Authentication
- **Input Sequence:** `LOGIN_SUCCESS` $\to$ `LOGOUT` $\to$ `LOGIN_SUCCESS`
- **State Trajectory:** $S_0 \to S_0 \to S_0 \to S_0$
- **Output Sequence:** `NORMAL` $\to$ `INFO` $\to$ `NORMAL`
- **Viva Point:** Shows that baseline legitimate activity generates no alerts and never leaves state $S_0$.

### Scenario 2: Brute-Force Attack Detection
- **Input Sequence:** `LOGIN_FAIL` $\to$ `LOGIN_FAIL` $\to$ `LOGIN_FAIL`
- **State Trajectory:** $S_0 \to S_1 \to S_2 \to S_3$
- **Output Sequence:** `WARNING` $\to$ `WARNING` $\to$ `SECURITY_ALERT`
- **Viva Point:** Shows the step-by-step escalation of suspicion. The 3rd failure reaches the threshold and transitions the automaton into $S_3$ (`SECURITY_ALERT`).

### Scenario 3: Remediation & Security Reset (Core Verification Sequence)
- **Input Sequence:** `LOGIN_FAIL` $\to$ `LOGIN_FAIL` $\to$ `LOGIN_FAIL` $\to$ `LOGIN_SUCCESS`
- **State Trajectory:** $S_0 \to S_1 \to S_2 \to S_3 \to S_0$
- **Output Sequence:** `WARNING` $\to$ `WARNING` $\to$ `SECURITY_ALERT` $\to$ `RESET`
- **Viva Point:** Demonstrates the core characteristic of a Mealy Machine:  
  $\lambda(S_3, \text{LOGIN\_SUCCESS}) = \text{RESET}$, whereas $\lambda(S_0, \text{LOGIN\_SUCCESS}) = \text{NORMAL}$.  
  The output explicitly informs administrators that a previous incident has been resolved.

---

## 8. Viva Questions and Answers

### Q1: Why use a Mealy Machine instead of a Moore Machine?
> **Answer:** In a Moore machine, output depends strictly on the current state ($\lambda: Q \to \Delta$). In a Mealy machine, output depends on both state and input ($\lambda: Q \times \Sigma \to \Delta$).  
> This allows LogGuard to generate different outputs on the same input symbol depending on which state it occurred in. For instance, `LOGIN_SUCCESS` from $S_0$ produces `NORMAL`, while from $S_3$ it produces `RESET`. Implementing this in a Moore machine would require additional intermediate states.

### Q2: What is the computational complexity of event processing?
> **Answer:** Event processing uses a Python dictionary hash table lookup `TRANSITION_TABLE[(current_state, event)]`.  
> - **Time Complexity:** $O(1)$ constant time per log line.  
> - **Space Complexity:** $O(1)$ constant auxiliary memory (only the current state variable is stored).

### Q3: What happens if an unexpected input symbol appears?
> **Answer:** The function `execute_mealy_transition` implements a fallback mechanism that preserves the current state and emits an `INFO` diagnostic output without crashing.

### Q4: Does this project cover Context-Free Grammars or Turing Machines?
> **Answer:** No. LogGuard strictly implements **Module I (Regular Languages and Finite Automata with Output)**. It focuses exclusively on deterministic finite state transducers.

---

## 9. Limitations & Scope

1. **Single Account Stream:** The model tracks sequential failed attempts on a per-session / aggregated stream. Multi-tenant tracking would require parallel automata instances per IP/username.
2. **Deterministic Time Windows:** The automaton transitions based on event ordering rather than sliding-window clock delta timers (which would require timed automata).
3. **Syntactic Scope:** Focuses purely on finite state transductions without natural language log parsing or unstructured regex normalization.

