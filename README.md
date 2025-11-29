# FlowMind AI – Intelligent Enterprise Workflow Automation System

**Track:** Enterprise Agents – Kaggle x Google AI Agents Capstone  
**Author:** [vivek kumar]  

---

## 1. Problem:-

Modern enterprises still process a large portion of approvals, routing, and expense workflows manually.  
This leads to:

- Delayed approvals
- Inconsistent decisions and errors
- Limited traceability and analytics
- Lost productivity (hours per employee per week)

Traditional rule-based automation is rigid and hard to adapt to new patterns or policies.

---

## 2. Solution – FlowMind AI:-

FlowMind AI is a **multi-agent AI workflow system** that:

- Understands natural language workflow requests
- Extracts structured fields using a Gemini-powered Analyzer Agent
- Applies configurable business rules + context using a Decision Agent
- Executes actions (auto-approve, route to approver, reject) via an Execution Agent
- Logs and stores all decisions in a Memory Bank for analytics

---

## 3. Architecture:-

### High-Level Flow

1. **User** submits a free-text workflow request (e.g., an expense, approval, or purchase request).
2. **Coordinator Agent** orchestrates the full flow.
3. **Analyzer Agent (Gemini)** parses the text into structured JSON:
   - `amount`, `currency`
   - `category` (TRAVEL, EQUIPMENT, GENERAL, …)
   - `urgency` (NORMAL, HIGH)
4. **Decision Agent** combines:-
   - Parsed request
   - Business rules
   - Historical context (session history)
5. **Execution Agent**:-
   - Auto-approves, routes for review, or rejects.
6. **Session Service & Memory Bank**:-
   - Store complete workflow records
   - Provide history and long-term memory
7. **Observability**:-
   - Logging (Python `logging`)
   - Basic metrics (counts, avg latency)

### Concepts Demonstrated:-

- ✅ Multi-Agent System (Coordinator, Analyzer, Decision, Execution)
- ✅ Sequential Agents
- ✅ Custom Tool (Rule engine)
- ✅ Session & State Management (`InMemorySessionService`)
- ✅ Long-Term Memory (`MemoryBank`)
- ✅ Observability (logging + basic metrics)
- ✅ Gemini usage (Analyzer Agent powered by LLM – simulated + ready for real integration)

---

## 4. Tech Stack:-

- Language: **Python 3**
- LLM: **Google Gemini** (plug-in via `call_gemini_system`)
- Runtime: Local / Replit / Kaggle Notebook
- Architecture: Multi-agent, sequential orchestration, in-memory state

---

## 5. Setup & Run:-

### 5.1 Requirements:-

Create `requirements.txt`:

```txt
google-genai  # or the latest official Gemini Python client

5.2 Configure Gemini (Optional for Real Calls):-
Set environment variable:-
export GEMINI_API_KEY="YOUR_API_KEY".
Then implement the real Gemini call inside call_gemini_system().

5.3 Run Locally:-
python main.py

THANKS......
