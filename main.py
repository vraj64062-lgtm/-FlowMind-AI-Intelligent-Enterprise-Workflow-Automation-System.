# -FlowMind-AI-Intelligent-Enterprise-Workflow-Automation-System.
“FlowMind AI is an autonomous multi-agent workflow automation system that interprets enterprise requests, applies smart decision rules, and executes approvals end-to-end. Powered by LLM-based analysis, rule-driven decisions, session memory, and full observability for scalable enterprise operations.”

"""
FlowMind AI – Intelligent Enterprise Workflow Automation System
Enterprise Agents Track – Kaggle x Google AI Agents Capstone.
"""

import uuid
import time
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
import logging
import random
import os

# ------------------------------------------------------------------------------
# 0. LOGGING & METRICS (Observability)
# ------------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("flowmind")

METRICS = {
    "total_requests": 0,
    "auto_approvals": 0,
    "manual_approvals": 0,
    "avg_latency_ms": [],
}

# ------------------------------------------------------------------------------
# 1. SESSION & MEMORY
# ------------------------------------------------------------------------------

@dataclass
class WorkflowRecord:
    request_id: str
    session_id: str
    raw_text: str
    parsed: Dict[str, Any]
    decision: Dict[str, Any]
    executed_action: Dict[str, Any]
    timestamp: float = field(default_factory=time.time)


class InMemorySessionService:
    """
    Simplified Session & State Management.
    Maps session_id -> list of WorkflowRecord.
    """

    def __init__(self):
        self.sessions: Dict[str, List[WorkflowRecord]] = {}

    def create_session(self) -> str:
        session_id = str(uuid.uuid4())
        self.sessions[session_id] = []
        logger.info(f"Created new session: {session_id}")
        return session_id

    def add_record(self, session_id: str, record: WorkflowRecord):
        if session_id not in self.sessions:
            self.sessions[session_id] = []
        self.sessions[session_id].append(record)
        logger.info(
            f"Session {session_id}: Added record {record.request_id}, "
            f"decision={record.decision.get('decision_type')}"
        )

    def get_history(self, session_id: str) -> List[WorkflowRecord]:
        return self.sessions.get(session_id, [])


class MemoryBank:
    """
    Long-term memory (very simple in-memory store).
    Could later be backed by a DB / vector store.
    """

    def __init__(self):
        self._storage: List[WorkflowRecord] = []

    def persist(self, record: WorkflowRecord):
        self._storage.append(record)
        logger.debug(f"MemoryBank: stored record {record.request_id}")

    def get_recent(self, limit: int = 5) -> List[WorkflowRecord]:
        return self._storage[-limit:]


SESSION_SERVICE = InMemorySessionService()
MEMORY_BANK = MemoryBank()

# ------------------------------------------------------------------------------
# 2. GEMINI / LLM STUB
# ------------------------------------------------------------------------------

def call_gemini_system(prompt: str) -> Dict[str, Any]:
    """
    This function represents a call to Gemini.

    In your actual implementation, replace this with real Gemini code, e.g.:

    from google import genai

    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    response = client.models.generate_content(
        model="gemini-1.5-flash",
        contents=prompt,
    )
    # parse JSON from response.text or similar

    For this prototype we simulate structured output.
    """
    logger.info("Calling Gemini (simulated)...")

    # Very naive simulation: detect amount, type, and urgency from prompt.
    amount = 0
    for token in prompt.replace(",", " ").split():
        if token.replace(".", "").isdigit():
            amount = float(token)
            break

    if "travel" in prompt.lower() or "flight" in prompt.lower():
        category = "TRAVEL"
    elif "laptop" in prompt.lower() or "equipment" in prompt.lower():
        category = "EQUIPMENT"
    else:
        category = "GENERAL"

    urgency = "HIGH" if "urgent" in prompt.lower() else "NORMAL"

    return {
        "amount": amount,
        "currency": "USD",
        "category": category,
        "urgency": urgency,
        "summary": f"Expense request of {amount} {category} ({urgency})",
    }

# ------------------------------------------------------------------------------
# 3. RULES ENGINE (Custom Tool)
# ------------------------------------------------------------------------------

def apply_business_rules(parsed: Dict[str, Any]) -> Dict[str, Any]:
    """
    Simple rule engine for decisions.
    You can extend this with config-driven rules.
    """
    amount = parsed.get("amount", 0)
    category = parsed.get("category", "GENERAL")
    urgency = parsed.get("urgency", "NORMAL")

    decision = {
        "decision_type": None,          # "AUTO_APPROVE" | "REVIEW" | "REJECT"
        "approver_role": None,
        "reason": "",
        "sla_hours": None,
    }

    # Rule 1: small expenses auto-approve
    if amount <= 500:
        decision["decision_type"] = "AUTO_APPROVE"
        decision["approver_role"] = "SYSTEM"
        decision["reason"] = "Amount below auto-approval threshold."
        decision["sla_hours"] = 0
    # Rule 2: medium expenses -> manager review
    elif 500 < amount <= 5000:
        decision["decision_type"] = "REVIEW"
        decision["approver_role"] = "LINE_MANAGER"
        decision["reason"] = "Standard manager review tier."
        decision["sla_hours"] = 24
    # Rule 3: high expenses -> finance review
    elif amount > 5000 and amount <= 20000:
        decision["decision_type"] = "REVIEW"
        decision["approver_role"] = "FINANCE_CONTROLLER"
        decision["reason"] = "High value request, finance approval required."
        decision["sla_hours"] = 48
    else:
        decision["decision_type"] = "REJECT"
        decision["approver_role"] = "SYSTEM"
        decision["reason"] = "Amount exceeds policy limits."
        decision["sla_hours"] = 0

    # Category-specific tweak
    if category == "TRAVEL" and decision["decision_type"] != "REJECT":
        decision["reason"] += " Travel policy rules applied."

    # Urgency tweak
    if urgency == "HIGH" and decision["decision_type"] == "REVIEW":
        decision["sla_hours"] = max(4, decision["sla_hours"] // 2)
        decision["reason"] += " Marked urgent; SLA escalated."

    return decision

# ------------------------------------------------------------------------------
# 4. BASE AGENT CLASS
# ------------------------------------------------------------------------------

class Agent:
    def __init__(self, name: str):
        self.name = name
        self.logger = logging.getLogger(name)

    def run(self, *args, **kwargs):
        raise NotImplementedError

# ------------------------------------------------------------------------------
# 5. INDIVIDUAL AGENTS
# ------------------------------------------------------------------------------

class AnalyzerAgent(Agent):
    """
    LLM-powered agent (Gemini).
    Input: raw text
    Output: structured dict (amount, category, urgency, summary, ...)
    """

    def run(self, request_text: str) -> Dict[str, Any]:
        self.logger.info("Analyzing workflow request...")
        parsed = call_gemini_system(
            f"Analyze this enterprise expense/workflow request and extract "
            f"amount, currency, category, urgency, and summary as JSON:\n\n{request_text}"
        )
        self.logger.info(f"Analyzer output: {parsed}")
        return parsed


class DecisionAgent(Agent):
    """
    Applies business rules and past memory to determine decision.
    """

    def run(self, parsed: Dict[str, Any], history: List[WorkflowRecord]) -> Dict[str, Any]:
        self.logger.info("Deciding on workflow request...")
        decision = apply_business_rules(parsed)

        # Simple "learning": if many similar small rejections, tighten rules etc.
        # Here we just log; you can expand.
        self.logger.debug(f"History length for decision context: {len(history)}")
        return decision


class ExecutionAgent(Agent):
    """
    Executes the decision. In a real system this would:
    - Call APIs (e.g., ticketing, email, Slack, SAP, etc.)
    - Write to a database
    For demo, we simulate.
    """

    def run(self, decision: Dict[str, Any]) -> Dict[str, Any]:
        self.logger.info("Executing decision...")
        decision_type = decision["decision_type"]

        if decision_type == "AUTO_APPROVE":
            METRICS["auto_approvals"] += 1
            action = {
                "status": "APPROVED",
                "performed_by": "FlowMind-System",
                "notes": "Auto-approved by policy.",
            }
        elif decision_type == "REVIEW":
            METRICS["manual_approvals"] += 1
            action = {
                "status": "PENDING_REVIEW",
                "assigned_to_role": decision["approver_role"],
                "notes": "Routed for manual review.",
            }
        else:  # REJECT
            action = {
                "status": "REJECTED",
                "performed_by": "FlowMind-System",
                "notes": "Rejected by policy.",
            }

        self.logger.info(f"Execution result: {action}")
        return action


class CoordinatorAgent(Agent):
    """
    Main orchestrator (Sequential Agents, Multi-Agent System).
    """

    def __init__(self, analyzer: AnalyzerAgent, decision: DecisionAgent,
                 executor: ExecutionAgent, session_service: InMemorySessionService,
                 memory_bank: MemoryBank):
        super().__init__("CoordinatorAgent")
        self.analyzer = analyzer
        self.decision_agent = decision
        self.executor = executor
        self.session_service = session_service
        self.memory_bank = memory_bank

    def run_workflow(self, session_id: str, raw_text: str) -> WorkflowRecord:
        start = time.time()
        METRICS["total_requests"] += 1

        request_id = str(uuid.uuid4())
        self.logger.info(f"--- New Workflow Request: {request_id} ---")
        self.logger.info(f"Raw input: {raw_text}")

        # Step 1: Analyzer Agent (Gemini)
        parsed = self.analyzer.run(raw_text)

        # Step 2: Decision Agent uses history & rules
        history = self.session_service.get_history(session_id)
        decision = self.decision_agent.run(parsed, history)

        # Step 3: Execution Agent
        executed_action = self.executor.run(decision)

        # Step 4: Build record
        record = WorkflowRecord(
            request_id=request_id,
            session_id=session_id,
            raw_text=raw_text,
            parsed=parsed,
            decision=decision,
            executed_action=executed_action,
        )

        # Step 5: Store in session + long-term memory
        self.session_service.add_record(session_id, record)
        self.memory_bank.persist(record)

        latency_ms = (time.time() - start) * 1000
        METRICS["avg_latency_ms"].append(latency_ms)
        self.logger.info(f"Workflow completed in {latency_ms:.2f} ms")
        return record


# ------------------------------------------------------------------------------
# 6. SIMPLE CLI DEMO
# ------------------------------------------------------------------------------

def print_record(record: WorkflowRecord):
    print("\n=== FlowMind AI – Workflow Result ===")
    print(f"Request ID: {record.request_id}")
    print(f"Session ID: {record.session_id}")
    print(f"Raw Input: {record.raw_text}")
    print("\n[Analyzer Output]")
    for k, v in record.parsed.items():
        print(f"  {k}: {v}")

    print("\n[Decision]")
    for k, v in record.decision.items():
        print(f"  {k}: {v}")

    print("\n[Execution]")
    for k, v in record.executed_action.items():
        print(f"  {k}: {v}")
    print("=====================================\n")


def print_metrics():
    print("\n=== FlowMind Metrics (Demo) ===")
    total = METRICS["total_requests"]
    print(f"Total Requests: {total}")
    print(f"Auto Approvals: {METRICS['auto_approvals']}")
    print(f"Manual / Review: {METRICS['manual_approvals']}")
    if METRICS["avg_latency_ms"]:
        avg_lat = sum(METRICS["avg_latency_ms"]) / len(METRICS["avg_latency_ms"])
        print(f"Avg Latency: {avg_lat:.2f} ms")
    print("===============================\n")


def demo_script():
    """
    This function can be used as the core for your YouTube demo.
    It runs multiple sample workflows.
    """
    analyzer = AnalyzerAgent("AnalyzerAgent")
    decision = DecisionAgent("DecisionAgent")
    executor = ExecutionAgent("ExecutionAgent")

    session_id = SESSION_SERVICE.create_session()
    coordinator = CoordinatorAgent(
        analyzer, decision, executor, SESSION_SERVICE, MEMORY_BANK
    )

    sample_requests = [
        "Requesting reimbursement of 120 USD for team lunch. Not urgent.",
        "Urgent: need approval for 420 USD travel expenses for client visit.",
        "Purchase request for 2500 USD laptop equipment for new engineer.",
        "Requesting 7500 USD for conference sponsorship package.",
        "Need budget approval of 35000 USD for office renovation. urgent.",
    ]

    for text in sample_requests:
        record = coordinator.run_workflow(session_id, text)
        print_record(record)
        time.sleep(0.5)

    print_metrics()


if __name__ == "__main__":
    print("=== FlowMind AI – Enterprise Workflow Agent System ===")
    print("1) Run interactive CLI")
    print("2) Run demo script (sample workflows)")
    choice = input("Choose an option (1/2): ").strip()

    analyzer = AnalyzerAgent("AnalyzerAgent")
    decision = DecisionAgent("DecisionAgent")
    executor = ExecutionAgent("ExecutionAgent")

    session_id = SESSION_SERVICE.create_session()
    coordinator = CoordinatorAgent(
        analyzer, decision, executor, SESSION_SERVICE, MEMORY_BANK
    )

    if choice == "2":
        demo_script()
    else:
        while True:
            raw = input("\nEnter workflow/expense description (or 'quit'): ")
            if raw.lower() in ["quit", "exit"]:
                break
            record = coordinator.run_workflow(session_id, raw)
            print_record(record)

        print_metrics()
