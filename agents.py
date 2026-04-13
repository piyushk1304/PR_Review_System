# agents.py
"""
agents.py — Multi-agent PR Reviewer
Uses Azure OpenAI with 7 explicit PR data inputs per agent.
"""
import os
import time

from dotenv import load_dotenv
load_dotenv()

from logger_config import logger

from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

try:
    from state import PRReviewerState
except ImportError:
    from typing import TypedDict
    class PRReviewerState(TypedDict, total=False):
        fmt_title: str
        fmt_description: str
        fmt_commits: str
        fmt_files: str
        fmt_inline_comments: str
        fmt_discussion: str
        fmt_diff: str
        intent_report: str
        security_report: str
        perf_report: str
        coverage_report: str
        suggestion_report: str
        pr_title: str


# ---------------------------------------------------------------------------
# Azure OpenAI
# ---------------------------------------------------------------------------
_ENDPOINT        = os.getenv("AZURE_OPENAI_ENDPOINT")
_API_KEY         = os.getenv("AZURE_OPENAI_API_KEY")
_API_VERSION     = os.getenv("AZURE_OPENAI_API_VERSION")
_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")

logger.info("=" * 60)
logger.info("🤖 Initialising Azure OpenAI LLM")
logger.info("   Endpoint   : {}", _ENDPOINT)
logger.info("   Deployment : {}", _DEPLOYMENT_NAME)
logger.info("   API Version: {}", _API_VERSION)

_llm = AzureChatOpenAI(
    azure_endpoint=_ENDPOINT,
    api_key=_API_KEY,
    api_version=_API_VERSION,
    azure_deployment=_DEPLOYMENT_NAME,
    temperature=0,
    max_retries=3,
)

logger.info("✅ Azure OpenAI LLM ready")
logger.info("=" * 60)


# ---------------------------------------------------------------------------
# Shared Human Template — 7 explicit inputs
# ---------------------------------------------------------------------------
_HUMAN_TEMPLATE = """
INPUT 1 — PR TITLE & METADATA
{fmt_title}

INPUT 2 — PR DESCRIPTION
{fmt_description}

INPUT 3 — COMMITS
{fmt_commits}

INPUT 4 — CHANGED FILES
{fmt_files}

INPUT 5 — INLINE CODE REVIEW COMMENTS
{fmt_inline_comments}

INPUT 6 — PR DISCUSSION COMMENTS
{fmt_discussion}

INPUT 7 — GIT DIFF
{fmt_diff}
"""


def _get_inputs(state: PRReviewerState) -> dict[str, str]:
    return {
        "fmt_title":           state.get("fmt_title",           ""),
        "fmt_description":     state.get("fmt_description",     ""),
        "fmt_commits":         state.get("fmt_commits",         ""),
        "fmt_files":           state.get("fmt_files",           ""),
        "fmt_inline_comments": state.get("fmt_inline_comments", ""),
        "fmt_discussion":      state.get("fmt_discussion",      ""),
        "fmt_diff":            state.get("fmt_diff",            ""),
    }


# ---------------------------------------------------------------------------
# Shared agent logging helpers
# ---------------------------------------------------------------------------
def _agent_start(name: str, icon: str) -> float:
    logger.info("=" * 60)
    logger.info("{} AGENT START | {}", icon, name)
    logger.info("=" * 60)
    return time.perf_counter()


def _agent_inputs(name: str, inputs: dict) -> None:
    logger.info("   📥 Inputs to {}:", name)
    for key, val in inputs.items():
        logger.info("      {:<24} : {:>6} chars", key, len(val))


def _agent_done(name: str, icon: str, start: float, report: str) -> None:
    elapsed = round(time.perf_counter() - start, 2)
    logger.info("─" * 60)
    logger.info("{} AGENT DONE  | {} | ⏱ {}s", icon, name, elapsed)
    logger.info("   Output : {} chars | {} lines",
                len(report), len(report.splitlines()))
    logger.info("─" * 60)
    logger.info("📋 {} REPORT:", name)
    logger.info("─" * 60)
    for line in report.splitlines():
        logger.info("   {}", line)
    logger.info("─" * 60)


def _agent_error(name: str, icon: str, start: float, error: Exception) -> None:
    elapsed = round(time.perf_counter() - start, 2)
    logger.error("=" * 60)
    logger.error("{} AGENT FAILED | {} | ⏱ {}s", icon, name, elapsed)
    logger.error("   Error : {}", error)
    logger.error("=" * 60)


# ===========================================================================
# Agent 1 — Vulnerability Scanner
# ===========================================================================
_SECURITY_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a security-focused code reviewer specializing in OWASP Top 10. "
        "You are given 7 inputs about a GitHub Pull Request: title, description, "
        "commits, changed files, inline review comments, PR discussion, and git diff.\n\n"
        "Your task: Review ALL inputs for security vulnerabilities.\n\n"
        "Check for:\n"
        "- OWASP Top 10 vulnerabilities (injection, XSS, CSRF, broken auth, etc.)\n"
        "- Hardcoded secrets, API keys, passwords, or tokens\n"
        "- SQL injection patterns\n"
        "- Insecure dependencies or imports\n"
        "- Insecure deserialization\n"
        "- Missing input validation or sanitization\n"
        "- Sensitive data exposure in logs, responses, or comments\n"
        "- Auth or permission checks missing on new endpoints\n\n"
        "Use the description and commits to understand intent. "
        "Use the diff and files as the primary source of findings. "
        "Use inline comments and discussion to check if issues were already raised.\n\n"
        "For each finding include:\n"
        "  - Severity: High / Medium / Low\n"
        "  - File and line reference (from diff)\n"
        "  - Clear description of the vulnerability\n"
        "  - Recommended fix\n\n"
        "If no issues are found, say so explicitly. Be concise and precise.",
    ),
    ("human", _HUMAN_TEMPLATE),
])


def agent_vulnerability_scanner(state: PRReviewerState) -> dict:
    """Agent 1: Scan all PR data for security vulnerabilities."""
    start  = _agent_start("Vulnerability Scanner", "🔴")
    inputs = _get_inputs(state)
    _agent_inputs("Vulnerability Scanner", inputs)

    try:
        logger.info("   🔍 Calling LLM — OWASP Top 10, secrets, injection...")
        res    = (_SECURITY_PROMPT | _llm).invoke(inputs)
        report = res.content.strip()

        high   = report.lower().count("high")
        medium = report.lower().count("medium")
        low    = report.lower().count("low")
        logger.info("   🔴 Findings | High: {} | Medium: {} | Low: {}",
                    high, medium, low)

        _agent_done("Vulnerability Scanner", "🔴", start, report)
        return {"security_report": report}

    except Exception as e:
        _agent_error("Vulnerability Scanner", "🔴", start, e)
        return {"security_report": f"Security agent error: {e}"}


# ===========================================================================
# Agent 2 — Performance Reviewer
# ===========================================================================
_PERF_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a performance-focused code reviewer. "
        "You are given 7 inputs about a GitHub Pull Request: title, description, "
        "commits, changed files, inline review comments, PR discussion, and git diff.\n\n"
        "Your task: Review ALL inputs for performance issues.\n\n"
        "Check for:\n"
        "- N+1 query patterns\n"
        "- Unnecessary nested loops (O(n²) or worse where avoidable)\n"
        "- Missing database index hints or full-table scans\n"
        "- Unbounded data structures (growing lists/dicts with no size limit)\n"
        "- Synchronous blocking calls where async would significantly help\n"
        "- Repeated expensive computations that could be cached or memoized\n"
        "- Large payloads loaded fully into memory unnecessarily\n"
        "- Missing pagination on queries that may return large result sets\n\n"
        "Use the description and commits to understand intent. "
        "Use the diff and files as the primary source of findings. "
        "Use inline comments and discussion to check if issues were already raised.\n\n"
        "For each finding include:\n"
        "  - Severity: High / Medium / Low\n"
        "  - File and line reference (from diff)\n"
        "  - Description of the performance issue\n"
        "  - Recommended fix\n\n"
        "If no issues are found, say so explicitly. Be concise and precise.",
    ),
    ("human", _HUMAN_TEMPLATE),
])


def agent_performance_reviewer(state: PRReviewerState) -> dict:
    """Agent 2: Flag performance issues across all PR data."""
    start  = _agent_start("Performance Reviewer", "🟡")
    inputs = _get_inputs(state)
    _agent_inputs("Performance Reviewer", inputs)

    try:
        logger.info("   ⚡ Calling LLM — N+1, loops, memory, async...")
        res    = (_PERF_PROMPT | _llm).invoke(inputs)
        report = res.content.strip()

        high   = report.lower().count("high")
        medium = report.lower().count("medium")
        low    = report.lower().count("low")
        logger.info("   🟡 Findings | High: {} | Medium: {} | Low: {}",
                    high, medium, low)

        _agent_done("Performance Reviewer", "🟡", start, report)
        return {"perf_report": report}

    except Exception as e:
        _agent_error("Performance Reviewer", "🟡", start, e)
        return {"perf_report": f"Performance agent error: {e}"}


# ===========================================================================
# Agent 3 — Test Coverage Auditor
# ===========================================================================
_COVERAGE_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a test coverage auditor. "
        "You are given 7 inputs about a GitHub Pull Request: title, description, "
        "commits, changed files, inline review comments, PR discussion, and git diff.\n\n"
        "Your task: Review ALL inputs for missing or inadequate test coverage.\n\n"
        "Check for:\n"
        "- New functions, classes, or methods with no associated tests\n"
        "- Missing edge cases: null/None inputs, empty collections, boundary values\n"
        "- External calls (HTTP, DB, filesystem, queue) that are not mocked in tests\n"
        "- Error paths and exception handling that are untested\n"
        "- Tests that exist but do not assert meaningful outcomes\n"
        "- New API endpoints or routes with no integration tests\n"
        "- Business logic changes with no corresponding test updates\n\n"
        "Use the description and commits to understand what was intended to change. "
        "Use the diff and files as the primary source of coverage gaps. "
        "Use inline comments and discussion to check if coverage was already discussed.\n\n"
        "For each finding include:\n"
        "  - Severity: High / Medium / Low\n"
        "  - What is missing (function, class, or scenario)\n"
        "  - What specific test cases should be added\n\n"
        "If coverage looks adequate, say so explicitly. Be concise and precise.",
    ),
    ("human", _HUMAN_TEMPLATE),
])


def agent_coverage_auditor(state: PRReviewerState) -> dict:
    """Agent 3: Identify test coverage gaps across all PR data."""
    start  = _agent_start("Coverage Auditor", "🟢")
    inputs = _get_inputs(state)
    _agent_inputs("Coverage Auditor", inputs)

    try:
        logger.info("   🧪 Calling LLM — test coverage gaps...")
        res    = (_COVERAGE_PROMPT | _llm).invoke(inputs)
        report = res.content.strip()

        high   = report.lower().count("high")
        medium = report.lower().count("medium")
        low    = report.lower().count("low")
        logger.info("   🟢 Findings | High: {} | Medium: {} | Low: {}",
                    high, medium, low)

        _agent_done("Coverage Auditor", "🟢", start, report)
        return {"coverage_report": report}

    except Exception as e:
        _agent_error("Coverage Auditor", "🟢", start, e)
        return {"coverage_report": f"Coverage agent error: {e}"}


# ===========================================================================
# Agent 4 — Code Suggestion Agent
# ===========================================================================
_SUGGESTION_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a senior software engineer specializing in code quality and improvement. "
        "You are given 7 inputs about a GitHub Pull Request: title, description, "
        "commits, changed files, inline review comments, PR discussion, and git diff.\n\n"
        "Your task: Provide concrete, actionable code improvement suggestions.\n\n"
        "Focus on:\n"
        "- Readability and clarity improvements\n"
        "- Better naming for variables, functions, and classes\n"
        "- Simplification of overly complex logic\n"
        "- Code duplication that could be extracted into shared utilities\n"
        "- Missing or misleading docstrings and comments\n"
        "- Pythonic improvements (or language-idiomatic equivalents)\n"
        "- Better error handling patterns\n"
        "- Type hints or annotations that are missing or incorrect\n\n"
        "Use the description and commits to understand intent. "
        "Use the diff and files as the primary source. "
        "Check inline comments and discussion to avoid repeating already-raised suggestions.\n\n"
        "For each suggestion:\n"
        "  - Cite the exact file and line from the diff\n"
        "  - Show the current code block\n"
        "  - Show the suggested replacement\n"
        "  - Explain why the change improves the code\n\n"
        "Do not repeat findings already covered by security or performance review. "
        "Be specific, not generic. Prioritize the most impactful suggestions.",
    ),
    ("human", _HUMAN_TEMPLATE),
])


def agent_code_suggestion(state: PRReviewerState) -> dict:
    """Agent 4: Provide code improvement suggestions across all PR data."""
    start  = _agent_start("Code Suggestion", "🟣")
    inputs = _get_inputs(state)
    _agent_inputs("Code Suggestion", inputs)

    try:
        logger.info("   💡 Calling LLM — code improvement suggestions...")
        res    = (_SUGGESTION_PROMPT | _llm).invoke(inputs)
        report = res.content.strip()

        count = report.count("Suggestion")
        logger.info("   🟣 Suggestions found : ~{}", count)

        _agent_done("Code Suggestion", "🟣", start, report)
        return {"suggestion_report": report}

    except Exception as e:
        _agent_error("Code Suggestion", "🟣", start, e)
        return {"suggestion_report": f"Suggestion agent error: {e}"}


# ===========================================================================
# Agent 5 — Intent Verifier
# ===========================================================================
_INTENT_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a PR intent verification specialist. "
        "You are given 7 inputs about a GitHub Pull Request: title, description, "
        "commits, changed files, inline review comments, PR discussion, and git diff.\n\n"
        "Your task: Verify that the PR implementation matches its stated intent.\n\n"
        "Check:\n"
        "1. Description vs Implementation\n"
        "   - Does the code actually do what the description claims?\n"
        "   - Are there features described but not implemented?\n"
        "   - Are there code changes not mentioned in the description?\n\n"
        "2. Commit Messages vs Code\n"
        "   - Do commit messages accurately reflect the actual code changes?\n"
        "   - Are there misleading or vague commit messages?\n\n"
        "3. Linked Issues vs Changes\n"
        "   - If issues are linked (Closes #N), does the code actually resolve them?\n\n"
        "4. Reviewer Requests vs Implementation\n"
        "   - Were inline review comments addressed in the diff?\n"
        "   - Were discussion decisions actually implemented?\n\n"
        "Score the PR intent alignment:\n"
        "  PASS    — implementation fully matches stated intent\n"
        "  PARTIAL — implementation partially matches; gaps noted\n"
        "  FAIL    — significant mismatch between intent and implementation\n\n"
        "Be specific. Cite the description claim and then the diff evidence. "
        "Be concise and factual.",
    ),
    ("human", _HUMAN_TEMPLATE),
])


def agent_intent_verifier(state: PRReviewerState) -> dict:
    """Agent 5: Verify PR intent matches implementation across all inputs."""
    start  = _agent_start("Intent Verifier", "🟠")
    inputs = _get_inputs(state)
    _agent_inputs("Intent Verifier", inputs)

    try:
        logger.info("   🎯 Calling LLM — description vs implementation...")
        res    = (_INTENT_PROMPT | _llm).invoke(inputs)
        report = res.content.strip()

        if "FAIL" in report:
            verdict = "FAIL ❌"
        elif "PARTIAL" in report:
            verdict = "PARTIAL ⚠️"
        elif "PASS" in report:
            verdict = "PASS ✅"
        else:
            verdict = "UNKNOWN"

        logger.info("   🟠 Intent Verdict : {}", verdict)

        _agent_done("Intent Verifier", "🟠", start, report)
        return {"intent_report": report}

    except Exception as e:
        _agent_error("Intent Verifier", "🟠", start, e)
        return {"intent_report": f"Intent agent error: {e}"}


# ===========================================================================
# Agent 6 — Lead Reviewer
# ===========================================================================
_LEAD_HUMAN_TEMPLATE = _HUMAN_TEMPLATE + """
INTENT VERIFICATION REPORT:
{intent_report}

SECURITY REPORT:
{security_report}

PERFORMANCE REPORT:
{perf_report}

TEST COVERAGE REPORT:
{coverage_report}

CODE SUGGESTION REPORT:
{suggestion_report}
"""

_LEAD_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a lead code reviewer synthesizing findings from five specialist agents:\n"
        "  1. Intent Verifier\n"
        "  2. Vulnerability Scanner\n"
        "  3. Performance Reviewer\n"
        "  4. Test Coverage Auditor\n"
        "  5. Code Suggestion Agent\n\n"
        "You also have full access to all 7 PR inputs: title, description, commits, "
        "changed files, inline comments, discussion, and git diff.\n\n"
        "Your tasks:\n"
        "1. Merge duplicate findings across all five reports (same issue reported by "
        "   multiple agents → keep once, note it was flagged by multiple reviewers).\n"
        "2. Write a single GitHub-flavored Markdown PR review comment.\n"
        "3. Structure the comment with these sections:\n"
        "   ## 🎯 Intent Verification\n"
        "   ## 🔴 High Severity\n"
        "   ## 🟡 Medium Severity\n"
        "   ## 🟢 Low Severity / Suggestions\n"
        "   ## ✅ Summary\n"
        "4. Start with an overall verdict on the very first line:\n"
        "   **Verdict: Approve ✅** — no blocking issues\n"
        "   **Verdict: Request Changes ❌** — blocking issues found\n"
        "   **Verdict: Comment 💬** — minor issues only\n"
        "5. In the Summary section include:\n"
        "   - Total issues by severity\n"
        "   - Intent verdict (PASS / PARTIAL / FAIL)\n"
        "   - Top 3 most critical actions for the author\n"
        "6. Keep it actionable, developer-friendly, and free of waffle.\n\n"
        "Output ONLY the Markdown comment text. No preamble.",
    ),
    ("human", _LEAD_HUMAN_TEMPLATE),
])


def agent_lead_reviewer(state: PRReviewerState) -> dict:
    """Agent 6: Synthesize all specialist reports into a final PR review comment."""
    start  = _agent_start("Lead Reviewer", "🔵")

    inputs = _get_inputs(state)
    inputs.update({
        "intent_report":     state.get("intent_report",     "No intent report available."),
        "security_report":   state.get("security_report",   "No security report available."),
        "perf_report":       state.get("perf_report",       "No performance report available."),
        "coverage_report":   state.get("coverage_report",   "No coverage report available."),
        "suggestion_report": state.get("suggestion_report", "No suggestion report available."),
    })

    logger.info("   📥 All inputs to Lead Reviewer:")
    for key, val in inputs.items():
        logger.info("      {:<24} : {:>6} chars", key, len(str(val)))

    try:
        logger.info("   📝 Calling LLM — synthesising 5 reports...")
        res    = (_LEAD_PROMPT | _llm).invoke(inputs)
        report = res.content.strip()

        if "Approve ✅" in report:
            verdict = "APPROVE ✅"
        elif "Request Changes ❌" in report:
            verdict = "REQUEST CHANGES ❌"
        elif "Comment 💬" in report:
            verdict = "COMMENT 💬"
        else:
            verdict = "UNKNOWN"

        logger.info("   🔵 Final Verdict : {}", verdict)

        _agent_done("Lead Reviewer", "🔵", start, report)
        return {"final_comment": report}

    except Exception as e:
        _agent_error("Lead Reviewer", "🔵", start, e)
        return {"final_comment": f"Lead reviewer error: {e}"}