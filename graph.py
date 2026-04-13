# graph.py
"""
graph.py — LangGraph pipeline
Fetches ALL GitHub PR data, runs agents, then posts
the review back to GitHub (inline comments + summary).
"""
import re
import os
import time

import requests
from langgraph.graph import StateGraph, END

from logger_config import logger
from state import PRReviewerState
from agents import (
    agent_vulnerability_scanner,
    agent_performance_reviewer,
    agent_coverage_auditor,
    agent_code_suggestion,
    agent_intent_verifier,
    agent_lead_reviewer,
)

_GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")


def _github_headers(accept: str = "application/vnd.github.v3+json") -> dict:
    headers = {"Accept": accept}
    if _GITHUB_TOKEN:
        headers["Authorization"] = f"token {_GITHUB_TOKEN}"
    return headers


# ===========================================================================
# FETCHERS
# ===========================================================================

def _fetch_pr_metadata(base_api: str) -> dict:
    logger.info("   📋 [1/6] Fetching PR metadata...")
    resp = requests.get(base_api, headers=_github_headers(), timeout=15)
    resp.raise_for_status()
    meta = resp.json()
    body = meta.get("body") or "No description provided."
    linked_issues = re.findall(
        r"(?:closes?|fixes?|resolves?)\s+#(\d+)", body, re.IGNORECASE
    )
    result = {
        "title":         meta.get("title", "No title"),
        "body":          body,
        "author":        meta.get("user", {}).get("login", "unknown"),
        "branch":        meta.get("head", {}).get("ref", "unknown"),
        "base_branch":   meta.get("base", {}).get("ref", "main"),
        "state":         meta.get("state", "unknown"),
        "merged":        meta.get("merged", False),
        "labels":        [l["name"] for l in meta.get("labels", [])],
        "milestone":     (meta.get("milestone") or {}).get("title", "none"),
        "reviewers":     [r["login"] for r in meta.get("requested_reviewers", [])],
        "changed_files": meta.get("changed_files", 0),
        "additions":     meta.get("additions", 0),
        "deletions":     meta.get("deletions", 0),
        "commits":       meta.get("commits", 0),
        "linked_issues": linked_issues,
        "head_sha":      meta.get("head", {}).get("sha", ""),
    }
    logger.info("   ✅ [1/6] Metadata fetched")
    logger.info("         Title   : {}", result["title"])
    logger.info("         Author  : {}", result["author"])
    logger.info("         Branch  : {} → {}", result["branch"], result["base_branch"])
    logger.info("         SHA     : {}", result["head_sha"][:7])
    logger.info("         Stats   : {} files | +{} -{} lines | {} commits",
                result["changed_files"], result["additions"],
                result["deletions"], result["commits"])
    logger.info("         Issues  : {}", result["linked_issues"] or "none")
    return result


def _fetch_commits(base_api: str) -> list[dict]:
    logger.info("   📝 [2/6] Fetching commits...")
    resp = requests.get(
        f"{base_api}/commits", headers=_github_headers(), timeout=15
    )
    resp.raise_for_status()
    commits = []
    for c in resp.json():
        commits.append({
            "sha":     c["sha"][:7],
            "message": c["commit"]["message"].strip(),
            "author":  c["commit"]["author"]["name"],
            "date":    c["commit"]["author"]["date"],
        })
    logger.info("   ✅ [2/6] Commits | Count: {}", len(commits))
    for i, c in enumerate(commits, 1):
        logger.info("         {}. [{}] {} — {}",
                    i, c["sha"], c["message"][:55], c["author"])
    return commits


def _fetch_review_comments(base_api: str) -> list[dict]:
    logger.info("   💬 [3/6] Fetching inline review comments...")
    resp = requests.get(
        f"{base_api}/comments", headers=_github_headers(), timeout=15
    )
    resp.raise_for_status()
    comments = []
    for c in resp.json():
        comments.append({
            "author":    c["user"]["login"],
            "file":      c.get("path", "unknown"),
            "line":      c.get("line") or c.get("original_line", "?"),
            "body":      c["body"].strip(),
            "date":      c["created_at"],
            "diff_hunk": c.get("diff_hunk", ""),
        })
    logger.info("   ✅ [3/6] Inline comments | Count: {}", len(comments))
    for c in comments:
        logger.info("         [{}:{}] {} — {}",
                    c["file"], c["line"], c["author"], c["body"][:50])
    return comments


def _fetch_issue_comments(owner: str, repo: str, pr_number: str) -> list[dict]:
    logger.info("   🗨️  [4/6] Fetching PR discussion...")
    url = (
        f"https://api.github.com/repos/{owner}/{repo}"
        f"/issues/{pr_number}/comments"
    )
    resp = requests.get(url, headers=_github_headers(), timeout=15)
    resp.raise_for_status()
    comments = []
    for c in resp.json():
        comments.append({
            "author": c["user"]["login"],
            "body":   c["body"].strip(),
            "date":   c["created_at"],
        })
    logger.info("   ✅ [4/6] Discussion | Count: {}", len(comments))
    for c in comments:
        logger.info("         {} — {}", c["author"], c["body"][:60])
    return comments


def _fetch_changed_files(base_api: str) -> list[dict]:
    logger.info("   📂 [5/6] Fetching changed files...")
    resp = requests.get(
        f"{base_api}/files", headers=_github_headers(), timeout=15
    )
    resp.raise_for_status()
    files = []
    for f in resp.json():
        files.append({
            "filename":  f["filename"],
            "status":    f["status"],
            "additions": f["additions"],
            "deletions": f["deletions"],
            "changes":   f["changes"],
            "patch":     f.get("patch", "binary or too large"),
        })
    logger.info("   ✅ [5/6] Files | Count: {}", len(files))
    for f in files:
        logger.info("         [{}] {} +{} -{}",
                    f["status"].upper()[:3], f["filename"],
                    f["additions"], f["deletions"])
    return files


def _fetch_raw_diff(base_api: str) -> str:
    logger.info("   🔀 [6/6] Fetching raw git diff...")
    resp = requests.get(
        base_api,
        headers=_github_headers("application/vnd.github.v3.diff"),
        timeout=30,
    )
    resp.raise_for_status()
    diff = resp.text.strip()
    logger.info("   ✅ [6/6] Diff | {} chars | {} lines",
                len(diff), len(diff.splitlines()))
    return diff


# ===========================================================================
# FORMATTERS
# ===========================================================================

def _fmt_title(meta: dict) -> str:
    labels = ", ".join(meta["labels"]) if meta["labels"] else "none"
    issues = (
        ", ".join(f"#{i}" for i in meta["linked_issues"])
        if meta["linked_issues"] else "none"
    )
    return (
        f"PR Title     : {meta['title']}\n"
        f"Author       : {meta['author']}\n"
        f"Branch       : {meta['branch']} → {meta['base_branch']}\n"
        f"State        : {meta['state']}"
        f"{' (merged)' if meta['merged'] else ''}\n"
        f"Labels       : {labels}\n"
        f"Milestone    : {meta['milestone']}\n"
        f"Linked Issues: {issues}\n"
        f"Stats        : {meta['changed_files']} files "
        f"| +{meta['additions']} -{meta['deletions']} lines "
        f"| {meta['commits']} commits"
    )


def _fmt_description(body: str) -> str:
    return body.strip() if body.strip() else "No description provided."


def _fmt_commits(commits: list[dict]) -> str:
    if not commits:
        return "No commits available."
    lines = []
    for i, c in enumerate(commits, 1):
        lines.append(
            f"{i}. [{c['sha']}] {c['message']}\n"
            f"   Author : {c['author']}\n"
            f"   Date   : {c['date'][:10]}"
        )
    return "\n\n".join(lines)


def _fmt_files(changed_files: list[dict]) -> str:
    if not changed_files:
        return "No changed files available."
    lines = []
    for f in changed_files:
        lines.append(
            f"File   : {f['filename']}\n"
            f"Status : {f['status'].upper()}\n"
            f"Changes: +{f['additions']} -{f['deletions']} lines"
        )
    return "\n\n".join(lines)


def _fmt_inline_comments(review_comments: list[dict]) -> str:
    if not review_comments:
        return "No inline code review comments."
    lines = []
    for c in review_comments:
        lines.append(
            f"Reviewer : {c['author']}\n"
            f"File     : {c['file']}\n"
            f"Line     : {c['line']}\n"
            f"Date     : {c['date'][:10]}\n"
            f"Comment  : {c['body']}\n"
            f"Code Context:\n{c['diff_hunk']}"
        )
    return "\n\n---\n\n".join(lines)


def _fmt_discussion(issue_comments: list[dict]) -> str:
    if not issue_comments:
        return "No PR discussion comments."
    lines = []
    for c in issue_comments:
        lines.append(
            f"Author  : {c['author']}\n"
            f"Date    : {c['date'][:10]}\n"
            f"Comment : {c['body']}"
        )
    return "\n\n---\n\n".join(lines)


# ===========================================================================
# INLINE COMMENT PARSER
# Parse agent reports and extract file + line + comment triples
# ===========================================================================

def _parse_inline_comments(
    changed_files: list[dict],
    security_report: str,
    perf_report: str,
    coverage_report: str,
    suggestion_report: str,
) -> list[dict]:
    """
    Extract inline comments from agent reports.
    Match file references in reports against actual changed files.
    Returns list of {path, line, body} dicts for GitHub Review API.
    """
    logger.info("   🔍 Parsing agent reports for inline comments...")

    inline_comments = []
    valid_files     = {f["filename"] for f in changed_files}
    file_patches    = {f["filename"]: f.get("patch", "") for f in changed_files}

    # Each report section with its icon prefix
    report_sections = [
        ("🔴 Security",     security_report),
        ("🟡 Performance",  perf_report),
        ("🟢 Coverage",     coverage_report),
        ("🟣 Suggestion",   suggestion_report),
    ]

    for section_name, report in report_sections:
        if not report:
            continue

        # Split report into individual findings by common separators
        # Look for patterns like "File: x.py" or "`x.py`" or "in x.py"
        findings = _split_into_findings(report)

        for finding in findings:
            matched_file = _match_file(finding, valid_files)
            if not matched_file:
                continue

            line_no = _extract_line_number(finding, file_patches.get(matched_file, ""))
            if not line_no:
                continue

            # Clean up the finding text for the comment body
            body = _format_comment_body(section_name, finding)

            inline_comments.append({
                "path": matched_file,
                "line": line_no,
                "body": body,
            })

            logger.info("      📌 Inline comment | {}:{} | {}",
                        matched_file, line_no, section_name)

    logger.info("   ✅ Inline comments parsed | Count: {}", len(inline_comments))
    return inline_comments


def _split_into_findings(report: str) -> list[str]:
    """Split a report into individual finding blocks."""
    # Split on numbered items, severity markers, or blank lines
    import re
    blocks = re.split(
        r'\n(?=\d+\.\s|###|##\s|---|\*\*(?:High|Medium|Low|Severity))',
        report,
    )
    # Filter out very short blocks
    return [b.strip() for b in blocks if len(b.strip()) > 30]


def _match_file(finding: str, valid_files: set) -> str | None:
    """Find a valid changed file referenced in a finding."""
    finding_lower = finding.lower()
    # Sort by length descending so longer/more specific paths match first
    for filepath in sorted(valid_files, key=len, reverse=True):
        filename   = filepath.split("/")[-1]   # basename
        if filename.lower() in finding_lower or filepath.lower() in finding_lower:
            return filepath
    return None


def _extract_line_number(finding: str, patch: str) -> int | None:
    """
    Extract a valid line number from finding text.
    Cross-reference against the actual patch to ensure the line exists.
    """
    import re

    # Try to extract explicit line reference from finding text
    patterns = [
        r'line[s]?\s*:?\s*(\d+)',
        r'line[s]?\s+(\d+)',
        r':(\d+)',
        r'\((\d+)\)',
        r'L(\d+)',
    ]

    candidate = None
    for pat in patterns:
        m = re.search(pat, finding, re.IGNORECASE)
        if m:
            candidate = int(m.group(1))
            break

    if not patch:
        return candidate if candidate else None

    # Extract valid line numbers from patch hunks
    valid_lines = _get_patch_lines(patch)
    if not valid_lines:
        return None

    if candidate and candidate in valid_lines:
        return candidate

    # If no valid match, return the last added line in the patch
    return valid_lines[-1] if valid_lines else None


def _get_patch_lines(patch: str) -> list[int]:
    """
    Parse a git patch and return list of valid line numbers
    that can be commented on (added/context lines only).
    """
    import re
    valid_lines = []
    current_line = 0

    for raw_line in patch.splitlines():
        hunk = re.match(r'^@@ -\d+(?:,\d+)? \+(\d+)(?:,\d+)? @@', raw_line)
        if hunk:
            current_line = int(hunk.group(1)) - 1
            continue

        if raw_line.startswith("-"):
            # Deleted lines — not commentable
            continue
        elif raw_line.startswith("+") or not raw_line.startswith("\\"):
            current_line += 1
            if current_line > 0:
                valid_lines.append(current_line)

    return valid_lines


def _format_comment_body(section_name: str, finding: str) -> str:
    """Format a finding as a clean GitHub comment body."""
    # Trim very long findings
    max_chars = 600
    text = finding.strip()
    if len(text) > max_chars:
        text = text[:max_chars] + "...\n\n_[truncated — see full report in PR summary]_"
    return f"**{section_name}**\n\n{text}"


# ===========================================================================
# GITHUB REVIEW POSTER
# ===========================================================================

def _post_github_review(
    owner:          str,
    repo:           str,
    pr_number:      str,
    head_sha:       str,
    final_comment:  str,
    inline_comments: list[dict],
    verdict:        str,
) -> dict:
    """
    Post a full GitHub PR review with:
    - Summary comment (final_comment)
    - Inline comments on specific lines
    - Verdict (APPROVE / REQUEST_CHANGES / COMMENT)
    """
    logger.info("=" * 60)
    logger.info("📮 POSTING GITHUB REVIEW")
    logger.info("   Owner    : {}", owner)
    logger.info("   Repo     : {}", repo)
    logger.info("   PR#      : {}", pr_number)
    logger.info("   SHA      : {}", head_sha[:7])
    logger.info("   Verdict  : {}", verdict)
    logger.info("   Inline   : {} comments", len(inline_comments))
    logger.info("=" * 60)

    # Map verdict to GitHub event
    event_map = {
        "APPROVE":          "APPROVE",
        "REQUEST CHANGES":  "REQUEST_CHANGES",
        "COMMENT":          "COMMENT",
    }
    event = event_map.get(verdict, "COMMENT")

    url = (
        f"https://api.github.com/repos/{owner}/{repo}"
        f"/pulls/{pr_number}/reviews"
    )

    payload = {
        "commit_id": head_sha,
        "body":      final_comment,
        "event":     event,
        "comments":  inline_comments,
    }

    logger.info("   📤 Sending review to GitHub API...")
    logger.info("      URL    : {}", url)
    logger.info("      Event  : {}", event)
    logger.info("      Body   : {} chars", len(final_comment))
    logger.info("      Comments: {}", len(inline_comments))

    try:
        resp = requests.post(
            url,
            headers=_github_headers(),
            json=payload,
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()

        review_id  = data.get("id")
        review_url = data.get("html_url", "")

        logger.info("   ✅ GitHub review posted successfully!")
        logger.info("      Review ID  : {}", review_id)
        logger.info("      Review URL : {}", review_url)
        logger.info("      Event      : {}", event)
        logger.info("=" * 60)

        return {
            "github_review_id":     review_id,
            "github_review_url":    review_url,
            "github_review_posted": True,
        }

    except requests.HTTPError as e:
        logger.error("   ❌ GitHub API error: {}", e)
        logger.error("      Status : {}", e.response.status_code)
        logger.error("      Body   : {}", e.response.text[:300])
        logger.error("=" * 60)
        return {
            "github_review_id":     None,
            "github_review_url":    "",
            "github_review_posted": False,
        }

    except Exception as e:
        logger.error("   ❌ Unexpected error posting review: {}", e)
        logger.error("=" * 60)
        return {
            "github_review_id":     None,
            "github_review_url":    "",
            "github_review_posted": False,
        }


# ===========================================================================
# MAIN NODE — fetch_diff
# ===========================================================================

def fetch_diff(state: PRReviewerState) -> dict:
    logger.info("=" * 60)
    logger.info("📥 STEP 1 START | Fetch Full PR Context")
    logger.info("   PR URL : {}", state["pr_url"])
    logger.info("=" * 60)

    start  = time.perf_counter()
    pr_url = state["pr_url"]

    match = re.match(
        r"https://github\.com/([^/]+)/([^/]+)/pull/(\d+)", pr_url
    )
    if not match:
        raise ValueError(f"Invalid GitHub PR URL: {pr_url}")

    owner     = match.group(1)
    repo      = match.group(2)
    pr_number = match.group(3)

    logger.info("   Owner : {} | Repo : {} | PR# : {}", owner, repo, pr_number)

    base_api = (
        f"https://api.github.com/repos/{owner}/{repo}/pulls/{pr_number}"
    )

    meta            = _fetch_pr_metadata(base_api)
    commits         = _fetch_commits(base_api)
    review_comments = _fetch_review_comments(base_api)
    issue_comments  = _fetch_issue_comments(owner, repo, pr_number)
    changed_files   = _fetch_changed_files(base_api)
    raw_diff        = _fetch_raw_diff(base_api)

    fmt_title           = _fmt_title(meta)
    fmt_description     = _fmt_description(meta["body"])
    fmt_commits         = _fmt_commits(commits)
    fmt_files           = _fmt_files(changed_files)
    fmt_inline_comments = _fmt_inline_comments(review_comments)
    fmt_discussion      = _fmt_discussion(issue_comments)

    elapsed = round(time.perf_counter() - start, 2)

    logger.info("=" * 60)
    logger.info("✅ STEP 1 DONE | {}s", elapsed)
    logger.info("   fmt_title           : {} chars", len(fmt_title))
    logger.info("   fmt_description     : {} chars", len(fmt_description))
    logger.info("   fmt_commits         : {} chars ({} commits)",
                len(fmt_commits), len(commits))
    logger.info("   fmt_files           : {} chars ({} files)",
                len(fmt_files), len(changed_files))
    logger.info("   fmt_inline_comments : {} chars ({} comments)",
                len(fmt_inline_comments), len(review_comments))
    logger.info("   fmt_discussion      : {} chars ({} comments)",
                len(fmt_discussion), len(issue_comments))
    logger.info("   fmt_diff            : {} chars ({} lines)",
                len(raw_diff), len(raw_diff.splitlines()))
    logger.info("=" * 60)
    logger.info("🚀 STEP 2 | Dispatching 5 parallel agents...")
    logger.info("   → 🔴 Vulnerability Scanner")
    logger.info("   → 🟡 Performance Reviewer")
    logger.info("   → 🟢 Coverage Auditor")
    logger.info("   → 🟣 Code Suggestion")
    logger.info("   → 🟠 Intent Verifier")
    logger.info("=" * 60)

    return {
        "fmt_title":           fmt_title,
        "fmt_description":     fmt_description,
        "fmt_commits":         fmt_commits,
        "fmt_files":           fmt_files,
        "fmt_inline_comments": fmt_inline_comments,
        "fmt_discussion":      fmt_discussion,
        "fmt_diff":            raw_diff,
        "raw_diff":            raw_diff,
        "head_sha":            meta["head_sha"],
        "pr_title":            meta["title"],
        "pr_body":             meta["body"],
        "pr_author":           meta["author"],
        "pr_branch":           meta["branch"],
        "pr_base_branch":      meta["base_branch"],
        "pr_state":            meta["state"],
        "pr_milestone":        meta["milestone"],
        "pr_reviewers":        meta["reviewers"],
        "pr_labels":           meta["labels"],
        "linked_issues":       meta["linked_issues"],
        "commits":             commits,
        "review_comments":     review_comments,
        "issue_comments":      issue_comments,
        "changed_files":       changed_files,
        "pr_stats": {
            "files":     meta["changed_files"],
            "additions": meta["additions"],
            "deletions": meta["deletions"],
            "commits":   meta["commits"],
        },
    }


# ===========================================================================
# POST REVIEW NODE
# ===========================================================================

def post_github_review(state: PRReviewerState) -> dict:
    """
    Final node: parse agent reports → extract inline comments
    → post full review to GitHub PR.
    """
    logger.info("=" * 60)
    logger.info("📮 STEP 3 START | Post Review to GitHub")
    logger.info("=" * 60)

    pr_url = state["pr_url"]
    match  = re.match(
        r"https://github\.com/([^/]+)/([^/]+)/pull/(\d+)", pr_url
    )
    if not match:
        logger.error("   ❌ Cannot parse PR URL: {}", pr_url)
        return {
            "github_review_posted": False,
            "github_review_id":     None,
            "github_review_url":    "",
        }

    owner     = match.group(1)
    repo      = match.group(2)
    pr_number = match.group(3)
    head_sha  = state.get("head_sha", "")

    if not head_sha:
        logger.error("   ❌ No head SHA available — cannot post review")
        return {
            "github_review_posted": False,
            "github_review_id":     None,
            "github_review_url":    "",
        }

    if not _GITHUB_TOKEN:
        logger.error("   ❌ No GITHUB_TOKEN set — cannot post review")
        return {
            "github_review_posted": False,
            "github_review_id":     None,
            "github_review_url":    "",
        }

    # ── Extract verdict from final comment ───────────────────────────────
    final_comment = state.get("final_comment") or ""
    if "Approve ✅" in final_comment:
        verdict = "APPROVE"
    elif "Request Changes ❌" in final_comment:
        verdict = "REQUEST CHANGES"
    else:
        verdict = "COMMENT"

    # ── Parse inline comments from agent reports ──────────────────────────
    changed_files = state.get("changed_files") or []
    inline_comments = _parse_inline_comments(
        changed_files        = changed_files,
        security_report      = state.get("security_report")   or "",
        perf_report          = state.get("perf_report")        or "",
        coverage_report      = state.get("coverage_report")    or "",
        suggestion_report    = state.get("suggestion_report")  or "",
    )

    # ── Post review to GitHub ─────────────────────────────────────────────
    result = _post_github_review(
        owner           = owner,
        repo            = repo,
        pr_number       = pr_number,
        head_sha        = head_sha,
        final_comment   = final_comment,
        inline_comments = inline_comments,
        verdict         = verdict,
    )

    return result


# ===========================================================================
# BUILD LANGGRAPH PIPELINE
# ===========================================================================
logger.info("=" * 60)
logger.info("🔧 Building LangGraph pipeline...")

_builder = StateGraph(PRReviewerState)

_builder.add_node("fetch_diff",          fetch_diff)
_builder.add_node("security",            agent_vulnerability_scanner)
_builder.add_node("perf",               agent_performance_reviewer)
_builder.add_node("coverage",           agent_coverage_auditor)
_builder.add_node("suggestion",         agent_code_suggestion)
_builder.add_node("intent",             agent_intent_verifier)
_builder.add_node("lead_reviewer",      agent_lead_reviewer)
_builder.add_node("post_github_review", post_github_review)   # ← NEW

_builder.set_entry_point("fetch_diff")

# fetch_diff → 5 parallel agents
_builder.add_edge("fetch_diff", "security")
_builder.add_edge("fetch_diff", "perf")
_builder.add_edge("fetch_diff", "coverage")
_builder.add_edge("fetch_diff", "suggestion")
_builder.add_edge("fetch_diff", "intent")

# 5 agents → lead reviewer
_builder.add_edge("security",   "lead_reviewer")
_builder.add_edge("perf",       "lead_reviewer")
_builder.add_edge("coverage",   "lead_reviewer")
_builder.add_edge("suggestion", "lead_reviewer")
_builder.add_edge("intent",     "lead_reviewer")

# lead reviewer → post to GitHub → END
_builder.add_edge("lead_reviewer",      "post_github_review")
_builder.add_edge("post_github_review", END)

graph = _builder.compile()

logger.info("✅ LangGraph pipeline built")
logger.info(
    "   fetch_diff → [security, perf, coverage, suggestion, intent]"
    " → lead_reviewer → post_github_review → END"
)
logger.info("=" * 60)