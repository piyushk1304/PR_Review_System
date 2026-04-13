# state.py
from typing import TypedDict


class PRReviewerState(TypedDict, total=False):
    # ── Input ────────────────────────────────────────────────────────────────
    pr_url:              str

    # ── Raw fetched data ──────────────────────────────────────────────────────
    pr_title:            str
    pr_body:             str
    pr_author:           str
    pr_branch:           str
    pr_base_branch:      str
    pr_labels:           list[str]
    pr_state:            str
    pr_milestone:        str
    pr_reviewers:        list[str]
    linked_issues:       list[str]
    changed_files:       list[dict]
    commits:             list[dict]
    review_comments:     list[dict]
    issue_comments:      list[dict]
    pr_stats:            dict
    raw_diff:            str
    head_sha:            str        # ← needed for posting review

    # ── Formatted inputs passed to agents ─────────────────────────────────────
    fmt_title:           str
    fmt_description:     str
    fmt_commits:         str
    fmt_files:           str
    fmt_inline_comments: str
    fmt_discussion:      str
    fmt_diff:            str

    # ── Agent reports ─────────────────────────────────────────────────────────
    security_report:     str
    perf_report:         str
    coverage_report:     str
    suggestion_report:   str
    intent_report:       str
    final_comment:       str

    # ── GitHub review post result ─────────────────────────────────────────────
    github_review_id:    int
    github_review_url:   str
    github_review_posted: bool