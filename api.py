# api.py
"""
Project 02 — FastAPI PR Reviewer
No Redis — pure in-memory storage
"""
import hashlib
import os
import uuid
import threading
import time
from datetime import datetime

from dotenv import load_dotenv
load_dotenv()

from logger_config import logger

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from graph import graph


app = FastAPI(
    title="Automated GitHub PR Code Reviewer",
    version="5.0.0",
    description=(
        "Multi-agent PR reviewer using Azure OpenAI.\n\n"
        "Posts inline comments and summary review directly to GitHub PR.\n\n"
        "Agents:\n"
        "  1. Vulnerability Scanner\n"
        "  2. Performance Reviewer\n"
        "  3. Test Coverage Auditor\n"
        "  4. Code Suggestion Agent\n"
        "  5. Intent Verification Agent\n"
        "  6. Lead Reviewer\n\n"
        "Pipeline:\n"
        "  fetch_diff → [5 agents] → lead_reviewer → post_github_review\n"
    ),
)

_jobs: dict    = {}
_cache: dict   = {}
_history: list = []


def _cache_key(pr_url: str) -> str:
    return "pr_review:" + hashlib.sha256(pr_url.encode()).hexdigest()


def _save_job(job_id: str, data: dict) -> None:
    _jobs[job_id] = data


def _load_job(job_id: str) -> dict | None:
    return _jobs.get(job_id)


def _save_cache(pr_url: str, data: dict) -> None:
    _cache[_cache_key(pr_url)] = data


def _load_cache(pr_url: str) -> dict | None:
    return _cache.get(_cache_key(pr_url))


def _save_history(job_data: dict) -> None:
    _history.insert(0, {
        "job_id":               job_data["job_id"],
        "pr_url":               job_data["pr_url"],
        "pr_title":             job_data.get("pr_title",  "unknown"),
        "pr_author":            job_data.get("pr_author", "unknown"),
        "status":               job_data["status"],
        "verdict":              job_data.get("verdict",   "unknown"),
        "intent":               job_data.get("intent_verdict", "unknown"),
        "pr_stats":             job_data.get("pr_stats",  {}),
        "github_review_posted": job_data.get("github_review_posted", False),
        "github_review_url":    job_data.get("github_review_url", ""),
        "created_at":           job_data.get("created_at"),
        "finished_at":          job_data.get("finished_at"),
    })
    if len(_history) > 50:
        _history.pop()


def _extract_verdict(final_comment: str) -> str:
    if "Approve ✅" in final_comment:
        return "APPROVE"
    if "Request Changes ❌" in final_comment:
        return "REQUEST CHANGES"
    return "COMMENT"


def _extract_intent_verdict(intent_report: str) -> str:
    if "FAIL" in intent_report:
        return "FAIL"
    if "PARTIAL" in intent_report:
        return "PARTIAL"
    return "PASS"


def _run_pipeline(job_id: str, pr_url: str) -> None:
    pipeline_start = time.perf_counter()

    logger.info("=" * 60)
    logger.info("🚀 PIPELINE START")
    logger.info("   Job ID : {}", job_id)
    logger.info("   PR URL : {}", pr_url)
    logger.info("   Time   : {}", datetime.utcnow().isoformat())
    logger.info("=" * 60)

    try:
        job               = _load_job(job_id)
        job["status"]     = "running"
        job["started_at"] = datetime.utcnow().isoformat()
        _save_job(job_id, job)
        logger.info("📌 Job {}... → RUNNING", job_id[:8])

        logger.info("▶️  Invoking LangGraph pipeline...")
        result  = graph.invoke({"pr_url": pr_url})
        elapsed = round(time.perf_counter() - pipeline_start, 2)

        final_comment  = result.get("final_comment")  or ""
        intent_report  = result.get("intent_report")  or ""
        verdict        = _extract_verdict(final_comment)
        intent_verdict = _extract_intent_verdict(intent_report)

        job["status"]               = "completed"
        job["finished_at"]          = datetime.utcnow().isoformat()
        job["verdict"]              = verdict
        job["intent_verdict"]       = intent_verdict
        job["pr_title"]             = result.get("pr_title")
        job["pr_author"]            = result.get("pr_author")
        job["pr_branch"]            = result.get("pr_branch")
        job["pr_base_branch"]       = result.get("pr_base_branch")
        job["pr_state"]             = result.get("pr_state")
        job["pr_milestone"]         = result.get("pr_milestone")
        job["pr_reviewers"]         = result.get("pr_reviewers")
        job["pr_labels"]            = result.get("pr_labels")
        job["linked_issues"]        = result.get("linked_issues")
        job["pr_stats"]             = result.get("pr_stats")
        job["changed_files"]        = result.get("changed_files")
        job["commits"]              = result.get("commits")
        job["review_comments"]      = result.get("review_comments")
        job["issue_comments"]       = result.get("issue_comments")
        job["head_sha"]             = result.get("head_sha")
        job["intent_report"]        = intent_report
        job["security_report"]      = result.get("security_report")
        job["perf_report"]          = result.get("perf_report")
        job["coverage_report"]      = result.get("coverage_report")
        job["suggestion_report"]    = result.get("suggestion_report")
        job["final_comment"]        = final_comment
        job["github_review_id"]     = result.get("github_review_id")
        job["github_review_url"]    = result.get("github_review_url")
        job["github_review_posted"] = result.get("github_review_posted", False)

        _save_job(job_id, job)
        _save_cache(pr_url, job)
        _save_history(job)

        security_report   = result.get("security_report")   or ""
        perf_report       = result.get("perf_report")       or ""
        coverage_report   = result.get("coverage_report")   or ""
        suggestion_report = result.get("suggestion_report") or ""

        logger.info("=" * 60)
        logger.info("🎉 PIPELINE COMPLETE")
        logger.info("   Job ID              : {}", job_id)
        logger.info("   PR Title            : {}", job["pr_title"])
        logger.info("   PR Author           : {}", job["pr_author"])
        logger.info("   Verdict             : {}", verdict)
        logger.info("   Intent Verdict      : {}", intent_verdict)
        logger.info("   GitHub Review Posted: {}", job["github_review_posted"])
        logger.info("   GitHub Review URL   : {}", job["github_review_url"])
        logger.info("   Total Time          : {}s", elapsed)
        logger.info("=" * 60)

        # ── Print all agent reports ───────────────────────────────────────
        logger.info("")
        logger.info("─" * 60)
        logger.info("🟠 INTENT VERIFICATION REPORT")
        logger.info("─" * 60)
        for line in intent_report.splitlines():
            logger.info("   {}", line)

        logger.info("")
        logger.info("─" * 60)
        logger.info("🔴 SECURITY / VULNERABILITY REPORT")
        logger.info("─" * 60)
        for line in security_report.splitlines():
            logger.info("   {}", line)

        logger.info("")
        logger.info("─" * 60)
        logger.info("🟡 PERFORMANCE REPORT")
        logger.info("─" * 60)
        for line in perf_report.splitlines():
            logger.info("   {}", line)

        logger.info("")
        logger.info("─" * 60)
        logger.info("🟢 TEST COVERAGE REPORT")
        logger.info("─" * 60)
        for line in coverage_report.splitlines():
            logger.info("   {}", line)

        logger.info("")
        logger.info("─" * 60)
        logger.info("🟣 CODE SUGGESTION REPORT")
        logger.info("─" * 60)
        for line in suggestion_report.splitlines():
            logger.info("   {}", line)

        logger.info("")
        logger.info("─" * 60)
        logger.info("🔵 FINAL LEAD REVIEWER COMMENT")
        logger.info("─" * 60)
        for line in final_comment.splitlines():
            logger.info("   {}", line)

        logger.info("")
        logger.info("=" * 60)
        logger.info("✅ ALL DONE | Job: {}...", job_id[:8])
        logger.info("   Intent     : {} chars", len(intent_report))
        logger.info("   Security   : {} chars", len(security_report))
        logger.info("   Perf       : {} chars", len(perf_report))
        logger.info("   Coverage   : {} chars", len(coverage_report))
        logger.info("   Suggestion : {} chars", len(suggestion_report))
        logger.info("   Final      : {} chars", len(final_comment))
        logger.info("   GH Review  : {}", job["github_review_url"] or "not posted")
        logger.info("=" * 60)

    except Exception as exc:
        elapsed = round(time.perf_counter() - pipeline_start, 2)
        logger.error("=" * 60)
        logger.error("💥 PIPELINE FAILED")
        logger.error("   Job ID : {}", job_id)
        logger.error("   Error  : {}", exc)
        logger.error("   Time   : {}s", elapsed)
        logger.error("=" * 60)

        job                = _load_job(job_id) or {}
        job["status"]      = "failed"
        job["error"]       = str(exc)
        job["finished_at"] = datetime.utcnow().isoformat()
        _save_job(job_id, job)


class ReviewRequest(BaseModel):
    pr_url: str


@app.get("/health", tags=["System"])
def health():
    logger.info("💚 GET /health")
    return {
        "status":  "ok",
        "version": "5.0.0",
        "features": [
            "inline GitHub comments",
            "summary GitHub review",
            "5 specialist agents",
            "intent verification",
        ],
        "stats": {
            "jobs_total":     len(_jobs),
            "cache_total":    len(_cache),
            "history_total":  len(_history),
            "jobs_running":   sum(1 for j in _jobs.values() if j.get("status") == "running"),
            "jobs_queued":    sum(1 for j in _jobs.values() if j.get("status") == "queued"),
            "jobs_completed": sum(1 for j in _jobs.values() if j.get("status") == "completed"),
            "jobs_failed":    sum(1 for j in _jobs.values() if j.get("status") == "failed"),
        },
    }


@app.post("/review", tags=["Review"])
def submit_review(request: ReviewRequest):
    logger.info("=" * 60)
    logger.info("📨 POST /review")
    logger.info("   PR URL : {}", request.pr_url)
    logger.info("=" * 60)

    if "github.com" not in request.pr_url or "/pull/" not in request.pr_url:
        logger.error("❌ Invalid PR URL: {}", request.pr_url)
        raise HTTPException(status_code=400, detail="Invalid GitHub PR URL.")

    cached = _load_cache(request.pr_url)
    if cached:
        logger.info("⚡ Cache HIT")
        logger.info("   Verdict             : {}", cached.get("verdict"))
        logger.info("   GitHub Review Posted: {}", cached.get("github_review_posted"))
        logger.info("   GitHub Review URL   : {}", cached.get("github_review_url"))
        return JSONResponse({
            "job_id":               cached.get("job_id", "cached"),
            "status":               "completed",
            "cached":               True,
            "pr_url":               request.pr_url,
            "verdict":              cached.get("verdict"),
            "intent_verdict":       cached.get("intent_verdict"),
            "github_review_posted": cached.get("github_review_posted"),
            "github_review_url":    cached.get("github_review_url"),
            "result":               cached,
        })

    logger.info("   Cache MISS — creating new job")
    job_id = str(uuid.uuid4())
    job = {
        "job_id":      job_id,
        "pr_url":      request.pr_url,
        "status":      "queued",
        "created_at":  datetime.utcnow().isoformat(),
        "started_at":  None,
        "finished_at": None,
        "error":       None,
    }
    _save_job(job_id, job)
    logger.info("   ✅ Job created  | {}", job_id)

    thread = threading.Thread(
        target=_run_pipeline,
        args=(job_id, request.pr_url),
        daemon=True,
    )
    thread.start()
    logger.info("   ✅ Pipeline thread started | Returning 202")
    logger.info("=" * 60)

    return JSONResponse(
        status_code=202,
        content={
            "job_id":  job_id,
            "status":  "queued",
            "pr_url":  request.pr_url,
            "message": (
                f"Review pipeline started. "
                f"Results will be posted directly to the GitHub PR. "
                f"Poll GET /review/{job_id}/status for updates."
            ),
        },
    )


@app.get("/review/{job_id}/status", tags=["Review"])
def get_status(job_id: str):
    logger.info("🔍 GET /review/{}/status", job_id)
    job = _load_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found.")

    logger.info("   Status              : {}", job["status"])
    if job.get("pr_title"):
        logger.info("   PR Title            : {}", job["pr_title"])
    if job.get("verdict"):
        logger.info("   Verdict             : {}", job["verdict"])
    if job.get("github_review_url"):
        logger.info("   GitHub Review URL   : {}", job["github_review_url"])

    return JSONResponse({
        "job_id":               job["job_id"],
        "status":               job["status"],
        "pr_url":               job["pr_url"],
        "pr_title":             job.get("pr_title"),
        "pr_author":            job.get("pr_author"),
        "verdict":              job.get("verdict"),
        "intent_verdict":       job.get("intent_verdict"),
        "github_review_posted": job.get("github_review_posted"),
        "github_review_url":    job.get("github_review_url"),
        "created_at":           job.get("created_at"),
        "started_at":           job.get("started_at"),
        "finished_at":          job.get("finished_at"),
        "error":                job.get("error"),
    })


@app.get("/review/{job_id}/report", tags=["Review"])
def get_report(job_id: str):
    logger.info("📄 GET /review/{}/report", job_id)
    job = _load_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found.")
    if job["status"] != "completed":
        raise HTTPException(
            status_code=400,
            detail=f"Not completed. Status: {job['status']}",
        )

    logger.info("   Verdict             : {}", job.get("verdict"))
    logger.info("   GitHub Review Posted: {}", job.get("github_review_posted"))
    logger.info("   GitHub Review URL   : {}", job.get("github_review_url"))

    return JSONResponse({
        "job_id":               job_id,
        "created_at":           job.get("created_at"),
        "finished_at":          job.get("finished_at"),
        "verdict":              job.get("verdict"),
        "intent_verdict":       job.get("intent_verdict"),
        "github_review_posted": job.get("github_review_posted"),
        "github_review_url":    job.get("github_review_url"),
        "github_review_id":     job.get("github_review_id"),
        "pr_url":               job["pr_url"],
        "pr_title":             job.get("pr_title"),
        "pr_author":            job.get("pr_author"),
        "pr_branch":            job.get("pr_branch"),
        "pr_base_branch":       job.get("pr_base_branch"),
        "pr_state":             job.get("pr_state"),
        "pr_labels":            job.get("pr_labels"),
        "pr_milestone":         job.get("pr_milestone"),
        "pr_reviewers":         job.get("pr_reviewers"),
        "linked_issues":        job.get("linked_issues"),
        "pr_stats":             job.get("pr_stats"),
        "changed_files":        job.get("changed_files"),
        "commits":              job.get("commits"),
        "review_comments":      job.get("review_comments"),
        "issue_comments":       job.get("issue_comments"),
        "intent_report":        job.get("intent_report"),
        "security_report":      job.get("security_report"),
        "perf_report":          job.get("perf_report"),
        "coverage_report":      job.get("coverage_report"),
        "suggestion_report":    job.get("suggestion_report"),
        "final_comment":        job.get("final_comment"),
    })


@app.get("/review/{job_id}/github", tags=["Review"])
def get_github_review(job_id: str):
    """GitHub review post status and URL."""
    logger.info("🐙 GET /review/{}/github", job_id)
    job = _load_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found.")
    if job["status"] != "completed":
        raise HTTPException(
            status_code=400,
            detail=f"Not completed. Status: {job['status']}",
        )
    logger.info("   GitHub Review Posted: {}", job.get("github_review_posted"))
    logger.info("   GitHub Review URL   : {}", job.get("github_review_url"))
    return JSONResponse({
        "job_id":               job_id,
        "pr_url":               job["pr_url"],
        "pr_title":             job.get("pr_title"),
        "verdict":              job.get("verdict"),
        "github_review_posted": job.get("github_review_posted"),
        "github_review_url":    job.get("github_review_url"),
        "github_review_id":     job.get("github_review_id"),
    })


@app.get("/review/{job_id}/intent", tags=["Review"])
def get_intent(job_id: str):
    logger.info("🎯 GET /review/{}/intent", job_id)
    job = _load_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found.")
    if job["status"] != "completed":
        raise HTTPException(status_code=400, detail=f"Not completed. Status: {job['status']}")
    return JSONResponse({
        "job_id":         job_id,
        "pr_url":         job["pr_url"],
        "pr_title":       job.get("pr_title"),
        "pr_author":      job.get("pr_author"),
        "intent_verdict": job.get("intent_verdict"),
        "intent_report":  job.get("intent_report"),
    })


@app.get("/review/{job_id}/suggestions", tags=["Review"])
def get_suggestions(job_id: str):
    logger.info("💡 GET /review/{}/suggestions", job_id)
    job = _load_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found.")
    if job["status"] != "completed":
        raise HTTPException(status_code=400, detail=f"Not completed. Status: {job['status']}")
    report = job.get("suggestion_report", "")
    count  = report.count("Suggestion")
    return JSONResponse({
        "job_id":            job_id,
        "pr_url":            job["pr_url"],
        "pr_title":          job.get("pr_title"),
        "suggestion_count":  count,
        "suggestion_report": report,
    })


@app.get("/review/{job_id}/security", tags=["Review"])
def get_security(job_id: str):
    logger.info("🔴 GET /review/{}/security", job_id)
    job = _load_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found.")
    if job["status"] != "completed":
        raise HTTPException(status_code=400, detail=f"Not completed. Status: {job['status']}")
    return JSONResponse({
        "job_id":          job_id,
        "pr_url":          job["pr_url"],
        "pr_title":        job.get("pr_title"),
        "security_report": job.get("security_report"),
    })


@app.get("/review/{job_id}/performance", tags=["Review"])
def get_performance(job_id: str):
    logger.info("🟡 GET /review/{}/performance", job_id)
    job = _load_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found.")
    if job["status"] != "completed":
        raise HTTPException(status_code=400, detail=f"Not completed. Status: {job['status']}")
    return JSONResponse({
        "job_id":      job_id,
        "pr_url":      job["pr_url"],
        "pr_title":    job.get("pr_title"),
        "perf_report": job.get("perf_report"),
    })


@app.get("/review/{job_id}/coverage", tags=["Review"])
def get_coverage(job_id: str):
    logger.info("🟢 GET /review/{}/coverage", job_id)
    job = _load_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found.")
    if job["status"] != "completed":
        raise HTTPException(status_code=400, detail=f"Not completed. Status: {job['status']}")
    return JSONResponse({
        "job_id":          job_id,
        "pr_url":          job["pr_url"],
        "pr_title":        job.get("pr_title"),
        "coverage_report": job.get("coverage_report"),
    })


@app.get("/review/{job_id}", tags=["Review"])
def get_review(job_id: str):
    logger.info("📋 GET /review/{}", job_id)
    job = _load_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found.")
    logger.info("   Status : {}", job["status"])
    return JSONResponse(job)


@app.get("/history", tags=["System"])
def get_history(limit: int = Query(default=10, ge=1, le=50)):
    logger.info("📜 GET /history | limit: {}", limit)
    records = _history[:limit]
    return JSONResponse({"total": len(records), "limit": limit, "reviews": records})


@app.get("/jobs", tags=["System"])
def list_jobs(status: str | None = Query(default=None)):
    logger.info("📋 GET /jobs | filter: {}", status or "all")
    all_jobs = list(_jobs.values())
    if status:
        all_jobs = [j for j in all_jobs if j.get("status") == status]
    summary = [{
        "job_id":               j["job_id"],
        "pr_url":               j["pr_url"],
        "pr_title":             j.get("pr_title"),
        "pr_author":            j.get("pr_author"),
        "status":               j["status"],
        "verdict":              j.get("verdict"),
        "intent_verdict":       j.get("intent_verdict"),
        "github_review_posted": j.get("github_review_posted"),
        "github_review_url":    j.get("github_review_url"),
        "created_at":           j.get("created_at"),
        "finished_at":          j.get("finished_at"),
        "error":                j.get("error"),
    } for j in all_jobs]
    return JSONResponse({"total": len(summary), "filter": status or "all", "jobs": summary})


@app.delete("/review/{job_id}", tags=["System"])
def delete_review(job_id: str):
    logger.info("🗑️  DELETE /review/{}", job_id)
    job = _load_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found.")
    _jobs.pop(job_id, None)
    return JSONResponse({"message": f"Job {job_id} deleted.", "job_id": job_id})


@app.delete("/cache", tags=["System"])
def clear_cache():
    logger.info("🧹 DELETE /cache")
    count = len(_cache)
    _cache.clear()
    return JSONResponse({"message": "Cache cleared.", "entries_removed": count})


@app.delete("/jobs", tags=["System"])
def clear_all_jobs():
    logger.info("🧹 DELETE /jobs")
    count = len(_jobs)
    _jobs.clear()
    return JSONResponse({"message": "All jobs cleared.", "jobs_removed": count})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8001,
        reload=False,
        log_level="warning",
    )