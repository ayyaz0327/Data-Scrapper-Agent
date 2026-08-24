"""
AI Data Scraper Agent — CrewAI + Streamlit (single file)

This is a 1:1 port of the n8n workflow "AI Data Scraper Agent".
Every node's logic is kept exactly the same — nothing changed, nothing added:

    Chat Trigger -> Set - Config -> AI Agent - Planner (Gemini) -> Code - Parse Plan
    -> Split Out Queries -> HTTP Request SerpApi (paginated) -> Split Out Results
    -> Code - Extract From Results -> Code - Clean & Normalize
    -> Read Existing XLSX -> Code - Deduplicate -> IF - Dry Run?
         (true)  -> Code - Build Preview  -> Respond to Chat
         (false) -> Code - Merge Final Rows -> Write XLSX -> Code - Build Summary -> Respond to Chat

Run:
    pip install streamlit crewai pandas openpyxl requests
    streamlit run app.py
"""

import json
import os
import re
import time
from urllib.parse import urlparse

import pandas as pd
import requests
import streamlit as st
from crewai import Agent, Task, Crew, Process, LLM
from dotenv import load_dotenv

load_dotenv()  # loads variables from a local .env file (never committed to git)

# ===========================================================================
# Set - Config  (same defaults as the n8n Set node)
# ===========================================================================
DEFAULT_CONFIG = {
    "maxResultsPerQuery": 10,
    "maxPages": 2,
    "dryRun": True,
    "rateLimitSeconds": 2,
    "outputFilePath": "/data/scraper_output.xlsx",
}

# ===========================================================================
# AI Agent - Planner + Gemini Model - Planner
# (system prompt copied word for word from the n8n node's systemMessage)
# ===========================================================================
PLANNER_SYSTEM_PROMPT = """You are a research planner. Given a user's niche/topic request, output ONLY valid JSON, no markdown, no explanation:
{
  "search_queries": ["query1", "query2", "query3"],
  "columns": ["company_name", "website", "price_range", "moq", "contact_email", "source_url"],
  "required_fields": ["company_name"]
}
Rules:
- Produce 3-5 highly specific search queries based on the user's fields/niche (use site: operators when relevant, e.g. site:fiverr.com)
- Columns must reflect exactly what the user asked to extract
- Always include "source_url" as the last column
- required_fields must be MINIMAL: list only 1 field that reliably identifies a row (usually "company_name"). Do NOT put "website" or "contact_email" in required_fields, because many pages legitimately lack them.
- For job-posting or hiring niches (e.g. LinkedIn, Indeed, job boards), use these columns instead: "job_title", "company_name", "location", "requirements_summary", "date_added", "source_url". Do NOT include "website" or "contact_email" for job-posting niches — these fields are not available on public job listing pages and will always come back empty.
- Never merge a job title and a company name into the same field — job_title and company_name must always be separate columns when both are relevant.
- Return raw JSON only"""

GEMINI_MODEL = "gemini/gemini-3.1-flash-lite"


def build_planner_agent(gemini_api_key, model_name=GEMINI_MODEL):
    llm = LLM(model=model_name, api_key=gemini_api_key)
    return Agent(
        role="Research Planner",
        goal="Turn a user's niche/topic request into a structured search plan (search_queries, columns, required_fields).",
        backstory=PLANNER_SYSTEM_PROMPT,
        llm=llm,
        verbose=False,
        allow_delegation=False,
    )


def run_planner(planner_agent, user_request):
    task = Task(
        description=f"{PLANNER_SYSTEM_PROMPT}\n\nUser request: {user_request}",
        expected_output="Raw JSON object only, matching exactly the schema described in the instructions. No markdown fences, no explanation.",
        agent=planner_agent,
    )
    crew = Crew(agents=[planner_agent], tasks=[task], process=Process.sequential, verbose=False)
    result = crew.kickoff()
    return str(result)


# ===========================================================================
# Code - Parse Plan
# ===========================================================================
def parse_plan_output(raw_text):
    cleaned = re.sub(r"```json|```", "", str(raw_text)).strip()
    try:
        plan = json.loads(cleaned)
    except Exception as e:
        raise ValueError(f"Planner did not return valid JSON. Raw output: {raw_text}") from e

    if "search_queries" not in plan or "columns" not in plan:
        raise ValueError("Planner JSON missing required keys (search_queries/columns).")

    if not isinstance(plan.get("required_fields"), list) or len(plan["required_fields"]) == 0:
        plan["required_fields"] = ["company_name"]

    return plan


# ===========================================================================
# Split Out - Queries / HTTP Request - SerpApi Search / Split Out - Results
# ===========================================================================
def serpapi_search(queries, max_results_per_query, max_pages, rate_limit_seconds, api_key):
    all_results = []
    for q in queries:
        for page_count in range(max_pages):
            params = {
                "q": q,
                "num": max_results_per_query,
                "engine": "google",
                "start": page_count * max_results_per_query,
                "api_key": api_key,
            }
            try:
                resp = requests.get("https://serpapi.com/search", params=params, timeout=30)
                data = resp.json()
            except Exception:
                break  # onError: continueRegularOutput

            organic = data.get("organic_results") or []
            if len(organic) == 0:
                break

            all_results.extend(organic)
            time.sleep(rate_limit_seconds)
    return all_results


# ===========================================================================
# Code - Extract From Results
# ===========================================================================
JOB_BOARDS = [
    "linkedin", "indeed", "upwork", "glassdoor", "ziprecruiter",
    "monster", "greenhouse", "lever", "wellfound", "angel", "google",
]

TITLE_SPLIT_RE = re.compile(r" - | – | — | \| | at ", re.IGNORECASE)


def _domain_of(url):
    try:
        host = (urlparse(url).hostname or "").lower()
        if host.startswith("www."):
            host = host[4:]
        return host
    except Exception:
        return ""


def _is_job_board(host):
    return any(b in host for b in JOB_BOARDS)


def extract_rows(results, columns):
    out = []
    for r in results:
        link = r.get("link") or r.get("redirect_link") or ""
        title = str(r.get("title") or "").strip()
        snippet = str(r.get("snippet") or "").strip()
        source = str(r.get("source") or "").strip()
        host = _domain_of(link)

        title_parts = [p.strip() for p in TITLE_SPLIT_RE.split(title) if p.strip()]
        job_title = title_parts[0] if title_parts else title

        company = source
        if not company or _is_job_board(company.lower()):
            company = title_parts[-1] if len(title_parts) > 1 else ""

        website = host if host and not _is_job_board(host) else ""

        row = {}
        for col in columns:
            c = col.lower()
            if c == "source_url":
                row[col] = link
            elif re.search(r"(job_title|role_title|position|^title$)", c):
                row[col] = job_title
            elif re.search(r"(job_post|post_link|job_link|apply|listing)", c):
                row[col] = link
            elif re.search(r"(company|business|agency|organization|org_name|^name$)", c) and not re.search(
                r"(user|contact|job)", c
            ):
                row[col] = company
            elif re.search(r"website|url|domain|site", c):
                row[col] = website
            elif re.search(r"(need|require|description|summary|about|role|detail|what)", c):
                row[col] = snippet
            elif re.search(r"(email|linkedin|contact|phone|twitter|social|handle)", c):
                row[col] = ""
            elif re.search(r"location|city|country", c):
                row[col] = ""
            else:
                row[col] = ""

        if not row.get("source_url"):
            row["source_url"] = link

        out.append(row)
    return out


# ===========================================================================
# Code - Clean & Normalize
# ===========================================================================
def clean_normalize(rows, required_fields):
    out = []
    for original in rows:
        row = dict(original)

        for k, v in list(row.items()):
            if isinstance(v, str):
                row[k] = v.strip()

        if row.get("website"):
            w = str(row["website"]).lower()
            w = re.sub(r"^https?://(www\.)?", "", w)
            w = re.sub(r"/$", "", w)
            row["website"] = w

        row["source_url"] = row.get("source_url") or ""

        missing_required = any(
            not row.get(f) or str(row.get(f)).strip() == "" for f in required_fields
        )
        if missing_required:
            continue

        other_keys = [k for k in row.keys() if k != "source_url"]
        has_content = any(
            row.get(k) not in (None, "") and str(row.get(k)).strip() != "" for k in other_keys
        )
        if not row["source_url"] or not has_content:
            continue

        out.append(row)
    return out


# ===========================================================================
# Read Existing XLSX / Spreadsheet File - Read
# ===========================================================================
def read_existing_xlsx(path):
    if not path or not os.path.exists(path):
        return []
    try:
        df = pd.read_excel(path)
        df = df.fillna("")
        return df.to_dict(orient="records")
    except Exception:
        return []


# ===========================================================================
# Code - Deduplicate
# ===========================================================================
def _key_of(r):
    w = str(r.get("website") or "").lower()
    w = re.sub(r"^https?://(www\.)?", "", w)
    w = re.sub(r"/$", "", w)
    email_field = r.get("contact_email") or r.get("contact_email_or_linkedin") or r.get("email") or ""
    e = str(email_field).lower().strip()
    n = str(r.get("company_name") or "").lower().strip()
    return w or e or n


def deduplicate(new_rows, existing_rows):
    existing = [r for r in existing_rows if r and not r.get("error") and len(r) > 0]
    existing_keys = set(filter(None, (_key_of(r) for r in existing)))

    deduped_new = []
    seen_this_run = set()
    skipped_count = 0

    for row in new_rows:
        k = _key_of(row)
        if not k or k in existing_keys or k in seen_this_run:
            skipped_count += 1
            continue
        seen_this_run.add(k)
        deduped_new.append(row)

    return {
        "newRows": deduped_new,
        "existingRows": existing,
        "skippedCount": skipped_count,
        "newCount": len(deduped_new),
    }


# ===========================================================================
# Code - Build Preview (dry run branch)
# ===========================================================================
def build_preview(dedup_result):
    rows = dedup_result.get("newRows", [])
    if len(rows) == 0:
        return (
            f"DRY RUN - no new rows found for this request. Nothing was written. "
            f"Skipped {dedup_result['skippedCount']} duplicates."
        )
    sample = json.dumps(rows[:10], indent=2)
    return (
        "DRY RUN - nothing was written.\n"
        f"Would add {dedup_result['newCount']} new rows and skip {dedup_result['skippedCount']} duplicates.\n\n"
        f"Preview of new rows:\n{sample}"
    )


# ===========================================================================
# Code - Merge Final Rows (write branch)
# ===========================================================================
FIRST = ["company_name", "website"]
LAST = ["source_url"]
CONTACT_PATTERNS = [
    r"email", r"phone", r"mobile", r"whatsapp", r"telegram", r"linkedin",
    r"twitter", r"^x_", r"instagram", r"facebook", r"tiktok", r"youtube",
    r"social", r"handle", r"contact",
]


def merge_final_rows(existing_rows, new_rows):
    all_rows = list(existing_rows) + list(new_rows)
    if len(all_rows) == 0:
        return []

    seen = set()
    all_keys = []
    for r in all_rows:
        if not isinstance(r, dict):
            continue
        for k in r.keys():
            if k == "_placeholder":
                continue
            if k not in seen:
                seen.add(k)
                all_keys.append(k)

    first_cols = [k for k in FIRST if k in seen]
    last_cols = [k for k in LAST if k in seen]
    used_first_last = set(first_cols) | set(last_cols)
    remaining = [k for k in all_keys if k not in used_first_last]
    contact_cols = [k for k in remaining if any(re.search(p, k, re.IGNORECASE) for p in CONTACT_PATTERNS)]
    contact_set = set(contact_cols)
    middle_cols = [k for k in remaining if k not in contact_set]

    ordered_cols = first_cols + middle_cols + contact_cols + last_cols

    out = []
    for r in all_rows:
        src = r if isinstance(r, dict) else {}
        row = {}
        for c in ordered_cols:
            v = src.get(c)
            row[c] = "" if v is None else v
        out.append(row)
    return out


# ===========================================================================
# Spreadsheet File - Write / Write XLSX To Disk
# ===========================================================================
def write_xlsx(rows, path):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_excel(path, index=False)


# ===========================================================================
# Code - Build Summary
# ===========================================================================
def build_summary(dedup_result, output_path):
    return (
        f"Done. Added {dedup_result['newCount']} new rows, "
        f"skipped {dedup_result['skippedCount']} duplicates. Saved to {output_path}."
    )


# ===========================================================================
# STREAMLIT UI  (Chat Trigger + Set-Config sidebar + IF-Dry-Run + Respond to Chat)
# ===========================================================================
st.set_page_config(page_title="AI Data Scraper Agent", page_icon="🔎", layout="centered")

st.title("🔎 AI Data Scraper Agent")
st.caption("CrewAI + Streamlit port of the n8n workflow — same planner, same search, same dedupe/write logic.")

def _get_secret(name):
    """Reads a key from (in order): .env / real env var -> st.secrets. Never hardcoded."""
    val = os.environ.get(name, "")
    if not val:
        try:
            val = st.secrets.get(name, "")
        except Exception:
            val = ""
    return val


with st.sidebar:
    st.header("Config")
    gemini_api_key = st.text_input(
        "Gemini API Key", type="password", value=_get_secret("GEMINI_API_KEY"),
        help="Loaded from .env or Streamlit secrets if set — leave as-is if configured there.",
    )
    serpapi_key = st.text_input(
        "SerpApi Key", type="password", value=_get_secret("SERPAPI_KEY"),
        help="Loaded from .env or Streamlit secrets if set — leave as-is if configured there.",
    )

    st.divider()
    maxResultsPerQuery = st.number_input(
        "maxResultsPerQuery", min_value=1, max_value=100, value=DEFAULT_CONFIG["maxResultsPerQuery"]
    )
    maxPages = st.number_input("maxPages", min_value=1, max_value=10, value=DEFAULT_CONFIG["maxPages"])
    rateLimitSeconds = st.number_input(
        "rateLimitSeconds", min_value=0, max_value=30, value=DEFAULT_CONFIG["rateLimitSeconds"]
    )
    dryRun = st.checkbox("dryRun", value=DEFAULT_CONFIG["dryRun"])
    outputFilePath = st.text_input("outputFilePath", value=DEFAULT_CONFIG["outputFilePath"])

userRequest = st.text_input(
    "What do you want to scrape? (this is the chat trigger input)",
    placeholder="e.g. Fiverr sellers offering n8n automation services",
)

run = st.button("Run", type="primary")

if run:
    if not userRequest.strip():
        st.error("Please enter a request first.")
        st.stop()
    if not gemini_api_key:
        st.error("Gemini API Key is required for the Planner agent.")
        st.stop()
    if not serpapi_key:
        st.error("SerpApi Key is required for the search step.")
        st.stop()

    # ---- AI Agent - Planner ----
    with st.spinner("Planning search queries and output schema..."):
        planner_agent = build_planner_agent(gemini_api_key)
        raw_plan_output = run_planner(planner_agent, userRequest)

    try:
        plan = parse_plan_output(raw_plan_output)
    except ValueError as e:
        st.error(str(e))
        st.stop()

    with st.expander("Planner output (search_queries / columns / required_fields)"):
        st.json(plan)

    # ---- SerpApi Search ----
    with st.spinner("Searching Google via SerpApi..."):
        results = serpapi_search(
            queries=plan["search_queries"],
            max_results_per_query=maxResultsPerQuery,
            max_pages=maxPages,
            rate_limit_seconds=rateLimitSeconds,
            api_key=serpapi_key,
        )

    # ---- Extract + Clean ----
    extracted = extract_rows(results, plan["columns"])
    cleaned = clean_normalize(extracted, plan["required_fields"])

    # ---- Read Existing + Dedupe ----
    existing_rows = read_existing_xlsx(outputFilePath)
    dedup_result = deduplicate(cleaned, existing_rows)

    st.divider()

    # ---- IF - Dry Run? ----
    if dryRun:
        preview_message = build_preview(dedup_result)
        st.subheader("Respond to Chat (Dry Run)")
        st.text(preview_message)
    else:
        final_rows = merge_final_rows(dedup_result["existingRows"], dedup_result["newRows"])
        write_xlsx(final_rows, outputFilePath)
        summary_message = build_summary(dedup_result, outputFilePath)

        st.subheader("Respond to Chat")
        st.text(summary_message)

        if os.path.exists(outputFilePath):
            with open(outputFilePath, "rb") as f:
                st.download_button(
                    "Download updated XLSX",
                    data=f.read(),
                    file_name=os.path.basename(outputFilePath),
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )