"""
SmartEx backend

How to run locally:
1) Install dependencies:
   pip install fastapi uvicorn openai pydantic python-multipart

2) Set your OpenAI API key (Windows CMD example):
   set OPENAI_API_KEY=your_key_here

3) Start the server from this folder:
   uvicorn main:app --reload --port 8000
"""

import os
import json
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from openai import OpenAI


# -----------------------------
# Configuration & constants
# -----------------------------

SMARTEX_SYSTEM_PROMPT = """
You are SmartEx, a virtual assistant specialized in the design and management of emergency exercises and serious games
for private companies and public organizations in civil protection, business continuity, crisis and emergency management.

You do NOT behave as a generic chat. You act as an AI teammate and always work in a structured, guided, step-by-step way.

You follow a 4-phase workflow:

Phase 1 – Analysis & Objectives:
- Understand context, risks, stakeholders, constraints and learning objectives.
- Produce scope, objectives and capabilities to be tested.

Phase 2 – Exercise / Game Design:
- Define exercise type, format, roles, timing, mechanics and governance.
- Produce a Concept of Exercise (CONEX).

Phase 3 – Scenario, MEL & Materials:
- Build scenario narrative, Master Events List (MEL) and all exercise documents (players’, controllers’ and evaluators’ handbooks, checklists).

Phase 4 – Conduct, Evaluation & Improvement:
- Plan delivery, evaluation framework, After Action Review and improvement plan (PDCA).

You guide the user using:
- numbered, targeted questions in small blocks,
- structured summaries of what you understood,
- explicit references to the current phase,
- critical thinking (you challenge weak assumptions and highlight risks).

You are designed to support non-experts, translating their answers into professional exercise designs and documents with clear structure and actionable content.
"""

DEFAULT_MODEL = "gpt-4o-mini"

BASE_DIR = Path(__file__).resolve().parent
GLOBAL_KB_PATH = BASE_DIR / "global_kb.json"
UPLOAD_ROOT = BASE_DIR / "uploaded_files"
UPLOAD_ROOT.mkdir(exist_ok=True)


# -----------------------------
# Pydantic models
# -----------------------------

class ProjectInfo(BaseModel):
    name: Optional[str] = None
    organisation: Optional[str] = None
    exercise_type: Optional[str] = None
    exercise_date: Optional[str] = None


class ChatRequest(BaseModel):
    mode: str = Field(..., description="SmartEx module selector")
    user_message: Optional[str] = Field(
        default=None, description="Optional user context or question"
    )
    project: Optional[ProjectInfo] = Field(
        default=None,
        description="Optional exercise project info (used for context and uploaded documents)",
    )


class ChatResponse(BaseModel):
    reply: str


# -----------------------------
# Global KB loading
# -----------------------------

GLOBAL_KB: List[Dict[str, Any]] = []


def load_global_kb() -> None:
    global GLOBAL_KB
    if not GLOBAL_KB_PATH.exists():
        GLOBAL_KB = []
        return
    try:
        with GLOBAL_KB_PATH.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            GLOBAL_KB = data
        else:
            GLOBAL_KB = []
    except Exception:
        GLOBAL_KB = []


def build_global_kb_context() -> Optional[str]:
    """
    Build a system-level context message from the global KB documents.
    """
    if not GLOBAL_KB:
        return None

    # Currently the number of KB documents is manageable, so include all.
    docs = GLOBAL_KB

    parts: List[str] = [
        "SmartEx has access to internal reference knowledge from the following open or proprietary documents provided by their original authors.",
        "Use them as high-level guidance when designing, conducting and evaluating exercises.",
        "You MUST NOT claim formal legal compliance with any standard or law. Instead, say that your outputs are aligned with the principles of these references, and clearly state assumptions and limitations when the user asks for compliance.",
        ""
    ]

    for doc in docs:
        title = doc.get("title", "Untitled reference")
        dtype = doc.get("type", "reference document")
        author = doc.get("author", "Unknown author or organisation")
        summary = (doc.get("summary") or "").strip()
        exercise_use = (doc.get("exercise_use") or "").strip()

        parts.append(f"- {title} ({dtype}) – author: {author}")
        if summary:
            parts.append(summary)
        if exercise_use:
            parts.append("Use in exercises: " + exercise_use)
        parts.append("")

    return "\n".join(parts)


# -----------------------------
# Uploaded documents management
# -----------------------------

def get_project_id(project_name: Optional[str]) -> str:
    """
    Convert a free-text project name into a safe folder identifier.
    """
    if not project_name:
        return "default"

    slug = "".join(c.lower() if c.isalnum() else "_" for c in project_name)
    slug = "_".join(part for part in slug.split("_") if part)
    return slug or "default"


def project_metadata_path(project_id: str) -> Path:
    project_dir = UPLOAD_ROOT / project_id
    project_dir.mkdir(parents=True, exist_ok=True)
    return project_dir / "metadata.json"


def load_project_uploaded_docs(project_id: str) -> List[Dict[str, Any]]:
    meta_path = project_metadata_path(project_id)
    if not meta_path.exists():
        return []
    try:
        with meta_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        docs = data.get("docs", [])
        if isinstance(docs, list):
            return docs
        return []
    except Exception:
        return []


def save_project_uploaded_docs(project_id: str, docs: List[Dict[str, Any]]) -> None:
    meta_path = project_metadata_path(project_id)
    meta = {"docs": docs}
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


def build_uploaded_docs_context(project_id: str) -> Optional[str]:
    """
    Build a system message that describes which documents have been uploaded
    for this project and (optionally) includes their summaries.
    """
    docs = load_project_uploaded_docs(project_id)
    if not docs:
        return None

    parts: List[str] = [
        "For this exercise project, the following user-uploaded documents are available as internal references.",
        "You only have access to the short summaries below; you do not see the full files.",
        "Use them to align exercise products and reports when the user explicitly asks for alignment with these documents.",
        ""
    ]

    for doc in docs:
        filename = doc.get("filename", "uploaded_document")
        summary = (doc.get("summary") or "").strip()
        uploaded_at = doc.get("uploaded_at", "")
        line = f"- {filename}"
        if uploaded_at:
            line += f" (uploaded at {uploaded_at})"
        parts.append(line)
        if summary:
            parts.append("Summary: " + summary)
        parts.append("")

    return "\n".join(parts)


def summarise_uploaded_document(client: OpenAI, filename: str, raw_text: str) -> str:
    """
    Create a structured summary of an uploaded document using the LLM.
    Uses only a truncated excerpt of the raw text to control token usage.
    """
    text = (raw_text or "").strip()
    if not text:
        return "No readable text could be extracted from this file. Treat it as an internal document whose detailed content is unknown."

    snippet = text[:15000]

    system_msg = (
        "You are an assistant that creates concise, structured summaries of documents "
        "for an emergency exercise design assistant called SmartEx. "
        "Highlight the domain (e.g. civil protection, business continuity, nuclear, cyber), "
        "main objectives, key requirements and anything relevant for designing or evaluating exercises."
    )
    user_msg = (
        f"Summarise the following document content for later use in emergency exercises. "
        f"Document name: {filename}\n\n"
        f"CONTENT EXCERPT (may be partial and noisy):\n{snippet}"
    )

    try:
        resp = client.chat.completions.create(
            model=DEFAULT_MODEL,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.2,
        )
        summary = resp.choices[0].message.content or ""
        return summary.strip()
    except Exception:
        return "A summary could not be generated due to an internal error. Treat this as an internal reference document without detailed content."


# -----------------------------
# Mode instructions
# -----------------------------

def build_mode_instruction(mode: str) -> str:
    """Return a concise instruction based on the selected SmartEx mode."""
    normalized = (mode or "").strip().lower()
    if normalized == "new_exercise":
        return (
            "The user wants to start a completely new emergency exercise from scratch. "
            "Begin at Phase 1 (Analysis & Objectives) and then guide them through all "
            "4 phases step-by-step."
        )
    if normalized == "objectives":
        return (
            "The user wants help mainly with Phase 1: Analysis & Objectives. Focus on "
            "clarifying context, risks, objectives and capabilities. Ask targeted "
            "questions in blocks and summarise."
        )
    if normalized == "design":
        return (
            "The user wants help mainly with Phase 2: Exercise / Game Design. Assume "
            "basic context exists or quickly ask for the minimum info needed, then "
            "design type, format, roles, time structure and mechanics."
        )
    if normalized == "scenario_mel":
        return (
            "The user wants help mainly with Phase 3: Scenario, MEL & Materials. Ask for "
            "any essential missing inputs (context, objectives), then build the narrative "
            "scenario and a Master Events List (MEL) with injects."
        )
    if normalized == "evaluation":
        return (
            "The user wants help mainly with Phase 4: Conduct, Evaluation & Improvement. "
            "Help them design conduct plans, evaluation criteria, After Action Review and "
            "an improvement plan aligned with PDCA."
        )
    return (
        "The user’s requested mode is unclear. Ask a short clarifying question to map "
        "their need to one or more SmartEx phases, then proceed accordingly."
    )


# -----------------------------
# FastAPI app setup
# -----------------------------

app = FastAPI(title="SmartEx Backend")

# For local development and the prototype, we allow all origins.
# This avoids CORS issues when opening index.html from different hosts/ports or file://.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,  # must be False when using "*"
    allow_methods=["*"],
    allow_headers=["*"],
)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Load global KB at startup
load_global_kb()


# -----------------------------
# Endpoints
# -----------------------------

@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    mode_instruction = build_mode_instruction(request.mode)

    # Determine project id from request.project
    project_name = None
    if request.project and request.project.name:
        project_name = request.project.name.strip()
    project_id = get_project_id(project_name)

    # Build additional context: global KB, uploaded docs, project info
    global_kb_ctx = build_global_kb_context()
    uploaded_docs_ctx = build_uploaded_docs_context(project_id)

    project_ctx = None
    if request.project:
        parts = []
        if request.project.name:
            parts.append(f"Project name: {request.project.name}")
        if request.project.organisation:
            parts.append(f"Lead organisation / authority: {request.project.organisation}")
        if request.project.exercise_type:
            parts.append(f"Exercise type: {request.project.exercise_type}")
        if request.project.exercise_date:
            parts.append(f"Planned exercise date: {request.project.exercise_date}")
        if parts:
            project_ctx = (
                "Exercise project context provided by the user:\n" + "\n".join(parts)
            )

    messages: List[Dict[str, str]] = [
        {"role": "system", "content": SMARTEX_SYSTEM_PROMPT},
        {"role": "system", "content": mode_instruction},
    ]

    if global_kb_ctx:
        messages.append({"role": "system", "content": global_kb_ctx})
    if uploaded_docs_ctx:
        messages.append({"role": "system", "content": uploaded_docs_ctx})
    if project_ctx:
        messages.append({"role": "system", "content": project_ctx})

    if request.user_message:
        user_text = request.user_message.strip()
        if user_text:
            messages.append({"role": "user", "content": user_text})

    response = client.chat.completions.create(
        model=DEFAULT_MODEL,
        messages=messages,
        temperature=0.4,
    )

    reply_text = response.choices[0].message.content or ""
    return ChatResponse(reply=reply_text)


@app.post("/upload")
async def upload(
    project_name: str = Form("default"),
    file: UploadFile = File(...),
) -> Dict[str, str]:
    """
    Upload a document linked to a given project.
    The file is stored on disk and a short summary is generated and saved in metadata.json.
    """
    project_id = get_project_id(project_name)
    project_dir = UPLOAD_ROOT / project_id
    project_dir.mkdir(parents=True, exist_ok=True)

    safe_filename = file.filename or "uploaded_document"
    dest_path = project_dir / safe_filename

    # Save file
    content = await file.read()
    with dest_path.open("wb") as f:
        f.write(content)

    # Attempt a very simple text extraction (partial) without extra dependencies
    try:
        raw_text = content[:50000].decode("latin-1", errors="ignore")
    except Exception:
        raw_text = ""

    summary = ""
    if raw_text.strip():
        summary = summarise_uploaded_document(client, safe_filename, raw_text)
    else:
        summary = "No readable text could be extracted from this file. Details are unknown; treat as an internal reference."

    # Update project metadata
    docs = load_project_uploaded_docs(project_id)
    docs.append(
        {
            "filename": safe_filename,
            "stored_path": str(dest_path),
            "summary": summary,
            "uploaded_at": datetime.utcnow().isoformat() + "Z",
        }
    )
    save_project_uploaded_docs(project_id, docs)

    return {"filename": safe_filename}
