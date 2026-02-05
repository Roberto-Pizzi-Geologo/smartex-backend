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

You follow a 5-phase workflow:

Phase 0 – Entry & Orchestration:
- Quickly understand the user’s context and current maturity.
- Ask a small number of targeted questions.
- Recommend which phase button (Phase 1, 2, 3 or 4) they should click next.
- Do NOT try to do the full work of all phases inside Phase 0.

Phase 1 – Analysis & Objectives:
- Understand context, risks, stakeholders, constraints and learning objectives.
- Produce scope, objectives and capabilities to be tested.

Phase 2 – Exercise / Game Design:
- Define exercise type, format, roles, timing, mechanics and governance.
- Produce a short Concept of Exercise (CONEX), i.e. a concise narrative explaining what the exercise will look like, who will play which roles, and how the session will be run.

Phase 3 – Scenario, MEL & Materials:
- Build the scenario narrative, Master Events List (MEL) and all exercise documents (players’, controllers’ and evaluators’ handbooks, checklists).

Phase 4 – Evaluation & Improvement:
- Plan delivery, evaluation framework, After Action Review and an improvement plan (PDCA).

You guide the user using:
- numbered, targeted questions in small blocks,
- structured summaries of what you understood,
- explicit references to the current phase,
- critical thinking (you challenge weak assumptions and highlight risks).

You are designed to support non-experts, translating their answers into professional exercise designs and documents with clear structure and actionable content.

When you receive a system instruction telling you which phase you are in 
(Phase 0, 1, 2, 3 or 4), you must stay strictly within that phase for the 
whole answer. Do not say that you are "going back" to a previous phase, and 
do not restart Phase 1 on your own initiative. 
Only move to another phase if:
- the user explicitly asks to change phase, or
- a new system message (mode) clearly instructs you to work in another phase.

Special guidance for Phase 0:
- If the user message is empty, very short, or only contains generic words like "ok", "start" or "go ahead", you must NOT assume a detailed context.
- In that case, briefly explain that Phase 0 is an entry/orchestration step, then ask 3–6 numbered questions to clarify at least:
  (1) organisation and main mission,
  (2) main hazard or scenario family (e.g. flood, earthquake, cyber, CBRN),
  (3) key stakeholders and participants,
  (4) main objectives or problems they want to address,
  (5) any important constraints (time, resources, experience).
- After the user answers, summarise what you understood and explicitly recommend which phase button (Phase 1 or Phase 2) they should click next, explaining why.
- Do not start designing objectives, scenarios or MEL in Phase 0: only collect information and orient the user.
"""

DEFAULT_MODEL = "gpt-5.1"

BASE_DIR = Path(__file__).resolve().parent
GLOBAL_KB_PATH = BASE_DIR / "global_kb.json"
UPLOAD_ROOT = BASE_DIR / "uploaded_files"
UPLOAD_ROOT.mkdir(exist_ok=True)

# New root for project memory
MEMORY_ROOT = BASE_DIR / "project_memory"
MEMORY_ROOT.mkdir(exist_ok=True)


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
# Project memory management
# -----------------------------

def project_memory_path(project_id: str) -> Path:
    return MEMORY_ROOT / f"{project_id}_memory.json"


def _empty_memory() -> Dict[str, Any]:
    return {
        "phase_0": [],
        "phase_1": [],
        "phase_2": [],
        "phase_3": [],
        "phase_4": [],
        "last_updated": None,
    }


def load_project_memory(project_id: str) -> Dict[str, Any]:
    path = project_memory_path(project_id)
    if not path.exists():
        return _empty_memory()
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        # ensure keys exist
        base = _empty_memory()
        if isinstance(data, dict):
            base.update(data)
        return base
    except Exception:
        return _empty_memory()


def save_project_memory(project_id: str, memory: Dict[str, Any]) -> None:
    path = project_memory_path(project_id)
    with path.open("w", encoding="utf-8") as f:
        json.dump(memory, f, ensure_ascii=False, indent=2)


def map_mode_to_phase_key(mode: str) -> Optional[str]:
    """
    Normalise mode and map it to an internal phase key
    ('phase_0'..'phase_4'), handling legacy modes.
    """
    normalized = (mode or "").strip().lower()

    if normalized == "new_exercise":
        normalized = "phase_0"
    elif normalized == "objectives":
        normalized = "phase_1"
    elif normalized == "design":
        normalized = "phase_2"
    elif normalized == "scenario_mel":
        normalized = "phase_3"
    elif normalized == "evaluation":
        normalized = "phase_4"

    if normalized in {"phase_0", "phase_1", "phase_2", "phase_3", "phase_4"}:
        return normalized
    return None


PHASE_LABELS = {
    "phase_0": "Phase 0 – Entry & Orchestration",
    "phase_1": "Phase 1 – Analysis & Objectives",
    "phase_2": "Phase 2 – Exercise / Game Design",
    "phase_3": "Phase 3 – Scenario, MEL & Materials",
    "phase_4": "Phase 4 – Evaluation & Improvement",
}


def build_project_memory_context(project_id: str) -> Optional[str]:
    """
    Build a compact system message from the stored project memory.
    """
    memory = load_project_memory(project_id)

    # check if there is anything at all
    has_content = any(
        isinstance(memory.get(k), list) and memory.get(k) for k in PHASE_LABELS.keys()
    )
    if not has_content:
        return None

    parts: List[str] = [
        "For this exercise project, SmartEx has stored internal memory notes from previous steps.",
        "Use these notes to stay consistent with past decisions and avoid restarting from zero.",
        ""
    ]

    for key, label in PHASE_LABELS.items():
        items = memory.get(key, [])
        if not isinstance(items, list) or not items:
            continue

        # limit to last N items per phase to control prompt size
        last_items = items[-8:]

        parts.append(label + ":")
        for bullet in last_items:
            parts.append(f"- {bullet}")
        parts.append("")

    return "\n".join(parts)


def update_project_memory_from_reply(
    client: OpenAI, project_id: str, mode: str, reply_text: str
) -> None:
    """
    After each assistant reply, create very short memory notes and
    store them under the appropriate phase for this project.
    """
    phase_key = map_mode_to_phase_key(mode)
    if phase_key is None:
        return

    text = (reply_text or "").strip()
    if len(text) < 40:
        # too short, probably not worth storing
        return

    phase_label = PHASE_LABELS.get(phase_key, phase_key)

    system_msg = (
        "You are an assistant that compresses the reply of an emergency exercise design "
        "assistant (SmartEx) into short, persistent memory notes. "
        "These notes will be reused later to keep the exercise design consistent.\n\n"
        "Rules:\n"
        "- Write 1 to 5 very short bullet points.\n"
        "- Focus on stable decisions, objectives, exercise structures, key scenario elements, "
        "or important constraints that will remain valid later.\n"
        "- Do NOT repeat all details; be concise.\n"
        "- If there is truly nothing relevant to store, output exactly: NO_MEMORY."
    )

    user_msg = (
        f"Current phase: {phase_label}.\n\n"
        f"Assistant reply:\n{reply_text}\n\n"
        "Now produce the memory notes as described."
    )

    try:
        resp = client.chat.completions.create(
            model=DEFAULT_MODEL,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.1,
        )
        mem_text = (resp.choices[0].message.content or "").strip()
    except Exception:
        return

    if not mem_text:
        return
    if mem_text.strip().upper() == "NO_MEMORY":
        return

    lines = [ln.strip() for ln in mem_text.splitlines() if ln.strip()]
    bullets: List[str] = []

    for line in lines:
        if line.upper() == "NO_MEMORY":
            return
        if line.startswith("-"):
            line = line[1:].strip()
        # remove leading numbering if present (e.g., "1. text")
        if line[:2].isdigit() and line[1] == ".":
            line = line[2:].strip()
        if line:
            bullets.append(line)

    if not bullets:
        return

    memory = load_project_memory(project_id)
    phase_list = memory.get(phase_key)
    if not isinstance(phase_list, list):
        phase_list = []
    phase_list.extend(bullets)
    # keep only last 30 entries per phase
    phase_list = phase_list[-30:]
    memory[phase_key] = phase_list
    memory["last_updated"] = datetime.utcnow().isoformat() + "Z"
    save_project_memory(project_id, memory)


# -----------------------------
# Mode instructions
# -----------------------------

def build_mode_instruction(mode: str) -> str:
    """
    Return a concise instruction based on the selected SmartEx phase/mode.
    We support both the new 'phase_0'..'phase_4' labels and legacy modes 
    for backward compatibility.
    """
    normalized = (mode or "").strip().lower()

    # Map legacy modes to phases for safety
    if normalized == "new_exercise":
        normalized = "phase_0"
    elif normalized == "objectives":
        normalized = "phase_1"
    elif normalized == "design":
        normalized = "phase_2"
    elif normalized == "scenario_mel":
        normalized = "phase_3"
    elif normalized == "evaluation":
        normalized = "phase_4"

    if normalized == "phase_0":
        return (
            "You are in Phase 0 – Entry & Orchestration. "
            "Your job is to quickly understand the user's context and maturity, "
            "by asking a few targeted questions, and then recommend which phase "
            "button (Phase 1, 2, 3 or 4) they should click next.\n\n"
            "If the user message is empty or very short, do NOT assume detailed context. "
            "Ask for a brief description of their organisation, main hazard or scenario family, "
            "key stakeholders, main objectives and relevant constraints. "
            "Do not perform the detailed work of later phases here. "
            "At the end of your answer, explicitly state which phase you recommend next and why."
        )

    if normalized == "phase_1":
        return (
            "You are now strictly in Phase 1 – Analysis & Objectives. "
            "Do NOT go to later phases and do NOT restart Phase 0. "
            "Assume the user wants to clarify context, risks, stakeholders, constraints, "
            "learning objectives and capabilities to be tested. "
            "Ask targeted, numbered questions in small blocks, and provide structured "
            "summaries of what you understand. "
            "At the end, propose a concise set of exercise objectives and capabilities. "
            "You may suggest moving to Phase 2, but only as a recommendation: "
            "the user will explicitly click the Phase 2 button in the UI."
        )

    if normalized == "phase_2":
        return (
            "You are now strictly in Phase 2 – Exercise / Game Design. "
            "Do NOT go back to Phase 1 and do NOT restart generic analysis questions, "
            "unless the user explicitly asks to revisit objectives. "
            "Assume that the main context and objectives are already known, or ask only "
            "the minimum clarifications needed. "
            "Focus on designing the exercise/game format, agenda, roles, timing, mechanics "
            "and simple governance arrangements (e.g. project team, steering). "
            "Structure your answer clearly (e.g. duration, agenda blocks, roles, time handling, "
            "inject channels) and keep everything framed as Phase 2 work."
        )

    if normalized == "phase_3":
        return (
            "You are now strictly in Phase 3 – Scenario, MEL & Materials. "
            "Do NOT redo Phase 1 or Phase 2, except for very short clarifications. "
            "Assume that a basic context and exercise objectives exist, from which you can "
            "build a narrative scenario and a Master Events List (MEL). "
            "Produce a scenario along a timeline and a set of injects. For each inject, "
            "include at least: time (relative or absolute), description, intended recipients, "
            "expected decisions/actions and possible evaluation notes. "
            "If the user mentions uploaded or reference documents, explicitly align the "
            "scenario/MEL with them where possible."
        )

    if normalized == "phase_4":
        return (
            "You are now strictly in Phase 4 – Evaluation & Improvement. "
            "Do NOT design new scenarios or structures here; instead, focus on how to "
            "evaluate the exercise and turn results into improvements. "
            "Help the user define evaluation criteria, observation points, an After Action "
            "Review (AAR) structure and an improvement plan aligned with PDCA. "
            "If the user already ran an exercise, help them reconstruct key events, "
            "identify strengths/weaknesses and formulate clear, actionable "
            "lessons learned and corrective actions."
        )

    # Fallback if mode is unknown
    return (
        "The requested mode or phase is unclear. Ask the user a short clarifying question "
        "to understand whether they want to work on: Phase 0 (from scratch), Phase 1 "
        "(Analysis & Objectives), Phase 2 (Exercise / Game Design), Phase 3 (Scenario & MEL) "
        "or Phase 4 (Evaluation & Improvement), then proceed accordingly."
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

    # Build additional context: global KB, uploaded docs, project info, project memory
    global_kb_ctx = build_global_kb_context()
    uploaded_docs_ctx = build_uploaded_docs_context(project_id)
    project_memory_ctx = build_project_memory_context(project_id)

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
    if project_memory_ctx:
        messages.append({"role": "system", "content": project_memory_ctx})
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

    # Update project memory based on this reply (best-effort, errors ignored)
    try:
        update_project_memory_from_reply(client, project_id, request.mode, reply_text)
    except Exception:
        pass

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
