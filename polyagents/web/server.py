"""FastAPI chat server — streams the polyagents trading agent to the browser.

    python -m polyagents.web            # http://127.0.0.1:8000

GET  /             → the chat UI (web/static/index.html)
GET  /api/skills   → registered skills (for the left-panel picker)
GET  /api/portfolio→ current paper portfolio (for the right panel)
POST /api/chat     → SSE: token / tool / tool_result / done / error
                     body: { messages:[...], skills:["polymarket-trading", ...] }

The engine (paper portfolio) persists across requests; the agent is rebuilt per
request from the selected skills. Needs ANTHROPIC_API_KEY.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, AsyncIterator

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from polyagents import mcp_server

from .agent import build_agent, list_skills

_STATIC = Path(__file__).resolve().parent / "static"

app = FastAPI(title="polyagents chat")
app.mount("/static", StaticFiles(directory=str(_STATIC)), name="static")


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(str(_STATIC / "index.html"))


@app.get("/api/skills")
async def skills() -> JSONResponse:
    return JSONResponse([{"id": s["id"], "name": s["name"], "description": s["description"]}
                         for s in list_skills()])


@app.get("/api/portfolio")
async def portfolio() -> JSONResponse:
    try:
        return JSONResponse(mcp_server.portfolio_status())
    except Exception as exc:
        return JSONResponse({"error": str(exc)})


def _sse(payload: dict) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


def _text_of(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            blk.get("text", "") if isinstance(blk, dict) and blk.get("type") == "text"
            else (blk if isinstance(blk, str) else "")
            for blk in content
        )
    return ""


def _to_lc_messages(history: list[dict]) -> list[tuple[str, str]]:
    return [("assistant" if m.get("role") == "assistant" else "user", str(m.get("content", "")))
            for m in history]


async def _stream(history: list[dict], skills: list[str]) -> AsyncIterator[str]:
    try:
        agent = build_agent(skills or None)
    except Exception as exc:
        yield _sse({"type": "error", "message": f"agent init failed: {exc}"})
        return
    try:
        async for ev in agent.astream_events({"messages": _to_lc_messages(history)}, version="v2"):
            kind = ev.get("event")
            if kind == "on_chat_model_stream":
                text = _text_of(ev["data"]["chunk"].content)
                if text:
                    yield _sse({"type": "token", "text": text})
            elif kind == "on_tool_start":
                yield _sse({"type": "tool", "name": ev.get("name"), "args": ev["data"].get("input")})
            elif kind == "on_tool_end":
                yield _sse({"type": "tool_result", "name": ev.get("name")})
        yield _sse({"type": "done"})
    except Exception as exc:
        yield _sse({"type": "error", "message": str(exc)})


@app.post("/api/chat")
async def chat(request: Request) -> StreamingResponse:
    body = await request.json()
    history = body.get("messages", [])
    skills = body.get("skills", [])
    return StreamingResponse(_stream(history, skills), media_type="text/event-stream")
