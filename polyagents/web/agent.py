"""The chat agent behind the web UI — Claude + polyagents trading tools.

Skills are discoverable: every ``skills/<id>/SKILL.md`` is registered and shown
in the UI for the user to select. The selected skills' instructions are composed
into the agent's system prompt (the tool surface is shared). Add a skill = drop a
new ``SKILL.md`` folder; it appears in the picker automatically.

Needs ``ANTHROPIC_API_KEY`` (the agent reasons via Claude); the tools themselves
are deterministic / paper-only.
"""
from __future__ import annotations

from pathlib import Path

from langchain_core.tools import StructuredTool

from polyagents import mcp_server
from polyagents.default_config import DEFAULT_CONFIG

_SKILLS_DIR = Path(__file__).resolve().parents[2] / "skills"

# Reuse the exact MCP tool functions so the web agent and an Alpha DevBox host
# expose an identical surface (one source of truth).
_TOOL_FUNCS = [
    mcp_server.scan_markets,
    mcp_server.market_snapshot,
    mcp_server.find_similar_markets,
    mcp_server.size_position,
    mcp_server.paper_execute,
    mcp_server.portfolio_status,
    mcp_server.settle_markets,
    mcp_server.pnl_report,
    mcp_server.evaluation_report,
]


def build_tools() -> list:
    return [StructuredTool.from_function(fn) for fn in _TOOL_FUNCS]


# ----- skills registry -------------------------------------------------------

def _parse_skill(text: str) -> tuple[str, str, str]:
    """Return (name, description, body) from a SKILL.md (--- frontmatter ---)."""
    name = desc = ""
    body = text
    if text.startswith("---"):
        parts = text.split("---", 2)
        if len(parts) == 3:
            front, body = parts[1], parts[2]
            for line in front.splitlines():
                if line.lower().startswith("name:"):
                    name = line.split(":", 1)[1].strip()
                elif line.lower().startswith("description:"):
                    desc = line.split(":", 1)[1].strip()
    return name, desc, body.strip()


def list_skills() -> list[dict]:
    """All registered skills: id (folder), name, description, body."""
    out: list[dict] = []
    if not _SKILLS_DIR.exists():
        return out
    for d in sorted(p for p in _SKILLS_DIR.iterdir() if p.is_dir()):
        f = d / "SKILL.md"
        if not f.exists():
            continue
        try:
            name, desc, body = _parse_skill(f.read_text(encoding="utf-8"))
        except OSError:
            continue
        out.append({"id": d.name, "name": name or d.name, "description": desc, "body": body})
    return out


def _compose_prompt(selected_ids: list[str] | None) -> str:
    skills = list_skills()
    if selected_ids:
        chosen = [s for s in skills if s["id"] in selected_ids]
        skills = chosen or skills
    if not skills:
        return "You are a disciplined Polymarket trading assistant. Use the tools provided."
    if len(skills) == 1:
        return skills[0]["body"]
    header = ("You have multiple skills enabled. Use the one that fits the user's "
              "request; respect the most restrictive discipline when they overlap.\n\n")
    return header + "\n\n---\n\n".join(f"## SKILL: {s['name']}\n{s['body']}" for s in skills)


# ----- agent -----------------------------------------------------------------

def build_agent(selected_ids: list[str] | None = None, llm=None):
    """Compile the ReAct agent (Claude + tools + selected skills' prompt)."""
    from langgraph.prebuilt import create_react_agent

    if llm is None:
        from langchain_anthropic import ChatAnthropic

        llm = ChatAnthropic(
            model=DEFAULT_CONFIG["anthropic_model"],
            temperature=DEFAULT_CONFIG.get("anthropic_temperature", 0.0),
        )
    return create_react_agent(llm, build_tools(), prompt=_compose_prompt(selected_ids))
