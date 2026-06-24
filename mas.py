"""
MAS — Multi-Agent System container and inference runner.

For inference AutoMas does not require cantante (no LangGraph dependency).
It executes agents sequentially in topological order, routing the shared
state dict from agent to agent, each call going to the LLM backend.

The graph format (edges as (from, to) tuples with "__start__" / "__end__"
sentinels) is identical to cantante's YAML setup format so that any MAS
defined here can be dropped straight into a cantante experiment config.
"""

import json
import logging
import os
import re
from typing import Any, Dict, List, Optional, Tuple

from automas.agent import Agent

log = logging.getLogger(__name__)

Edge = Tuple[str, str]  # ("__start__", "agent_a") etc.


class MAS:
    """
    A directed graph of Agents.

    Parameters
    ----------
    agents : list[Agent]
        All agent nodes.  Order does not matter; execution order is derived
        from the edge list.
    edges : list[Edge]
        Directed edges as (source, target) tuples.
        Use "__start__" as the source of the first agent and "__end__" as the
        target of the last agent — exactly as cantante expects.

    Example
    -------
    >>> writer = Agent(name="writer", system_prompt="...", ...)
    >>> mas = MAS(agents=[writer],
    ...           edges=[("__start__", "writer"), ("writer", "__end__")])
    """

    def __init__(self, agents: List[Agent], edges: List[Edge]) -> None:
        self.agents: Dict[str, Agent] = {a.name: a for a in agents}
        self.edges = edges
        self._order = self._topological_order()

    # ── Graph helpers ─────────────────────────────────────────────────────────

    def _topological_order(self) -> List[str]:
        """Return agent names in execution order (simple linear walk)."""
        # Build successor map, excluding sentinel nodes
        succs: Dict[str, List[str]] = {name: [] for name in self.agents}
        for src, tgt in self.edges:
            if src == "__start__":
                # Find the first real node — nothing to do, we start from it
                continue
            if tgt == "__end__":
                continue
            if src in succs:
                succs[src].append(tgt)

        # Find entry nodes (reachable from __start__)
        starts = [tgt for src, tgt in self.edges if src == "__start__"]
        visited, order = set(), []

        def visit(name: str) -> None:
            if name in visited or name not in self.agents:
                return
            visited.add(name)
            order.append(name)
            for nxt in succs.get(name, []):
                visit(nxt)

        for s in starts:
            visit(s)
        return order

    def to_cantante_setup(self) -> dict:
        """Export a cantante-compatible setup dict (equivalent to a setup YAML)."""
        return {
            "edges": [{"from": s, "to": t} for s, t in self.edges],
            "agents": [a.to_cantante_dict() for a in self.agents.values()],
            "init_agent_prompt_pool": {
                name: [agent.system_prompt]
                for name, agent in self.agents.items()
            },
        }


# ── Inference ─────────────────────────────────────────────────────────────────

def run_inference(
    mas: MAS,
    inputs: Dict[str, Any],
    *,
    model: str = "claude-haiku-4-5",
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    temperature: float = 0.7,
) -> Dict[str, Any]:
    """
    Run one forward pass through the MAS and return the accumulated state dict.

    Backend selection (in priority order):
    1. If ``base_url`` is given → use OpenAI-SDK with that endpoint.
    2. If ``OPENAI_API_KEY`` env var is set and no base_url → OpenAI default.
    3. If ``ANTHROPIC_API_KEY`` env var is set → Anthropic SDK directly.
    4. Raise ``RuntimeError`` if no key found.

    Parameters
    ----------
    mas     : MAS
    inputs  : dict  — initial state variables (must cover all first-agent input_vars)
    model   : str   — model identifier (provider-specific)
    api_key : str, optional — overrides environment variable
    base_url: str, optional — custom OpenAI-compatible base URL
    temperature : float

    Returns
    -------
    dict — final MAS state (all variables produced by all agents)
    """
    state = dict(inputs)

    for agent_name in mas._order:
        agent = mas.agents[agent_name]

        # Build the user message from the agent's input_vars
        user_payload = {k: state.get(k, "") for k in agent.input_vars}
        user_content = json.dumps(user_payload, ensure_ascii=False)

        log.debug("AutoMas › agent=%s  input=%s", agent_name, user_payload)

        raw = _llm_call(
            system_prompt=agent.system_prompt,
            user_content=user_content,
            model=model,
            api_key=api_key,
            base_url=base_url,
            temperature=temperature,
            max_tokens=agent.max_tokens,
        )

        # Parse JSON output and merge into state
        parsed = _parse_json(raw, agent.output_vars)
        state.update(parsed)
        log.debug("AutoMas › agent=%s  output=%s", agent_name, parsed)

    return state


# ── LLM backend ──────────────────────────────────────────────────────────────

def _llm_call(
    *,
    system_prompt: str,
    user_content: str,
    model: str,
    api_key: Optional[str],
    base_url: Optional[str],
    temperature: float,
    max_tokens: int,
) -> str:
    """Dispatch to the appropriate LLM backend and return raw text."""

    # ── 1. OpenAI-compatible (base_url given, or OPENAI_API_KEY set) ────────
    if base_url or os.getenv("OPENAI_API_KEY"):
        return _openai_compat_call(
            system_prompt=system_prompt,
            user_content=user_content,
            model=model,
            api_key=api_key or os.getenv("OPENAI_API_KEY", ""),
            base_url=base_url,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    # ── 2. Anthropic SDK ─────────────────────────────────────────────────────
    anthropic_key = api_key or os.getenv("ANTHROPIC_API_KEY")
    if anthropic_key:
        return _anthropic_call(
            system_prompt=system_prompt,
            user_content=user_content,
            model=model,
            api_key=anthropic_key,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    raise RuntimeError(
        "AutoMas: no LLM credentials found. "
        "Set ANTHROPIC_API_KEY or OPENAI_API_KEY, or pass api_key/base_url."
    )


def _anthropic_call(
    *,
    system_prompt: str,
    user_content: str,
    model: str,
    api_key: str,
    temperature: float,
    max_tokens: int,
) -> str:
    import anthropic  # optional dep — already in ig-repost-bot requirements

    client = anthropic.Anthropic(api_key=api_key)
    resp = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        system=system_prompt,
        messages=[{"role": "user", "content": user_content}],
    )
    return resp.content[0].text.strip()


def _openai_compat_call(
    *,
    system_prompt: str,
    user_content: str,
    model: str,
    api_key: str,
    base_url: Optional[str],
    temperature: float,
    max_tokens: int,
) -> str:
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise ImportError(
            "AutoMas: install 'openai' to use an OpenAI-compatible backend. "
            "pip install openai"
        ) from exc

    kwargs: dict = {"api_key": api_key}
    if base_url:
        kwargs["base_url"] = base_url
    client = OpenAI(**kwargs)

    resp = client.chat.completions.create(
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
    )
    return resp.choices[0].message.content.strip()


# ── JSON parsing ─────────────────────────────────────────────────────────────

def _parse_json(raw: str, expected_keys: List[str]) -> dict:
    """
    Extract the JSON object from raw LLM output.
    Falls back to a dict with the raw text under the first expected key.
    """
    # Strip markdown fences if present
    text = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.MULTILINE)
    text = re.sub(r"\s*```$", "", text, flags=re.MULTILINE).strip()

    try:
        result = json.loads(text)
        if isinstance(result, dict):
            return result
    except json.JSONDecodeError:
        pass

    # Best-effort: extract first JSON object from the response
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass

    log.warning("AutoMas: could not parse JSON from LLM output; using raw text.")
    fallback_key = expected_keys[0] if expected_keys else "output"
    return {fallback_key: raw}
