"""
Agent — the single unit of a Multi-Agent System.

Mirrors the agent spec format used by cantante (configs/setups/*.yaml) so that
an AutoMas MAS definition can be exported as a cantante config with zero
structural changes.
"""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Agent:
    """
    A single node in the MAS graph.

    Parameters
    ----------
    name : str
        Unique identifier for this agent within the MAS.
    system_prompt : str
        The instruction prompt.  In a cantante optimization run this is the
        "initial prompt" that the optimizer will evolve.
    input_vars : list[str]
        Variable names this agent reads from the shared MAS state.
    output_vars : list[str]
        Variable names this agent writes into the shared MAS state.
        The model is expected to return valid JSON with exactly these keys.
    task_description : str, optional
        Human-readable description used by cantante's meta-LLM when generating
        prompt mutations.  Falls back to the agent name if not provided.
    tools : list[str]
        Tool names (registered in cantante's tool registry) this agent may use.
        Leave empty for prompt-only agents.
    max_tokens : int
        Max tokens the underlying LLM is allowed to generate per call.
    """

    name: str
    system_prompt: str
    input_vars: List[str] = field(default_factory=list)
    output_vars: List[str] = field(default_factory=list)
    task_description: Optional[str] = None
    tools: List[str] = field(default_factory=list)
    max_tokens: int = 1024

    def to_cantante_dict(self) -> dict:
        """Serialize to a cantante-compatible agent config dict."""
        return {
            "name": self.name,
            "task_description": self.task_description or self.name,
            "input_vars": self.input_vars,
            "output_vars": self.output_vars,
            "tools": self.tools,
            "max_tool_calls": 0,
        }
