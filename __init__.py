"""
AutoMas — a clean, Promptolution-style library wrapper for cantante's
Multi-Agent System (MAS) inference and optimization.

Relationship map:
  Promptolution : CAPO  ==  AutoMas : cantante

AutoMas provides:
  - Agent          — define a single agent with prompt + IO vars
  - MAS            — compose agents into a graph
  - run_inference  — run a MAS on one input (no optimization)
  - (future) optimize — run cantante's attribution-guided optimizer

The backend is OpenAI-compatible (default: Anthropic via openai SDK compat,
or any base_url you point at). Falls back to the anthropic SDK when
ANTHROPIC_API_KEY is set and no base_url is given.
"""

from automas.agent import Agent
from automas.mas import MAS, run_inference

__all__ = ["Agent", "MAS", "run_inference"]
__version__ = "0.1.0"
