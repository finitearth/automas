# AutoMas

**AutoMas** is to [cantante](https://github.com/finitearth/cantante) what [Promptolution](https://github.com/finitearth/promptolution_agentic) is to CAPO — a clean, dependency-light wrapper library for running (and eventually optimising) Multi-Agent Systems.

## Why AutoMas?

Cantante is the optimisation engine. AutoMas is the **inference runtime**: you define agents and graphs here, run them cheaply, and later hand the same definition to cantante to optimise prompts. Zero structural changes required.

## Installation

```bash
pip install automas            # Anthropic backend only
pip install automas[openai]    # + OpenAI-compatible backends
```

## Quick start

```python
from automas import Agent, MAS, run_inference

writer = Agent(
    name="writer",
    system_prompt="You are a creative writer. Given a topic, return a short paragraph in JSON: {"text": "..."}",
    input_vars=["topic"],
    output_vars=["text"],
)

mas = MAS(
    agents=[writer],
    edges=[("__start__", "writer"), ("writer", "__end__")],
)

result = run_inference(mas, inputs={"topic": "sunset over the Alps"})
print(result["text"])
```

## Concepts

| Term | Meaning |
|---|---|
|  | A single LLM node — system prompt + input/output variable names |
|  | A directed graph of agents sharing a state dict |
|  | One forward pass: input dict → output dict |

## Backend selection

AutoMas tries backends in this order:

1. OpenAI-compatible if  is given or  is set
2. Anthropic SDK if  is set
3. Raises  if neither is available

## Cantante compatibility

Every  can be exported to a cantante-ready config:

```python
import json
print(json.dumps(mas.to_cantante_setup(), indent=2))
```

The graph format (edges with  /  sentinels) is identical to cantante's YAML setup format, so migrating from inference to optimisation is zero-effort.

## License

MIT
