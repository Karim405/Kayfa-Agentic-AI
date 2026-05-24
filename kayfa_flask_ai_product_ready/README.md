# Kayfa Agentic AI Flask Product

A dynamic Flask web application powered by LangGraph, LangChain, and OpenAI.

## What it does

The user enters:
- Student name
- Learning goal
- Current level
- Available study hours per week
- Known topics

The system runs a multi-agent workflow:
1. Assessment Agent
2. Content Selection Agent
3. Planning Agent
4. Progress Analysis Agent

## How to run

1. Install requirements:

```bash
pip install -r requirements.txt
```

2. Open `ai_engine.py` and replace:

```python
OPENAI_API_KEY = "PUT_YOUR_OPENAI_API_KEY_HERE"
```

with your real OpenAI API key.

3. Run:

```bash
python app.py
```

4. Open in browser:

```text
http://127.0.0.1:5000
```

## Notes

Do not upload the project online after adding your API key.
