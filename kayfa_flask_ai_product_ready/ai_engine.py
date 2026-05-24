import os
import re
import json
from datetime import datetime, timedelta
from typing import Annotated, TypedDict, Optional, Any

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.tools import tool



MODEL_NAME = "gpt-4o-mini"


def make_llm(temperature: float = 0.3) -> ChatOpenAI:
    if not OPENAI_API_KEY or OPENAI_API_KEY == "PUT_YOUR_OPENAI_API_KEY_HERE":
        raise ValueError(
            "Please open ai_engine.py and replace PUT_YOUR_OPENAI_API_KEY_HERE with your OpenAI API key."
        )

    return ChatOpenAI(
        model=MODEL_NAME,
        temperature=temperature,
        api_key=OPENAI_API_KEY,
    )


llm = make_llm(temperature=0.3)


class KayfaState(TypedDict):
    student_name: str
    student_goal: str
    student_level: str
    available_hours: int
    known_topics: list[str]

    assessment: Optional[dict]
    selected_content: Optional[list]
    learning_plan: Optional[dict]
    progress_report: Optional[dict]

    messages: Annotated[list, add_messages]

    current_step: str
    completed_lessons: list[str]
    quiz_scores: dict


def slugify(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    return text.strip("-") or "student"


def extract_json(text: str, is_array: bool = False) -> Optional[dict | list]:
    if not text:
        return None

    clean = text.strip()

    for fence in ["```json", "```"]:
        if fence in clean:
            parts = clean.split(fence)
            if len(parts) > 1:
                clean = parts[1].split("```")[0].strip()
                break

    marker = "[" if is_array else "{"
    end_marker = "]" if is_array else "}"
    start = clean.find(marker)
    end = clean.rfind(end_marker) + 1

    if start >= 0 and end > start:
        try:
            return json.loads(clean[start:end])
        except json.JSONDecodeError:
            return None

    return None


def get_tool_result(messages: list) -> Optional[str]:
    for msg in reversed(messages):
        content = msg.content if hasattr(msg, "content") else str(msg)
        if content and isinstance(content, str) and ("{" in content or "[" in content):
            return content
    return None


def safe_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except Exception:
        return default


@tool
def assess_student_knowledge(
    student_name: str,
    goal: str,
    current_level: str,
    known_topics: str,
    available_hours: int,
) -> str:
    """
    Assess a student's current knowledge level.
    Identify knowledge gaps, strong areas, and estimated time to reach the goal.
    Return JSON only.
    """
    assessment_llm = make_llm(temperature=0.2)

    prompt = f"""
You are an expert educational assessor for Kayfa Academy.

Return ONLY a valid raw JSON object. No markdown. No explanations outside JSON.

JSON format:
{{
  "current_level": "beginner|intermediate|advanced",
  "knowledge_gaps": ["gap1", "gap2", "gap3"],
  "strong_areas": ["area1", "area2"],
  "recommended_starting_point": "where to start",
  "estimated_weeks": 16,
  "assessment_summary": "brief professional summary"
}}

Student:
- Name: {student_name}
- Goal: {goal}
- Self-reported level: {current_level}
- Known topics: {known_topics}
- Available hours per week: {available_hours}

Rules:
- Be specific to the goal.
- Do not make the answer generic.
- Keep all output in English.
"""

    response = assessment_llm.invoke([HumanMessage(content=prompt)])
    return response.content


@tool
def search_kayfa_content(
    goal: str,
    current_level: str,
    knowledge_gaps: str,
) -> str:
    """
    Search the Kayfa Academy course catalog and select the best courses.
    Return JSON array only.
    """
    catalog = {
        "Python": [
            {"id": "py001", "title": "Python for Beginners", "level": "beginner", "hours": 8},
            {"id": "py002", "title": "Python OOP and Advanced Concepts", "level": "intermediate", "hours": 10},
            {"id": "py003", "title": "Python Automation Projects", "level": "intermediate", "hours": 8},
        ],
        "Data Analysis": [
            {"id": "da001", "title": "Pandas and NumPy Fundamentals", "level": "beginner", "hours": 8},
            {"id": "da002", "title": "Data Visualization with Matplotlib and Seaborn", "level": "intermediate", "hours": 6},
            {"id": "da003", "title": "Exploratory Data Analysis", "level": "intermediate", "hours": 8},
            {"id": "da004", "title": "Real-World Data Analysis Projects", "level": "advanced", "hours": 10},
        ],
        "Machine Learning": [
            {"id": "ml001", "title": "Machine Learning Fundamentals", "level": "intermediate", "hours": 12},
            {"id": "ml002", "title": "Scikit-learn Masterclass", "level": "intermediate", "hours": 10},
            {"id": "ml003", "title": "Model Evaluation and Deployment Basics", "level": "advanced", "hours": 8},
        ],
        "Statistics": [
            {"id": "st001", "title": "Statistics for Data Science", "level": "beginner", "hours": 10},
            {"id": "st002", "title": "Probability and Hypothesis Testing", "level": "intermediate", "hours": 8},
        ],
        "SQL": [
            {"id": "sql001", "title": "SQL for Data Analysts", "level": "beginner", "hours": 8},
            {"id": "sql002", "title": "Advanced SQL and Query Optimization", "level": "intermediate", "hours": 6},
        ],
        "Business Intelligence": [
            {"id": "bi001", "title": "Power BI Dashboard Design", "level": "beginner", "hours": 8},
            {"id": "bi002", "title": "Data Storytelling for Business Decisions", "level": "intermediate", "hours": 6},
            {"id": "bi003", "title": "Tableau Dashboard Design", "level": "beginner", "hours": 8},
        ],
        "AI and LLMs": [
            {"id": "ai001", "title": "AI Fundamentals", "level": "beginner", "hours": 8},
            {"id": "ai002", "title": "Prompt Engineering Basics", "level": "beginner", "hours": 5},
            {"id": "ai003", "title": "Building AI Applications with APIs", "level": "intermediate", "hours": 10},
        ],
        "Career Readiness": [
            {"id": "cr001", "title": "Building a Professional Portfolio", "level": "beginner", "hours": 5},
            {"id": "cr002", "title": "Technical Interview Preparation", "level": "intermediate", "hours": 4},
        ],
    }

    content_llm = make_llm(temperature=0.2)

    prompt = f"""
You are a content curator for Kayfa Academy.

Return ONLY a valid raw JSON array. No markdown.

JSON format:
[
  {{
    "id": "course_id",
    "title": "course title",
    "category": "category name",
    "level": "beginner|intermediate|advanced",
    "hours": 8,
    "reason": "short reason"
  }}
]

Student:
- Goal: {goal}
- Current level: {current_level}
- Knowledge gaps: {knowledge_gaps}

Full catalog:
{json.dumps(catalog, ensure_ascii=False)}

Select 6 to 8 courses.
Choose courses that fit the student's actual goal, not only Data Analysis.
Keep all output in English.
"""

    response = content_llm.invoke([HumanMessage(content=prompt)])
    return response.content


@tool
def create_learning_plan(
    student_name: str,
    goal: str,
    current_level: str,
    available_hours: int,
    estimated_weeks: int,
    selected_content: str,
    knowledge_gaps: str,
) -> str:
    """
    Create a week-by-week personalized learning plan.
    Return JSON object only.
    """
    planner_llm = make_llm(temperature=0.4)

    prompt = f"""
You are a senior learning path designer.

Return ONLY a valid raw JSON object. No markdown.

JSON format:
{{
  "total_weeks": 16,
  "weekly_plan": [
    {{
      "week": 1,
      "theme": "theme title",
      "content_ids": ["course_id"],
      "daily_tasks": ["task 1", "task 2", "task 3"],
      "mini_project": "project description or null",
      "hours_this_week": 10
    }}
  ],
  "milestones": ["milestone 1", "milestone 2"],
  "final_project": "capstone project",
  "success_metrics": ["metric1", "metric2"]
}}

Student:
- Name: {student_name}
- Goal: {goal}
- Current level: {current_level}
- Available hours per week: {available_hours}
- Estimated weeks: {estimated_weeks}
- Knowledge gaps: {knowledge_gaps}
- Selected content: {selected_content}

Rules:
- Make the plan practical and project-based.
- Create realistic weekly progression.
- Use the selected course IDs.
- Keep all output in English.
"""

    response = planner_llm.invoke([HumanMessage(content=prompt)])
    return response.content


@tool
def analyze_progress(
    student_name: str,
    goal: str,
    total_weeks: int,
    completed_lessons: str,
    quiz_scores: str,
    milestones: str,
) -> str:
    """
    Analyze student progress and recommend plan adjustments.
    Return JSON object only.
    """
    progress_llm = make_llm(temperature=0.3)

    prompt = f"""
You are a learning progress advisor.

Return ONLY a valid raw JSON object. No markdown.

JSON format:
{{
  "progress_percent": 35.5,
  "performance": "excellent|good|needs_improvement|struggling",
  "strengths": ["strength1"],
  "weak_areas": ["area1"],
  "recommendations": ["action1", "action2", "action3"],
  "plan_adjustments": ["adjustment1"],
  "motivational_message": "motivational message in English",
  "estimated_completion": "YYYY-MM-DD"
}}

Student:
- Name: {student_name}
- Goal: {goal}
- Total plan weeks: {total_weeks}
- Completed lessons: {completed_lessons}
- Quiz scores: {quiz_scores}
- Milestones: {milestones}

Keep all output in English.
"""

    response = progress_llm.invoke([HumanMessage(content=prompt)])
    return response.content


# ============================================================
# NODES
# ============================================================
def assessment_node(state: KayfaState) -> KayfaState:
    agent = create_react_agent(llm, tools=[assess_student_knowledge])

    result = agent.invoke({
        "messages": [
            HumanMessage(
                content=(
                    "Use assess_student_knowledge to assess this student:\n"
                    f"name={state['student_name']}\n"
                    f"goal={state['student_goal']}\n"
                    f"current_level={state['student_level']}\n"
                    f"known_topics={', '.join(state['known_topics'])}\n"
                    f"available_hours={state['available_hours']}"
                )
            )
        ]
    })

    raw = get_tool_result(result["messages"])
    assessment = extract_json(raw) if raw else None

    if not assessment or "knowledge_gaps" not in assessment:
        assessment = {
            "current_level": state["student_level"],
            "knowledge_gaps": ["Core foundations", "Hands-on practice", "Portfolio projects"],
            "strong_areas": state["known_topics"],
            "recommended_starting_point": "Start with the fundamentals and build small projects every week.",
            "estimated_weeks": 12,
            "assessment_summary": "The student needs a structured project-based plan to reach the target goal.",
        }

    return {
        **state,
        "assessment": assessment,
        "current_step": "content",
        "messages": [AIMessage(content=f"Assessment done: {assessment.get('assessment_summary', '')}")],
    }


def content_node(state: KayfaState) -> KayfaState:
    assessment = state["assessment"] or {}
    agent = create_react_agent(llm, tools=[search_kayfa_content])

    result = agent.invoke({
        "messages": [
            HumanMessage(
                content=(
                    "Use search_kayfa_content to find suitable courses:\n"
                    f"goal={state['student_goal']}\n"
                    f"current_level={assessment.get('current_level', 'beginner')}\n"
                    f"knowledge_gaps={', '.join(assessment.get('knowledge_gaps', []))}"
                )
            )
        ]
    })

    raw = get_tool_result(result["messages"])
    content_list = extract_json(raw, is_array=True) if raw else None

    if not content_list:
        content_list = [
            {"id": "py001", "title": "Python for Beginners", "category": "Python", "level": "beginner", "hours": 8, "reason": "Core foundation"},
            {"id": "cr001", "title": "Building a Professional Portfolio", "category": "Career Readiness", "level": "beginner", "hours": 5, "reason": "Proof of work"},
        ]

    return {
        **state,
        "selected_content": content_list,
        "current_step": "plan",
        "messages": [AIMessage(content=f"Found {len(content_list)} courses.")],
    }


def planner_node(state: KayfaState) -> KayfaState:
    assessment = state["assessment"] or {}
    selected_content = state["selected_content"] or []
    content_titles = [course.get("title", "") for course in selected_content]
    estimated_weeks = safe_int(assessment.get("estimated_weeks"), 12)

    agent = create_react_agent(llm, tools=[create_learning_plan])

    result = agent.invoke({
        "messages": [
            HumanMessage(
                content=(
                    "Use create_learning_plan to create a plan:\n"
                    f"student_name={state['student_name']}\n"
                    f"goal={state['student_goal']}\n"
                    f"current_level={assessment.get('current_level', 'beginner')}\n"
                    f"available_hours={state['available_hours']}\n"
                    f"estimated_weeks={estimated_weeks}\n"
                    f"selected_content={json.dumps(content_titles)}\n"
                    f"knowledge_gaps={', '.join(assessment.get('knowledge_gaps', []))}"
                )
            )
        ]
    })

    raw = get_tool_result(result["messages"])
    plan = extract_json(raw) if raw else None

    if not plan or "weekly_plan" not in plan:
        plan = {
            "total_weeks": estimated_weeks,
            "weekly_plan": [
                {
                    "week": 1,
                    "theme": "Foundation setup",
                    "content_ids": [],
                    "daily_tasks": ["Set up tools", "Study fundamentals", "Complete exercises"],
                    "mini_project": "Create a simple first project",
                    "hours_this_week": state["available_hours"],
                }
            ],
            "milestones": ["Complete foundations", "Build first project", "Create portfolio proof"],
            "final_project": f"Complete portfolio-ready project for {state['student_goal']}",
            "success_metrics": ["Complete weekly tasks", "Build projects", "Explain results clearly"],
        }

    return {
        **state,
        "learning_plan": plan,
        "current_step": "progress",
        "messages": [AIMessage(content=f"Plan created: {plan.get('total_weeks')} weeks.")],
    }


def progress_node(state: KayfaState) -> KayfaState:
    completed = state.get("completed_lessons", [])
    scores = state.get("quiz_scores", {})

    if not completed and not scores:
        return {**state, "current_step": "done"}

    plan = state.get("learning_plan", {}) or {}
    agent = create_react_agent(llm, tools=[analyze_progress])

    result = agent.invoke({
        "messages": [
            HumanMessage(
                content=(
                    "Use analyze_progress to evaluate student progress:\n"
                    f"student_name={state['student_name']}\n"
                    f"goal={state['student_goal']}\n"
                    f"total_weeks={plan.get('total_weeks', 12)}\n"
                    f"completed_lessons={json.dumps(completed)}\n"
                    f"quiz_scores={json.dumps(scores)}\n"
                    f"milestones={json.dumps(plan.get('milestones', []))}"
                )
            )
        ]
    })

    raw = get_tool_result(result["messages"])
    report = extract_json(raw) if raw else None

    if not report or "progress_percent" not in report:
        avg_score = sum(scores.values()) / len(scores) if scores else 0
        estimated_completion = (datetime.now() + timedelta(weeks=8)).strftime("%Y-%m-%d")

        report = {
            "progress_percent": round((len(completed) / 8) * 100, 1),
            "performance": "good" if avg_score >= 70 else "needs_improvement",
            "strengths": ["Consistent progress"],
            "weak_areas": [quiz for quiz, score in scores.items() if score < 70],
            "recommendations": [
                "Review weaker topics before moving forward.",
                "Practice with small projects.",
                "Track weekly progress consistently.",
            ],
            "plan_adjustments": ["Add one review session for topics below 70%."],
            "motivational_message": "You are moving in the right direction. Stay consistent and keep building.",
            "estimated_completion": estimated_completion,
        }

    return {
        **state,
        "progress_report": report,
        "current_step": "done",
        "messages": [AIMessage(content=report.get("motivational_message", "Keep going."))],
    }


# ============================================================
# GRAPH
# ============================================================
def build_kayfa_graph():
    memory = MemorySaver()
    graph = StateGraph(KayfaState)

    graph.add_node("assess", assessment_node)
    graph.add_node("content", content_node)
    graph.add_node("plan", planner_node)
    graph.add_node("progress", progress_node)

    graph.add_edge(START, "assess")
    graph.add_edge("assess", "content")
    graph.add_edge("content", "plan")
    graph.add_edge("plan", "progress")
    graph.add_edge("progress", END)

    return graph.compile(checkpointer=memory)


def build_initial_state(
    student_name: str,
    student_goal: str,
    student_level: str,
    available_hours: int,
    known_topics: list[str],
    completed_lessons: Optional[list[str]] = None,
    quiz_scores: Optional[dict] = None,
) -> KayfaState:
    return {
        "student_name": student_name,
        "student_goal": student_goal,
        "student_level": student_level,
        "available_hours": available_hours,
        "known_topics": known_topics,
        "assessment": None,
        "selected_content": None,
        "learning_plan": None,
        "progress_report": None,
        "messages": [],
        "current_step": "assess",
        "completed_lessons": completed_lessons or [],
        "quiz_scores": quiz_scores or {},
    }


def run_learning_plan(
    student_name: str,
    student_goal: str,
    student_level: str,
    available_hours: int,
    known_topics: list[str],
    completed_lessons: Optional[list[str]] = None,
    quiz_scores: Optional[dict] = None,
) -> dict:
    graph = build_kayfa_graph()

    initial_state = build_initial_state(
        student_name=student_name,
        student_goal=student_goal,
        student_level=student_level,
        available_hours=available_hours,
        known_topics=known_topics,
        completed_lessons=completed_lessons,
        quiz_scores=quiz_scores,
    )

    thread_id = f"{slugify(student_name)}-{slugify(student_goal)}-{datetime.now().strftime('%Y%m%d%H%M%S')}"
    config = {"configurable": {"thread_id": thread_id}}

    final_state = graph.invoke(initial_state, config=config)

    return {
        "thread_id": thread_id,
        "student": {
            "name": final_state["student_name"],
            "goal": final_state["student_goal"],
            "level": final_state["student_level"],
            "available_hours": final_state["available_hours"],
            "known_topics": final_state["known_topics"],
        },
        "assessment": final_state.get("assessment"),
        "selected_content": final_state.get("selected_content"),
        "learning_plan": final_state.get("learning_plan"),
        "progress_report": final_state.get("progress_report"),
        "powered_by": "LangGraph + LangChain + OpenAI",
        "model": MODEL_NAME,
        "generated_at": datetime.now().isoformat(),
    }
