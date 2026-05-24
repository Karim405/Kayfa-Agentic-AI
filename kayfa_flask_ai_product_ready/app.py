import json
from pathlib import Path
from flask import Flask, render_template, request, send_file, redirect, url_for, flash

from ai_engine import run_learning_plan

app = Flask(__name__)
app.secret_key = "kayfa-demo-secret-key"

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

LATEST_RESULT_FILE = OUTPUT_DIR / "latest_result.json"


def parse_topics(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def parse_quiz_scores(raw: str) -> dict:
    """
    Expected format:
    Python Quiz:80, SQL Quiz:65, Power BI Quiz:90
    """
    scores = {}
    if not raw.strip():
        return scores

    parts = [p.strip() for p in raw.split(",") if p.strip()]
    for part in parts:
        if ":" not in part:
            continue
        name, score = part.split(":", 1)
        try:
            scores[name.strip()] = float(score.strip())
        except ValueError:
            continue

    return scores


def save_latest_result(result: dict) -> None:
    with open(LATEST_RESULT_FILE, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)


def load_latest_result() -> dict | None:
    if not LATEST_RESULT_FILE.exists():
        return None
    with open(LATEST_RESULT_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/generate", methods=["POST"])
def generate():
    try:
        student_name = request.form.get("student_name", "").strip()
        student_goal = request.form.get("student_goal", "").strip()
        student_level = request.form.get("student_level", "beginner").strip()
        available_hours = int(request.form.get("available_hours", "10"))
        known_topics = parse_topics(request.form.get("known_topics", ""))

        if not student_name or not student_goal:
            flash("Please enter student name and goal.")
            return redirect(url_for("index"))

        result = run_learning_plan(
            student_name=student_name,
            student_goal=student_goal,
            student_level=student_level,
            available_hours=available_hours,
            known_topics=known_topics,
        )

        save_latest_result(result)
        return render_template("result.html", result=result)

    except Exception as e:
        return render_template("error.html", error=str(e))


@app.route("/progress", methods=["GET", "POST"])
def progress():
    latest = load_latest_result()

    if request.method == "POST":
        try:
            student_name = request.form.get("student_name", "").strip()
            student_goal = request.form.get("student_goal", "").strip()
            student_level = request.form.get("student_level", "beginner").strip()
            available_hours = int(request.form.get("available_hours", "10"))
            known_topics = parse_topics(request.form.get("known_topics", ""))
            completed_lessons = parse_topics(request.form.get("completed_lessons", ""))
            quiz_scores = parse_quiz_scores(request.form.get("quiz_scores", ""))

            result = run_learning_plan(
                student_name=student_name,
                student_goal=student_goal,
                student_level=student_level,
                available_hours=available_hours,
                known_topics=known_topics,
                completed_lessons=completed_lessons,
                quiz_scores=quiz_scores,
            )

            save_latest_result(result)
            return render_template("result.html", result=result)

        except Exception as e:
            return render_template("error.html", error=str(e))

    return render_template("progress.html", latest=latest)


@app.route("/download-json")
def download_json():
    if not LATEST_RESULT_FILE.exists():
        flash("No result found. Generate a plan first.")
        return redirect(url_for("index"))
    return send_file(LATEST_RESULT_FILE, as_attachment=True, download_name="kayfa_learning_plan.json")


@app.route("/about")
def about():
    return render_template("about.html")


if __name__ == "__main__":
    app.run(debug=True)
