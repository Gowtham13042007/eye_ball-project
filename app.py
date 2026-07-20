import asyncio
import os
import time
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, Request, Form
from fastapi.responses import RedirectResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.middleware.sessions import SessionMiddleware
from pydantic import BaseModel, Field
from agents import Agent, Runner

from monitor import monitor

BASE_DIR = Path(__file__).resolve().parent

app = FastAPI()

app.add_middleware(SessionMiddleware, secret_key=os.environ.get("SECRET_KEY", "dev-secret-key-change-me"))
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

load_dotenv(override=True)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
VALID_DIFFICULTIES = {"easy", "medium", "hard"}
VALID_QUESTION_COUNTS = {3, 5, 7}
DEFAULT_QUESTION_SECONDS = int(os.environ.get("QUESTION_SECONDS", "45"))
AGENT_MAX_RETRIES = int(os.environ.get("AGENT_MAX_RETRIES", "2"))
AGENT_RETRY_DELAY_SECONDS = float(os.environ.get("AGENT_RETRY_DELAY_SECONDS", "1.5"))


# ---------------------------------------------------------------------------
# Schemas + Agents (unchanged from the original quiz app)
# ---------------------------------------------------------------------------
class QuestionStructure(BaseModel):
    Question: str = Field(description="Generate a clear, grammatically correct multiple-choice question.")
    Options: list[str] = Field(description="Provide exactly 4 unique answer choices. Only one option should be correct.")
    Answer: str = Field(description="Return the exact correct option text from the four options. Do not include explanations.")


class ConceptList(BaseModel):
    Concepts: list[str] = Field(description="Provide the requested number of different concepts related to the given topic.")


Question_agent = Agent(
    name="Question Agent",
    instructions="""Generate one high-quality MCQ with exactly four options and one correct answer. Adjust difficulty to match what the prompt requests. Return the output strictly in the QuestionStructure schema.""",
    output_type=QuestionStructure,
)

Concept_agent = Agent(
    name="Concept Agent",
    instructions="Generate the exact number of distinct concepts requested for the given topic. Follow the ConceptList schema.",
    output_type=ConceptList,
)


# ---------------------------------------------------------------------------
# Retry helper -- wraps flaky LLM/network calls with a few retries so a
# transient API hiccup doesn't blow up the whole quiz-generation flow.
# ---------------------------------------------------------------------------
async def _run_with_retries(agent: Agent, prompt: str):
    last_exc = None
    for attempt in range(AGENT_MAX_RETRIES + 1):
        try:
            return await Runner.run(agent, input=prompt)
        except Exception as exc:  # noqa: BLE001 - deliberately broad, we retry any failure
            last_exc = exc
            if attempt < AGENT_MAX_RETRIES:
                await asyncio.sleep(AGENT_RETRY_DELAY_SECONDS * (attempt + 1))
    raise last_exc


# ---------------------------------------------------------------------------
# Agent helpers
# ---------------------------------------------------------------------------
async def generate_concepts(topic: str, count: int) -> list[str]:
    prompt = f"Topic: {topic}\nProvide exactly {count} different concepts related to this topic."
    result = await _run_with_retries(Concept_agent, prompt)
    concepts: ConceptList = result.final_output
    items = concepts.Concepts

    # Be defensive: the model may not return exactly `count` items.
    if len(items) > count:
        items = items[:count]
    elif len(items) < count and items:
        # Pad by cycling through what we already have rather than failing the quiz.
        i = 0
        while len(items) < count:
            items.append(items[i % len(items)])
            i += 1
    return items


async def generate_question(topic: str, concept: str, difficulty: str) -> dict:
    prompt = (
        f"Topic: {topic}\n"
        f"Concept to focus the question on: {concept}\n"
        f"Difficulty level: {difficulty}"
    )
    result = await _run_with_retries(Question_agent, prompt)
    q: QuestionStructure = result.final_output
    return {"question": q.Question, "options": q.Options, "answer": q.Answer}


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/")
async def index(request: Request):
    error = request.session.pop("error", None)
    monitor.stop()
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "request": request,
            "error": error,
        },
    )


@app.post("/generate")
async def generate(
    request: Request,
    topic: str = Form(...),
    difficulty: str = Form("medium"),
    num_questions: int = Form(5),
):
    topic = topic.strip()
    if not topic:
        request.session["error"] = "Please enter a topic."
        return RedirectResponse(url="/", status_code=303)

    if difficulty not in VALID_DIFFICULTIES:
        difficulty = "medium"
    if num_questions not in VALID_QUESTION_COUNTS:
        num_questions = 5

    try:
        concepts = await generate_concepts(topic, num_questions)
        if not concepts:
            raise ValueError("No concepts were generated for this topic.")
        questions = [await generate_question(topic, concept, difficulty) for concept in concepts]
    except Exception as exc:
        request.session["error"] = f"Could not generate quiz: {exc}"
        return RedirectResponse(url="/", status_code=303)

    request.session["topic"] = topic
    request.session["difficulty"] = difficulty
    request.session["questions"] = questions
    request.session["current"] = 0
    request.session["score"] = 0
    request.session["warnings_exceeded"] = False
    request.session["answers"] = []  # per-question review trail

    monitor.start()

    return RedirectResponse(url="/question/0", status_code=303)


@app.get("/question/{idx}")
async def question(request: Request, idx: int):
    questions = request.session.get("questions")
    if not questions:
        return RedirectResponse(url="/", status_code=303)

    status = monitor.get_status()
    if status["disqualified"]:
        request.session["warnings_exceeded"] = True
        monitor.stop()
        return RedirectResponse(url="/result", status_code=303)

    if idx < 0 or idx >= len(questions):
        monitor.stop()
        return RedirectResponse(url="/result", status_code=303)

    request.session["current"] = idx
    request.session["question_started_at"] = time.time()
    q = questions[idx]
    return templates.TemplateResponse(
        request=request,
        name="question.html",
        context={
            "request": request,
            "question": q["question"],
            "options": q["options"],
            "idx": idx,
            "total": len(questions),
            "topic": request.session.get("topic", ""),
            "warnings": status["warnings"],
            "warning_limit": status["limit"],
            "webcam_ok": status["webcam_ok"],
            "monitor_error": status["error"],
            "question_seconds": DEFAULT_QUESTION_SECONDS,
        },
    )


@app.post("/answer/{idx}")
async def answer(request: Request, idx: int, option: str = Form(default="")):
    questions = request.session.get("questions")
    if not questions or idx < 0 or idx >= len(questions):
        return RedirectResponse(url="/", status_code=303)

    # Check before grading in case the limit was hit while the question was on screen.
    status = monitor.get_status()
    if status["disqualified"]:
        request.session["warnings_exceeded"] = True
        monitor.stop()
        return RedirectResponse(url="/result", status_code=303)

    correct_answer = questions[idx]["answer"]
    is_correct = bool(option) and option == correct_answer
    if is_correct:
        request.session["score"] = request.session.get("score", 0) + 1

    answers = request.session.get("answers", [])
    answers.append(
        {
            "question": questions[idx]["question"],
            "selected": option or None,  # None means the timer ran out / skipped
            "correct": correct_answer,
            "is_correct": is_correct,
        }
    )
    request.session["answers"] = answers

    # Check again after grading, before deciding where to send the candidate next.
    status = monitor.get_status()
    if status["disqualified"]:
        request.session["warnings_exceeded"] = True
        monitor.stop()
        return RedirectResponse(url="/result", status_code=303)

    next_idx = idx + 1
    if next_idx >= len(questions):
        monitor.stop()
        return RedirectResponse(url="/result", status_code=303)
    return RedirectResponse(url=f"/question/{next_idx}", status_code=303)


@app.get("/result")
async def result(request: Request):
    questions = request.session.get("questions")
    if not questions:
        return RedirectResponse(url="/", status_code=303)

    score = request.session.get("score", 0)
    total = len(questions)
    status = monitor.get_status()
    monitor.stop()  # make sure the webcam is released once we reach the result page

    return templates.TemplateResponse(
        request=request,
        name="result.html",
        context={
            "request": request,
            "score": score,
            "total": total,
            "topic": request.session.get("topic", ""),
            "difficulty": request.session.get("difficulty", "medium"),
            "warnings_exceeded": request.session.get("warnings_exceeded", False),
            "warnings": status["warnings"],
            "warning_limit": status["limit"],
            "history": status["history"],
            "answers": request.session.get("answers", []),
        },
    )


@app.get("/restart")
async def restart(request: Request):
    monitor.stop()
    request.session.clear()
    return RedirectResponse(url="/", status_code=303)


@app.get("/api/monitor-status")
async def api_monitor_status():
    """Polled by question.html so a mid-question limit breach ends the quiz immediately."""
    return JSONResponse(monitor.get_status())


@app.post("/api/report-violation")
async def api_report_violation(request: Request):
    """Called by the front-end when it detects a non-camera violation, such
    as switching tabs or losing window focus during the quiz."""
    questions = request.session.get("questions")
    if not questions:
        return JSONResponse({"ok": False, "reason": "no active quiz"}, status_code=400)

    body = await request.json()
    reason = str(body.get("reason", "Left the quiz window"))[:120]
    status = monitor.report_external_violation(reason)
    if status.get("disqualified"):
        request.session["warnings_exceeded"] = True
    return JSONResponse({"ok": True, "status": status})


def _mjpeg_generator():
    boundary = b"--frame"
    while True:
        frame = monitor.get_frame()
        if frame is not None:
            yield (
                boundary + b"\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
            )
        time.sleep(1 / 15)  # ~15 fps is plenty for a proctoring preview


@app.get("/video-feed")
async def video_feed():
    """Live annotated webcam preview (status bar, warning count, pupil markers)."""
    return StreamingResponse(_mjpeg_generator(), media_type="multipart/x-mixed-replace; boundary=frame")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)