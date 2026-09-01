import os
import html
import markdown
from dotenv import load_dotenv
from flask import Flask, render_template, request, redirect, url_for
from google import genai

load_dotenv()

app = Flask(__name__)
client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

MODEL_NAME = "gemini-3.6-flash"

ACTIONS = {
    "summarize": {
        "label": "Summary",
        "title": "Here's your summary",
        "prompt": (
            "Summarize the following study notes for a student. "
            "Use short, clear paragraphs and bullet points for key facts. "
            "Do not add information that isn't in the notes.\n\n"
            "NOTES:\n{notes}"
        ),
    },
    "quiz": {
        "label": "Quiz",
        "title": "Test yourself",
        "prompt": (
            "Create a short quiz (5 questions) based on the following study "
            "notes. Number each question. After all questions, add an "
            "'Answers' section with the correct answers numbered the same "
            "way.\n\n"
            "NOTES:\n{notes}"
        ),
    },
    "better_answer": {
        "label": "Improved Answer",
        "title": "A stronger version of your answer",
        "prompt": (
            "The text below is a student's draft answer to a study "
            "question. Rewrite it to be clearer, more complete, and better "
            "organized, while keeping the original meaning. Briefly note "
            "what you changed and why at the end.\n\n"
            "DRAFT ANSWER:\n{notes}"
        ),
    },
}

def text_to_html(text: str) -> str:
    escaped = html.escape(text.strip())

    return markdown.markdown(
        escaped,
        extensions=["extra"]
    )


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/result", methods=["POST"])
def result():
    notes = request.form.get("notes", "").strip()
    action = request.form.get("action", "summarize")

    if not notes or action not in ACTIONS:
        return redirect(url_for("index"))

    config = ACTIONS[action]
    prompt = config["prompt"].format(notes=notes)

    try:
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt,
        )
        ai_text = response.text
    except Exception as exc:
        ai_text = (
            "Sorry, something went wrong while contacting the AI. "
            "Please try again in a moment."
)

    return render_template(
        "result.html",
        action_label=config["label"],
        action_title=config["title"],
        result_html=text_to_html(ai_text),
    )

if __name__ == "__main__":
    app.run(debug=True)