# Study With Buddy

An AI-powered study companion built with Flask and Google's Gemini API. Paste in your notes and get back a summary, a self-test quiz, or an improved version of a draft answer.

## Features

- **Summarize** — condenses study notes into short paragraphs and bullet points
- **Short Quiz** — generates a 5-question quiz (with answers) based on your notes
- **Better Answer** — rewrites a draft answer to be clearer and better organized

## Tech Stack

- **Backend:** Python, Flask
- **AI:** Google Gemini API (`google-genai`) MODEL:gemini 3.6 flash
- **Frontend:** HTML, CSS, JavaScript (Jinja2 templates)
- **Markdown rendering:** `markdown` library for formatting AI responses

## Project Structure

├── app.py                 # Flask app & routes
├── requirements.txt       # Python dependencies
├── templates/
│   ├── index.html         # Notes input page
│   └── result.html        # AI output page
└── static/
    ├── css/style.css
    └── js/script.js


## Setup

1. Clone the repo and enter the folder:

   git clone <your-repo-url>
   cd <repo-folder>

2. Create a virtual environment and activate it:
      python -m venv .venv        # Create the .venv file
      .venv\Scripts\activate      # Windows for activate virtual environment
      source .venv/bin/activate   # It's for activate virtual environment macOS/Linux 

3. Install dependencies:
   bash
      pip install -r requirements.txt

4. Create a `.env` file in the project root with your own Gemini API key:
      GEMINI_API_KEY="your-api-key-here"

5. Run the app:
      python app.py

6. Open `http://127.0.0.1:5000` in your browser.

## Notes

This project was built as part of a beginner-level AI Engineer internship. It's intended as a learning exercise in combining a Flask backend with a generative AI API.
