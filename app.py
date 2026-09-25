from flask import Flask, render_template, request
from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

client = OpenAI(
    api_key=os.getenv("HF_TOKEN"),
    base_url="https://router.huggingface.co/v1",
    timeout=10.0
)

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/ask", methods=["POST"])
def ask():
    question = request.form["question"]

    print("sending question to AI.....")

    response = client.chat.completions.create(
        model="Qwen/Qwen3.8-27B",
        messages=[
            {
                "role": "user",
                "content": question
            }
        ]
    )

    answer = response.choices[0].message.content

    print("AI responded!")

    return render_template(
        "home.html",
        question=question,
        answer=answer
    )
if __name__ == "__main__":
    app.run(debug=True)