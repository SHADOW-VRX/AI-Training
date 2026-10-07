import os
import time

from dotenv import load_dotenv
from huggingface_hub import InferenceClient


# ============================================================
# LOAD ENVIRONMENT
# ============================================================

load_dotenv()


# ============================================================
# MODELS
# ============================================================

# Primary model
MODEL_1 = "openai/gpt-oss-120b"

# Smaller fallback/comparison model
MODEL_2 = "google/gemma-2-2b-it"

# Change this when comparing models
MODEL_ID = MODEL_1


# ============================================================
# SETTINGS
# ============================================================

REQUEST_TIMEOUT = 60
MAX_TOKENS = 500
TEMPERATURE = 0.7


# ============================================================
# SYSTEM PROMPT
# ============================================================

SYSTEM_PROMPT = """
You are a helpful study assistant.

Create exactly 5 study flashcards.

Every flashcard must contain:
- One question
- One short answer

The answers must be concise and suitable for a second-year
computer science student.

Do not provide an introduction.
Do not provide a conclusion.
Do not provide extra explanations.

Use exactly this format:

Flashcard 1
Question: ...
Answer: ...

Flashcard 2
Question: ...
Answer: ...

Flashcard 3
Question: ...
Answer: ...

Flashcard 4
Question: ...
Answer: ...

Flashcard 5
Question: ...
Answer: ...
"""


# ============================================================
# TOKEN
# ============================================================

def get_token():
    token = os.getenv("HF_TOKEN", "").strip()

    if not token:
        raise RuntimeError(
            "HF_TOKEN is missing.\n"
            "Create a .env file and add:\n\n"
            "HF_TOKEN=your_real_token_here"
        )

    if not token.startswith("hf_"):
        raise RuntimeError(
            "HF_TOKEN does not look like a Hugging Face token.\n"
            "It should normally start with hf_."
        )

    return token


# ============================================================
# CLIENT
# ============================================================

def create_client():
    token = get_token()

    return InferenceClient(
        api_key=token,
        timeout=REQUEST_TIMEOUT,
    )


# ============================================================
# PROMPT
# ============================================================

def create_prompt(topic):
    return f"""
Create exactly 5 study flashcards about:

{topic}

Rules:
1. Create exactly 5 flashcards.
2. Each flashcard must have one question.
3. Each flashcard must have one short answer.
4. Keep answers concise.
5. Use second-year computer science level.
6. Do not add anything before Flashcard 1.
7. Do not add anything after Flashcard 5.

Use exactly:

Flashcard 1
Question: ...
Answer: ...

Flashcard 2
Question: ...
Answer: ...

Flashcard 3
Question: ...
Answer: ...

Flashcard 4
Question: ...
Answer: ...

Flashcard 5
Question: ...
Answer: ...
"""


# ============================================================
# RESPONSE EXTRACTION
# ============================================================

def extract_text(response):
    if response is None:
        return ""

    if hasattr(response, "choices") and response.choices:

        choice = response.choices[0]

        if hasattr(choice, "message"):

            message = choice.message

            if message is not None:

                content = getattr(message, "content", "")

                if isinstance(content, str):
                    return content.strip()

                if isinstance(content, list):

                    parts = []

                    for item in content:

                        if isinstance(item, dict):

                            text = item.get("text")

                            if text:
                                parts.append(str(text))

                    return "\n".join(parts).strip()

    if isinstance(response, dict):

        choices = response.get("choices")

        if isinstance(choices, list) and choices:

            choice = choices[0]

            if isinstance(choice, dict):

                message = choice.get("message")

                if isinstance(message, dict):

                    content = message.get("content")

                    if isinstance(content, str):
                        return content.strip()

        for key in (
            "generated_text",
            "text",
            "content",
        ):

            value = response.get(key)

            if isinstance(value, str) and value.strip():
                return value.strip()

    generated_text = getattr(
        response,
        "generated_text",
        "",
    )

    if isinstance(generated_text, str):
        return generated_text.strip()

    return ""


# ============================================================
# RUN MODEL
# ============================================================

def generate_flashcards(topic, model_id):

    print()
    print("=" * 60)
    print("MODEL REQUEST")
    print("=" * 60)
    print(f"Model: {model_id}")
    print(f"Topic: {topic}")
    print()
    print("Connecting to Hugging Face...")
    print("Please wait...")
    print()

    start = time.perf_counter()

    try:

        client = create_client()

        response = client.chat.completions.create(

            model=model_id,

            messages=[
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT,
                },
                {
                    "role": "user",
                    "content": create_prompt(topic),
                },
            ],

            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
        )

        result = extract_text(response)

        elapsed = time.perf_counter() - start

        if not result:

            print("ERROR: Hugging Face returned no text.")

            return False

        print()
        print("=" * 60)
        print("GENERATED FLASHCARDS")
        print("=" * 60)
        print()
        print(result)

        print()
        print("=" * 60)
        print("RESULT")
        print("=" * 60)
        print(f"Model: {model_id}")
        print(f"Response time: {elapsed:.2f} seconds")
        print("=" * 60)

        return True

    except Exception as error:

        elapsed = time.perf_counter() - start

        print()
        print("=" * 60)
        print("REQUEST FAILED")
        print("=" * 60)
        print(f"Model: {model_id}")
        print(f"Time waited: {elapsed:.2f} seconds")
        print()
        print(f"Error type: {type(error).__name__}")
        print(f"Error: {error}")
        print("=" * 60)

        return False


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 60)
    print("HUGGING FACE FLASHCARD GENERATOR")
    print("=" * 60)
    print()
    print(f"Model 1: {MODEL_1}")
    print(f"Model 2: {MODEL_2}")
    print()
    print(f"Current model: {MODEL_ID}")
    print()

    try:
        token = get_token()

        print("HF_TOKEN: detected")
        print(f"Token length: {len(token)}")

    except Exception as error:

        print()
        print("CONFIGURATION ERROR")
        print("=" * 60)
        print(error)
        print("=" * 60)

        return

    print()

    topic = input("Enter a topic: ").strip()

    if not topic:

        print()
        print("ERROR: Topic cannot be empty.")

        return

    success = generate_flashcards(
        topic,
        MODEL_ID,
    )

    if success:

        print()
        print("Done.")

    else:

        print()
        print("The selected model did not return a response.")
        print()
        print("You can try MODEL_2 by changing:")
        print()
        print("MODEL_ID = MODEL_1")
        print()
        print("to:")
        print()
        print("MODEL_ID = MODEL_2")


if __name__ == "__main__":
    main()
