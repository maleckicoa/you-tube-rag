import os
from fastapi import FastAPI, Request
from src.rag_runner import chat
from dotenv import load_dotenv


load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")
if not API_KEY:
    print("Error: OPENAI_API_KEY is not set. Please add it to your environment.")
    exit(1)


app = FastAPI()

@app.post("/you-tube-rag")
async def you_tube_rag(request: Request):
    data = await request.json()
    user_input = data.get("user_input")
    result = chat(user_input, API_KEY)
    return {"result": result}




def main():
    print("WealthMate RAG CLI. Type 'exit' or 'quit' to leave.")

    while True:
        try:
            user_input = input("\nYour question> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue
        if user_input.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break

        try:
            res = chat(user_input, API_KEY)
        except Exception as e:
            print(f"Error while getting answer: {e}")
            continue

        answer = res.get("answer", "")
        print("\nAnswer:")
        print(answer)

        sources = res.get("sources") or []
        if sources:
            print("\nSources:")
            for i, s in enumerate(sources, 1):
                title = s.get("title", "Untitled")
                url = s.get("url", "")
                print(f"  {i}. {title} - {url}")

if __name__ == "__main__":
    main()

