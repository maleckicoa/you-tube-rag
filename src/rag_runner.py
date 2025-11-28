import os

from langchain_chroma import Chroma

from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage

from utils import extract_metadata, format_context, trim_history

from dotenv import load_dotenv

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")




def setup_rag(API_KEY):

    emb = OpenAIEmbeddings(model="text-embedding-3-small", api_key=API_KEY)
    llm = ChatOpenAI(model="gpt-4.1", temperature=0, api_key=API_KEY)

    vectordb = Chroma(
        collection_name="youtube_captions",
        persist_directory="chroma_db",
        embedding_function=emb
    )

    #items = vectordb.get(include=["embeddings", "documents"])
    # print(items.keys(), items["ids"], items["documents"][150], items["embeddings"][0][:10])


    retriever = vectordb.as_retriever(
        search_kwargs={"k": 10}   # how many chunks to return
    )

    print("DOCUMENT COUNT IN CHROMA:", vectordb._collection.count())

    prompt = ChatPromptTemplate.from_messages([
        ("system",
        """
        You are a strict retrieval-augmented assistant called WealthMate.

        You MUST follow these rules exactly:

        1. You may ONLY answer using the provided context.
        2. First, inspect the context and determine whether it contains information
           that directly answers the user's question.
        3. If the context does NOT contain the answer, reply EXACTLY with:
           "Sorry, your question doesn't seem related to investing, could you please rephrase?"
        4. Do NOT rely on prior knowledge. Do NOT guess.
        5. If the context DOES contain the answer, answer concisely and suggest
           watching the YouTube video.

        Context:
        {context}
        """
        ),
        ("placeholder", "{history}"),
        ("human", "{question}"),
    ])

    # RAG CHAIN: retrieval → prepare prompt fields → answer + sources (+ echo question)
    rag_with_sources = (
        # 1) retrieve docs
        RunnableLambda(
            lambda x: {
                "question": x["question"],
                "history": x["history"],
                "docs": retriever.invoke(x["question"])
            }
        )
        # 2) prepare fields for prompt + sources
        | RunnableLambda(
            lambda x: {
                "question": x["question"],
                "history": x["history"],
                "context": format_context(x["docs"]),
                "docs": x["docs"],
            }
        )
        # 3) answer + sources; also return question so chat() can store it
        | RunnableParallel({
            "answer": (
                RunnableLambda(lambda x: {
                    "history": x["history"],
                    "question": x["question"],
                    "context": x["context"],
                })
                | prompt
                | llm
                | StrOutputParser()
            ),
            "sources": RunnableLambda(lambda x: extract_metadata(x["docs"])),
            "question": RunnableLambda(lambda x: x["question"]),
        })
    )

    return rag_with_sources

def rewrite_followup(question: str, history: list) -> str:
    """
    Rewrite vague follow-up questions into standalone ones
    using the last human message. No LLM involved.
    """
    if not history:
        return question

    # last human message in history
    last_human = None
    for msg in reversed(history):
        if isinstance(msg, HumanMessage):
            last_human = msg.content
            break

    if not last_human:
        return question

    # simple merge: "<last question>. Follow-up: <new question>"
    return f"{last_human}. Follow-up question: {question}"



def chat(user_input, API_KEY):

    if not hasattr(chat, "rag"):
        chat.rag = setup_rag(API_KEY=API_KEY)

    if not hasattr(chat, "history"):
        chat.history = []

    # rewrite follow-up using previous human question
    rewritten_for_retriever = rewrite_followup(user_input, chat.history)

    result = chat.rag.invoke({
        "history": chat.history,
        "question": rewritten_for_retriever
    })

    # we explicitly return "question" from the chain, so this is the rewritten one
    rewritten_question = result.get("question", rewritten_for_retriever)

    print("Rewritten question:", rewritten_question)
    answer = result["answer"]

    # update memory with the REWRITTEN question
    chat.history.append(HumanMessage(content=rewritten_question))
    chat.history.append(AIMessage(content=answer))
    chat.history[:] = trim_history(chat.history)

    deduped_sources = list({(s["title"], s["url"]): s for s in result["sources"]}.values())
    if "sorry" in answer.lower():
        deduped_sources = []

    return {
        "answer": answer,
        "history": chat.history,
        "sources": deduped_sources
    }

