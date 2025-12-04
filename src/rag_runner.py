import os

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import  RunnableLambda
from langchain_core.messages import HumanMessage, AIMessage
from dotenv import load_dotenv

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")




def setup_rag(api_key):

    emb = OpenAIEmbeddings(model="text-embedding-3-small", api_key=api_key)
    llm = ChatOpenAI(model="gpt-4.1", temperature=0, api_key=api_key)

    vectordb = Chroma(
        collection_name="youtube_captions",
        persist_directory="chroma_db",
        embedding_function=emb
    )
    retriever = vectordb.as_retriever(search_kwargs={"k": 10})
    print("DOCUMENT COUNT IN CHROMA:", vectordb._collection.count())





    rewrite_prompt = ChatPromptTemplate.from_messages([
        ("system", """
            Rewrite the user's latest question ONLY if needed.

            Rules:
            1. If the user's question already makes sense as an investing/finance question,
            return it EXACTLY as it is (no changes).
            2. If the user's question is about the assistant itself (name, identity, abilities),
            return it EXACTLY as it is (no changes).
            3. If it does NOT make sense as an investing question, rewrite it so it fits
            the context of the last 3 human messages.
            4. Never answer the question. Only rewrite it.
            5. If rewriting is impossible, return: "Unable to rewrite."

            Last 3 human messages: {history}
        """),
        ("human", "{input}")
    ])
    rewrite_chain = rewrite_prompt | llm | RunnableLambda(lambda x: x.content.strip())



    answer_prompt = ChatPromptTemplate.from_messages([
        ("system", """
            You are WealthMate, a friendly, beginner-focused investing assistant.
            Your name is WealthMate. If the user asks about your name, identity,
            or abilities, answer normally without using retrieved documents.

            Otherwise, you answer ONLY using the retrieved documents.
            If the answer is not there, apologize and say you don't know the answer.
            If the answer is there, answer the question, but don't mention you are
            referring to the context. Finish answers politely by suggesting that the
            user watch the approved YouTube sources.
        """),
        ("human", "Question: {question}\n\nContext:\n{context}\n\nAnswer:")
    ])


    def build_context(docs):
        return "\n\n---\n".join(d.page_content for d in docs)



    def last_three_human(history):
        human_msgs = [m.content for m in history if m.type == "human"]
        return "\n".join(human_msgs[-3:])


    rag_chain = (

        RunnableLambda(lambda x: {
            "question_rewritten":
                rewrite_chain.invoke({"input": x["question"], 
                "history": last_three_human(x["history"])})
        })

        | RunnableLambda(lambda x: {
            "question": x["question_rewritten"],
            "docs": retriever.invoke(x["question_rewritten"])
        })

        | RunnableLambda(lambda x: {
            "question": x["question"],
            "docs": x["docs"],
            "context": build_context(x["docs"])
        })

        | RunnableLambda(lambda x: {
            "question": x["question"],
            "docs": x["docs"],
            "answer":
                llm.invoke(answer_prompt.format(question=x["question"], context=x["context"])).content
        })
    )

    return rag_chain




def chat(user_input, API_KEY):

    if not hasattr(chat, "rag"):
        chat.rag = setup_rag(API_KEY)

    if not hasattr(chat, "history"):
        chat.history = []

    result = chat.rag.invoke({
        "history": chat.history,
        "question": user_input
    })

    answer = result["answer"]
    docs = result["docs"]

    # Format sources cleanly
    sources = []
    deduped_sources = []

    for d in docs:
        meta = d.metadata or {}
        sources.append({
            "title": meta.get("title"),
            "url": meta.get("url"),
            "metadata": meta,
            "excerpt": d.page_content[:300]
        })

    if sources:
        #deduped_sources = list({(s["title"], s["url"]): s for s in sources}.values())
        seen = set()
        for s in sources:  # preserves similarity order
            key = (s["title"], s["url"])
            if key not in seen:
                seen.add(key)
                deduped_sources.append(s)

        deduped_sources = deduped_sources[:4]


    def is_unknown(a: str) -> bool:
        a_low = (a or "").lower()
        return "don't know" in a_low or "do not know" in a_low

    if not is_unknown(answer):
        # update memory only if we had a valid answer
        chat.history.append(HumanMessage(content=user_input))
        chat.history.append(AIMessage(content=answer))
    else:
        # if unknown, do not update sources either
        deduped_sources = []

    return {
        "answer": answer,
        "history": chat.history,
        "sources": deduped_sources
    }
