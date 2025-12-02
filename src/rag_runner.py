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
        ("system", """Rewrite the user's question so it makes sense 
        in the context of the conversation. Do NOT answer. Only rewrite."""),
        ("human", "{input}")
    ])
    rewrite_chain = rewrite_prompt | llm | RunnableLambda(lambda x: x.content.strip())



    answer_prompt = ChatPromptTemplate.from_messages([
        ("system", """You answer ONLY using the retrieved documents.
        If the answer is not there, say you don't know."""),
        ("human", "Question: {question}\n\nContext:\n{context}\n\nAnswer:")
    ])


    def build_context(docs):
        return "\n\n---\n".join(d.page_content for d in docs)


    rag_chain = (

        RunnableLambda(lambda x: {
            "question_rewritten":
                rewrite_chain.invoke({"input": x["question"], "history": x["history"]})
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
        # 4. Generate answer
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
        deduped_sources = list({(s["title"], s["url"]): s for s in sources}.values())

    # update memory
    chat.history.append(HumanMessage(content=user_input))
    chat.history.append(AIMessage(content=answer))

    return {
        "answer": answer,
        "history": chat.history,
        "sources": deduped_sources
    }
