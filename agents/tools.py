import logging
import os

from langchain.tools import tool
from langchain.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser

from logs import PROMPT, VECTORSTORE_DIR, logger
from .config import get_embadings, get_llm
from .knowledge import update_vector_store

logging.basicConfig(level=logging.INFO)



@tool
def answer_renewable_energy(query: str) -> dict:
    """
    Use the knowledge base (via RAG) to answer the user's query.

    Falls back to a general response if the RAG result is insufficient.
    """
    try:
        if not os.path.exists(VECTORSTORE_DIR):
            logger.info("Vector store not found. Triggering update.")
            update_vector_store()

        vector_store = FAISS.load_local(
            VECTORSTORE_DIR,
            embeddings=get_embadings(),
            allow_dangerous_deserialization=True
        )

        results = vector_store.similarity_search(query=query, k=3)
        context = " ".join([doc.page_content for doc in results]) if results else ""
        logger.info(f"Retrieved {len(results)} relevant documents for query: '{query}'")

        if not context:
            logger.warning("No relevant context found. Falling back to general query.")
            return general_query.invoke({"query": query})

        chain = PROMPT | get_llm() | StrOutputParser()
        result = chain.invoke({"context": context, "question": query})

        if "does not provide information" in result.lower() or len(result.strip()) < 10:
            logger.warning("RAG output is vague. Falling back to general query.")
            return general_query.invoke({"query": query})

        return {"input": query, "output": result}

    except Exception as e:
        logger.exception("RAG-based answering failed.")
        return {"input": query, "output": f"An error occurred: {str(e)}"}



@tool
def answer_biography_query(query: str) -> str:
    """Answer APJ biography related query."""
    return {"input": query, "output": f"This is a APJ biography response to: {query}"}

@tool
def general_query(query: str) -> str:
    """
    Generates a generic response when no specific tool applies.
    """
    parser = StrOutputParser()
    chain = get_llm() | parser
    output = chain.invoke(query)
    return {"input": query, "output": f"This is a general response to: '{output}'"}


@tool
def escalate_to_human(query: str) -> str:
    """ 
    Escalating to live agent
    """
    return {"input": query, "output": f"Escalating to human support for query: '{query}'"}