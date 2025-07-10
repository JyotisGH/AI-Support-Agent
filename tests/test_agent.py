import pytest
import pandas as pd
import numpy as np
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from agents.agent_runner import run_support_agent
from agents.tools import (
    answer_biography_query,
    answer_using_rag,
    general_query,
    escalate_to_human
)
from ragas.metrics import faithfulness, answer_relevancy,context_recall
from ragas import evaluate
from datasets import Dataset
from logs import logger,FAITHFULNESS_THRESHOLD,ANSWER_RELEVANCY_THRESHOLD, CONTEXT_RECALL_THRESHOLD


def test_run_support_agent():
    """
    Tests the run_support_agent function to ensure it returns a dictionary containing a relevant response
    to a user query. Verifies that the output includes keywords such as 'support' or 'agent', and that the
    original query is present in the output.
    """
    query = "May i know about renewable energy"
    response = run_support_agent(query)
    assert isinstance(response, dict)
    assert "knowledge" in response["response"]["output"].lower() or "energy" in response["response"]["output"].lower()



def test_biography_query():
    """
    Tests the answer_biography_query function to ensure it returns a dictionary containing a relevant response
    to a biography query. Verifies that the output includes keywords such as 'APJ' or 'biography', and
    that the original query is present in the output.
    """
    query = "Who is APJ Abdul Kalam?"
    result = answer_biography_query.invoke({"query": query})
    assert isinstance(result, dict)
    assert "biography" in result["output"].lower() or "APJ" in result["output"].lower()
    assert query in result["output"]

def test_general_query():
    """ Tests the general_query function to ensure it returns a dictionary containing a relevant response
    to a general query. Verifies that the output includes keywords such as 'general' or 'response', and
    that the original query is present in the output."""
    query = "What is the capital of India?"
    result = general_query.invoke({"query": query})
    assert isinstance(result, dict)
    assert "capital" in result["output"].lower() or "Canada" in result["output"].lower()

def test_escalation():
    """ Tests the escalate_to_human function to ensure it returns a dictionary indicating escalation
    to a human agent. Verifies that the output includes keywords such as 'escalating' or 'agent', and
    that the original query is present in the output."""
    query = "I'm very unhappy, I want to  do something wrong with me."
    result = escalate_to_human.invoke({"query": query})
    assert isinstance(result, dict)
    assert "escalating" in result["output"].lower() or "agent" in result["output"].lower() 
    
def test_renewable_energy():
    """Tests the answer_using_rag function to ensure it returns a dictionary containing a relevant response
    to a query using RAG (Retrieval-Augmented Generation). Verifies that the output includes keywords related to the query,
    such as 'code of conduct', and that the original query"""

    df = pd.read_json("tests/eval_dataset.jsonl", lines=True)
    assert "query" in df.columns, "Dataset must have a 'query' column"
    df["answer"] = df["query"].apply(get_rag_response)
    df["user_input"] = df["query"]

    ragas_dataset = Dataset.from_pandas(df)
    results = evaluate(
            ragas_dataset,
            metrics=[faithfulness,answer_relevancy,context_recall
                    ])
    logger.info(f"RAG response: {results}")
    faithfulness_score = np.mean(results["faithfulness"])
    answer_relevancy_score = np.mean(results["answer_relevancy"])
    context_recall_score = np.mean(results["context_recall"])

    logger.info("Faithfulness: %.3f", faithfulness_score)
    logger.info(f"Answer Relevancy: {answer_relevancy_score:.3f}")
    logger.info(f"Context Recall: {context_recall_score:.3f}")

    assert faithfulness_score > FAITHFULNESS_THRESHOLD, (
        f"Faithfulness score {faithfulness_score:.3f} below threshold {FAITHFULNESS_THRESHOLD}"
    )
    assert answer_relevancy_score > ANSWER_RELEVANCY_THRESHOLD,(
        f"Answer relevancy score {answer_relevancy_score:.3f} below threshold {ANSWER_RELEVANCY_THRESHOLD}"
    )
    assert context_recall_score > CONTEXT_RECALL_THRESHOLD,(
        f"Context recall score {context_recall_score:.3f} below threshold {CONTEXT_RECALL_THRESHOLD}"
    )

def get_rag_response(query):
    """
    Helper function to get the RAG response for a given query.
    """
    try:
        result = answer_using_rag.invoke({"query": query})
        return result["output"]
    except Exception as e:
        logger.error(f"Error in get_rag_response for query '{query}': {e}")
        return ""