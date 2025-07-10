# AI Support Agent

This project implements an AI-powered support agent using LangChain, designed to answer user queries with a focus on specific knowledge domains (like renewable energy and biographies) and general inquiries, with a fallback to human escalation. It leverages Retrieval-Augmented Generation (RAG) for knowledge-based answers and exposes its functionality via a FastAPI web API.

## Table of Contents

- [Features](#features)
- [Project Structure](#project-structure)
- [Setup and Installation](#setup-and-installation)
  - [Prerequisites](#prerequisites)
  - [Clone the Repository](#clone-the-repository)
  - [Create and Activate Virtual Environment](#create-and-activate-virtual-environment)
  - [Install Dependencies](#install-dependencies)
  - [Environment Variables](#environment-variables)
- [Preparing the Knowledge Base](#preparing-the-knowledge-base)
- [Running the Application](#running-the-application)
- [API Endpoints](#api-endpoints)
- [Running Tests](#running-tests)
  - [Unit Tests](#unit-tests)
  - [RAG Evaluation (Ragas)](#rag-evaluation-ragas)
- [Troubleshooting Common Issues](#troubleshooting-common-issues)
- [Future Enhancements](#future-enhancements)
- [Contributing](#contributing)
- [License](#license)

## Features

* **Intelligent Agent:** Uses LangChain's `OPENAI_FUNCTIONS` agent type to route queries to appropriate tools.
* **Knowledge Base (RAG):** Answers domain-specific questions (e.g., renewable energy) by retrieving information from a local vector store (FAISS) built from provided documents.
* **Dynamic Knowledge Updates:** Automatically watches a specified directory (`logs/data/docs`) for changes (new, modified, or deleted documents) and updates the vector store in the background.
* **Specific Tools:**
    * `answer_renewable_energy`: Handles queries related to renewable energy using RAG.
    * `answer_biography_query`: Designed for biography-related questions (e.g., APJ Abdul Kalam).
    * `general_query`: Provides generic responses for queries not covered by specific tools.
    * `escalate_to_human`: Routes complex or sensitive queries for human intervention.
* **FastAPI Interface:** Provides a RESTful API endpoint for easy integration and interaction.
* **Logging:** Comprehensive logging to monitor agent activity and troubleshoot issues.
* **RAG Evaluation:** Includes Ragas metrics (Faithfulness, Answer Relevancy, Context Recall) for evaluating the quality of RAG responses.

## Project Structure

├── agents/
│   ├── init.py
│   ├── agent_builder.py    # Defines agent creation and available tools.
│   ├── agent_runner.py     # Executes the agent with a given query.
│   ├── config.py           # Handles LLM and Embedding model initialization, environment variables.
│   ├── knowledge.py        # Manages document loading, chunking, vector store creation/updates.
│   └── tools.py            # Defines the specific tools the agent can use.
├── logs/
│   ├── init.py
│   ├── data/
│   │   ├── docs/           # Place your .txt and .pdf knowledge base documents here.
│   │   └── vectorstore/    # FAISS vector store and metadata (generated automatically).
│   └── logs.py             # Global logging configuration and constants.
├── main.py                 # FastAPI application entry point.
├── tests/
│   ├── init.py
│   ├── eval_dataset.jsonl  # Dataset for Ragas evaluation.
│   └── test_agent.py       # Pytest unit and RAG evaluation tests.
├── .env.example            # Example for environment variables.
├── .env                    # Your actual environment variables (ignored by Git).
├── requirements.txt        # Python dependencies.
└── README.md               # This file.

## Setup and Installation

### Prerequisites

* Python 3.8+
* An OpenAI API Key (with sufficient quota/credits for embedding generation and LLM calls).
* (Optional but recommended) A LangSmith API Key for tracing and debugging.

Create and Activate Virtual Environment
It's highly recommended to use a virtual environment to manage project dependencies.

python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

You're asking how to add a README.md file to your project. This is an excellent idea! A README.md file is crucial for any software project as it provides essential information for anyone (including your future self!) trying to understand, set up, and run your code.

A README.md file should typically be placed in the root directory of your project. In your case, this would be C:\Users\j.c.shukla\Documents\R\ (or simply R/ if we're referring to the project's base folder).

Here's a template for a comprehensive README.md file tailored to your project, along with explanations for each section:

How to Create the README.md File:
Open a new text file.

Paste the content below into the file.

Save the file as README.md in the root of your project directory (e.g., C:\Users\j.c.shukla\Documents\R\README.md).

README.md Content Template:
Markdown

# AI Support Agent

This project implements an AI-powered support agent using LangChain, designed to answer user queries with a focus on specific knowledge domains (like renewable energy and biographies) and general inquiries, with a fallback to human escalation. It leverages Retrieval-Augmented Generation (RAG) for knowledge-based answers and exposes its functionality via a FastAPI web API.

## Table of Contents

- [Features](#features)
- [Project Structure](#project-structure)
- [Setup and Installation](#setup-and-installation)
  - [Prerequisites](#prerequisites)
  - [Clone the Repository](#clone-the-repository)
  - [Create and Activate Virtual Environment](#create-and-activate-virtual-environment)
  - [Install Dependencies](#install-dependencies)
  - [Environment Variables](#environment-variables)
- [Preparing the Knowledge Base](#preparing-the-knowledge-base)
- [Running the Application](#running-the-application)
- [API Endpoints](#api-endpoints)
- [Running Tests](#running-tests)
  - [Unit Tests](#unit-tests)
  - [RAG Evaluation (Ragas)](#rag-evaluation-ragas)
- [Troubleshooting Common Issues](#troubleshooting-common-issues)
- [Future Enhancements](#future-enhancements)
- [Contributing](#contributing)
- [License](#license)

## Features

* **Intelligent Agent:** Uses LangChain's `OPENAI_FUNCTIONS` agent type to route queries to appropriate tools.
* **Knowledge Base (RAG):** Answers domain-specific questions (e.g., renewable energy) by retrieving information from a local vector store (FAISS) built from provided documents.
* **Dynamic Knowledge Updates:** Automatically watches a specified directory (`logs/data/docs`) for changes (new, modified, or deleted documents) and updates the vector store in the background.
* **Specific Tools:**
    * `answer_renewable_energy`: Handles queries related to renewable energy using RAG.
    * `answer_biography_query`: Designed for biography-related questions (e.g., APJ Abdul Kalam).
    * `general_query`: Provides generic responses for queries not covered by specific tools.
    * `escalate_to_human`: Routes complex or sensitive queries for human intervention.
* **FastAPI Interface:** Provides a RESTful API endpoint for easy integration and interaction.
* **Logging:** Comprehensive logging to monitor agent activity and troubleshoot issues.
* **RAG Evaluation:** Includes Ragas metrics (Faithfulness, Answer Relevancy, Context Recall) for evaluating the quality of RAG responses.

## Project Structure

.
├── agents/
│   ├── init.py
│   ├── agent_builder.py    # Defines agent creation and available tools.
│   ├── agent_runner.py     # Executes the agent with a given query.
│   ├── config.py           # Handles LLM and Embedding model initialization, environment variables.
│   ├── knowledge.py        # Manages document loading, chunking, vector store creation/updates.
│   └── tools.py            # Defines the specific tools the agent can use.
├── logs/
│   ├── init.py
│   ├── data/
│   │   ├── docs/           # Place your .txt and .pdf knowledge base documents here.
│   │   └── vectorstore/    # FAISS vector store and metadata (generated automatically).
│   └── logs.py             # Global logging configuration and constants.
├── main.py                 # FastAPI application entry point.
├── tests/
│   ├── init.py
│   ├── eval_dataset.jsonl  # Dataset for Ragas evaluation.
│   └── test_agent.py       # Pytest unit and RAG evaluation tests.
├── .env.example            # Example for environment variables.
├── .env                    # Your actual environment variables (ignored by Git).
├── requirements.txt        # Python dependencies.
└── README.md               # This file.


## Setup and Installation

### Prerequisites

* Python 3.8+
* An OpenAI API Key (with sufficient quota/credits for embedding generation and LLM calls).
* (Optional but recommended) A LangSmith API Key for tracing and debugging.

### Clone the Repository

```bash
git clone <your_repository_url>
cd <your_project_directory> # e.g., cd R
Create and Activate Virtual Environment
It's highly recommended to use a virtual environment to manage project dependencies.

Bash

python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
Install Dependencies
Create a requirements.txt file in your project's root directory with the following content:

langchain
langchain-openai
langchain-community
python-dotenv
fastapi
uvicorn[standard]
pydantic
pandas
numpy
ragas
datasets
faiss-cpu

Then install them:
pip install -r requirements.txt
pip install "langchain[smith]" # For LangSmith tracing

Environment Variables
Create a file named .env in the root of your project directory. This file will store your sensitive API keys and configuration.
# .env file
OPENAI_API_KEY="your_openai_api_key_here"
LANGSMITH_API_KEY="your_langsmith_api_key_here"
LANGSMITH_PROJECT="your_langsmith_project_name_here" # e.g., "AI-Support-Agent"

Important: Replace the placeholder values with your actual API keys and desired LangSmith project name.

Running the Application
Once your environment is set up and your knowledge base documents are in place:

Ensure your virtual environment is active.

Start the FastAPI application using Uvicorn from the project root:

uvicorn main:app --reload

The --reload flag is useful for development as it restarts the server automatically when code changes.

Upon startup, a background thread will begin monitoring the logs/data/docs/ folder. On the first run, it will process your documents, generate embeddings, and build the FAISS vector store in logs/data/vectorstore/.

The application will be accessible at http://127.0.0.1:8000.

API Endpoints
You can interact with the agent via the /support endpoint.

Interactive API Docs: Access the Swagger UI at http://127.0.0.1:8000/docs to test the endpoint directly from your browser.

Example using curl:

# Query about renewable energy
curl -X POST "[http://127.0.0.1:8000/support](http://127.0.0.1:8000/support)" \
     -H "Content-Type: application/json" \
     -d '{"user_query": "What are the advantages of wind power?"}'

# Query about APJ Abdul Kalam
curl -X POST "[http://127.0.0.1:8000/support](http://127.0.0.1:8000/support)" \
     -H "Content-Type: application/json" \
     -d '{"user_query": "Who was APJ Abdul Kalam?"}'

# A general query
curl -X POST "[http://127.0.0.1:8000/support](http://127.0.0.1:8000/support)" \
     -H "Content-Type: application/json" \
     -d '{"user_query": "Tell me a fun fact."}'

# A query to escalate
curl -X POST "[http://127.0.0.1:8000/support](http://127.0.0.1:8000/support)" \
     -H "Content-Type: application/json" \
     -d '{"user_query": "I need urgent assistance with my account."}'

Running Tests
The project includes unit tests for the agent's tools and a RAG evaluation pipeline using Ragas.

Unit Tests
pytest tests/test_agent.py::test_run_support_agent
pytest tests/test_agent.py::test_biography_query
pytest tests/test_agent.py::test_general_query
pytest tests/test_agent.py::test_escalation

RAG Evaluation (Ragas)
For test_renewable_energy to run successfully, you need an eval_dataset.jsonl file.

Create tests/eval_dataset.jsonl:
This file should be in JSONL (JSON Lines) format. Each line is a JSON object with query, ground_truth, answer, and contexts fields. Populate it with queries and their expected answers/relevant contexts from your logs/data/docs content.

Example structure (you'll need to fill this with your own data):
{"query": "How does wind energy work?", "ground_truth": "Wind turbines convert the kinetic energy of wind into mechanical power, which is then used to generate electricity.", "answer": "Wind energy works by converting the kinetic energy of wind into mechanical power using wind turbines, which then generate electricity.", "contexts": ["Wind turbines convert the kinetic energy of wind into mechanical power, which is then used to generate electricity."]}

Run the RAG evaluation test:
pytest tests/test_agent.py::test_renewable_energy

Troubleshooting Common Issues
429 Too Many Requests / insufficient_quota: This means your OpenAI account has run out of credits or has billing issues.

Solution: Go to https://platform.openai.com/ -> Billing/Usage and ensure you have an active payment method and sufficient funds. Create a new API key after updating billing if necessary.

Failed to update vector Store ... No such file or directory: Occurs if the vector store is not found or corrupted during loading.

Solution: Ensure you've followed the "Preparing the Knowledge Base" steps. If it persists, delete the logs/data/vectorstore folder and restart the application to force a rebuild.

allow_dangerous_deserialization=True error: This means you need to explicitly allow dangerous deserialization for FAISS.

Solution: Add allow_dangerous_deserialization=True to all FAISS.load_local() calls in agents/knowledge.py and agents/tools.py.

LangChainDeprecationWarning: These are warnings indicating old import paths.

Solution: Update imports from langchain.vectorstores and langchain.document_loaders to langchain_community.vectorstores and langchain_community.document_loaders respectively.

Created a chunk of size X, which is longer than the specified Y: Your text splitter could not break down a piece of text to meet the desired CHUNK_SIZE.

Solution: This is a warning, usually harmless if chunks aren't excessively large. Consider increasing CHUNK_SIZE in logs.py if it impacts retrieval, or if you have very long, indivisible sections in your documents.
