import hashlib
import json
import logging
import os
import time

from langchain.docstore.document import Document
from langchain.document_loaders import DirectoryLoader, PyPDFLoader, TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS

from logs import (CHUNK_OVERLAP, CHUNK_SIZE, META_FILE, SLEEP_TIME,
                       VECTORSTORE_DIR, WATCH_DIR,logger)

from .config import get_embadings



def load_meta() -> dict:
    """Load metadata about file hashes."""
    if os.path.exists(META_FILE):
        with open(META_FILE, "r") as f:
            return json.load(f)
    return {}

def save_meta(meta: dict):
    """Persist metadata to disk."""
    os.makedirs(os.path.dirname(META_FILE), exist_ok=True)
    with open(META_FILE, "w") as f:
        json.dump(meta, f)

def file_hash(path: str) -> str:
    """Generate MD5 hash of a file."""
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()
    
def get_changed_files():
    """
    Determine which files in WATCH_DIR have been changed or deleted.
    Returns: (changed_files, deleted_files, current_meta)
    """
    meta = load_meta()
    current = {}
    changed, deleted = [], []

    for fname in os.listdir(WATCH_DIR):
        path = os.path.join(WATCH_DIR, fname)
        if os.path.isfile(path) and fname.endswith((".txt", ".pdf")):
            h = file_hash(path)
            current[fname] = h
            if meta.get(fname) != h:
                changed.append(fname)
    deleted = [f for f in meta if f not in current]
    return changed, deleted, current

def load_documents(path:str)->list[Document]:
    """
    load documents from folder
    """
    docs=list()
    loaders = [DirectoryLoader(path,glob="*.txt",loader_cls=TextLoader),
               DirectoryLoader(path,glob="*.pdf",loader_cls=PyPDFLoader)]
    for loader in loaders:
        docs.extend(loader.load()) 
    return docs

def chunk_documents(documents:list[Document])->list[Document]:
    
    """Split documents into chunks for embedding."""
    logger.info(f"Splitting {len(documents)} documents into chunks of size {CHUNK_SIZE} with overlap {CHUNK_OVERLAP}.")
    text_splitter = CharacterTextSplitter(chunk_size = CHUNK_SIZE, chunk_overlap= CHUNK_OVERLAP)
    return text_splitter.split_documents(documents=documents)

def update_vector_store():
    """
    This function will update the vector score when data changes inside the folder
    """
    try:
        changed, deleted, current_meta = get_changed_files()

        if not changed and not deleted:
            return

        logger.info(f"Changed: {changed} | Deleted: {deleted}")
        if os.path.exists(VECTORSTORE_DIR):
         db = FAISS.load_local(VECTORSTORE_DIR, get_embadings(), allow_dangerous_deserialization=True)
        else:
            docs = load_documents(WATCH_DIR)
            split_doc = chunk_documents(docs)
            db = FAISS.from_documents(documents=split_doc,embedding=get_embadings())
            db.save_local("data/vectorstore")
            logger.info("Vector store created successfully.")

        if deleted: # Remove deleted docs
            db.docstore._dict = {
                k: v for k, v in db.docstore._dict.items()
                if v.metadata.get("source") not in deleted
            }
            db.index_to_docstore_id = {
                i: id_ for i, id_ in db.index_to_docstore_id.items()
                if db.docstore._dict.get(id_) is not None
            }

        if changed:
            full_paths = [os.path.join(WATCH_DIR, fname) for fname in changed]
            docs = []
            for file in full_paths:
                if file.endswith(".txt"):
                    docs.extend(TextLoader(file).load())
                elif file.endswith(".pdf"):
                    docs.extend(PyPDFLoader(file).load())
            chunks = chunk_documents(docs)
            db.add_documents(chunks)

        db.save_local(VECTORSTORE_DIR)
        save_meta(current_meta)
        logger.info("Vector store updated successfully.")

    except Exception as e:
        logger.info(f"Failed to update vector Store {e}")

def watch_folder():
    """
    """
    while True:
        update_vector_store()
        time.sleep(SLEEP_TIME)