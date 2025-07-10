import os
import json
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

HISTORY_FILE = "history.json"

def load_chat_history():
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r") as f:
            raw = json.load(f)
        return [deserialize_message(m) for m in raw]
    return [SystemMessage(content="You are an AI interview coach helping users prepare for tech interviews in data, AI, and software roles.")]

def save_chat_history(history):
    with open(HISTORY_FILE, "w") as f:
        json.dump([serialize_message(m) for m in history], f, indent=2)

def serialize_message(msg):
    return {"type": msg.__class__.__name__, "content": msg.content}

def deserialize_message(msg):
    cls = {"SystemMessage": SystemMessage, "HumanMessage": HumanMessage, "AIMessage": AIMessage}[msg["type"]]
    return cls(content=msg["content"])
