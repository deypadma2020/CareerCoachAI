import os
import json
import numpy as np
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts.chat import ChatPromptTemplate, MessagesPlaceholder

from chat_utils import load_chat_history, save_chat_history
from embedding_utils import find_similar_response, embed_text, save_embedding_entry
from log_utils import log_context_classification

load_dotenv()

# Model setup
model = ChatAnthropic(model="claude-3-5-sonnet-20240620", temperature=0)

# Prompt template
chat_template = ChatPromptTemplate.from_messages([
    ("system", "You are an AI interview coach helping users prepare for tech interviews in data, AI, and software roles."),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{query}")
])

CACHE_FILE = "classification_cache.json"

# Load classification cache
def load_classification_cache():
    if not os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "w") as f:
            json.dump({}, f)
        return {}
    with open(CACHE_FILE, "r") as f:
        return json.load(f)

# Save classification cache
def save_classification_cache(cache):
    with open(CACHE_FILE, "w") as f:
        json.dump(cache, f, indent=2)

# Fuzzy match from cache
def get_fuzzy_cached_classification(query, cache, threshold=0.88):
    if not cache:
        return None, False, None

    query_vec = embed_text(query)
    keys = list(cache.keys())
    key_vecs = [cache[q]["embedding"] for q in keys if isinstance(cache[q], dict) and "embedding" in cache[q]]

    if not key_vecs:
        return None, False, None

    scores = cosine_similarity([query_vec], key_vecs)[0]
    max_idx = int(np.argmax(scores))
    max_score = float(scores[max_idx])

    if max_score >= threshold:
        matched_query = keys[max_idx]
        print(f"[Cache Hit - Fuzzy Match] Matched with: '{matched_query}' (score: {max_score:.2f})")
        return cache[matched_query]["in_context"], True, max_score

    return None, False, max_score

# Context classifier with cache, fuzzy, LLM
def is_career_context_llm(query, cache):
    query_lower = query.strip().lower()

    if query_lower in cache:
        print("[Cache Hit - Exact Match]")
        label = "in_context" if cache[query_lower]["in_context"] else "out_of_context"
        log_context_classification(query, label, "cache_exact", None, response_type="memory_exact")
        return cache[query_lower]["in_context"], "Exact match", None

    cached_value, matched, score = get_fuzzy_cached_classification(query_lower, cache)
    if matched:
        label = "in_context" if cached_value else "out_of_context"
        log_context_classification(query, label, "cache_fuzzy", score, response_type="memory_fuzzy")
        return cached_value, f"Fuzzy match (score: {score:.2f})", score

    print("[LLM Classification Triggered]")
    classification_prompt = f"""
You are an assistant for a career and interview preparation chatbot.

Classify the following user question as either:
- IN_CONTEXT: if it relates to interview preparation, job roles, resume building, HR questions, salary negotiation, or technical topics that are commonly part of interviews. These include programming, data structures, algorithms, databases, machine learning, deep learning, linear algebra, probability, statistics, system design, and computer science fundamentals.
- OUT_OF_CONTEXT: if it's about fashion, food, drink suggestions, health routines, personal lifestyle, emotions, or topics unrelated to interviews or job preparation.

Respond with only IN_CONTEXT or OUT_OF_CONTEXT.

Here are some examples:
- "What is Big O notation?" → IN_CONTEXT
- "Tips for HR interview?" → IN_CONTEXT
- "What is linear algebra?" → IN_CONTEXT
- "What are joins in SQL?" → IN_CONTEXT
- "Should I drink tea before my interview?" → OUT_OF_CONTEXT
- "What shoes should I wear for interview?" → OUT_OF_CONTEXT
- "Tell me about your family." → OUT_OF_CONTEXT

Now classify the following question:
"{query}"
"""
    classification_response = model.invoke(classification_prompt)
    decision = classification_response.content.strip().upper()
    result = decision == "IN_CONTEXT"

    query_vec = embed_text(query)
    cache[query_lower] = {
        "in_context": result,
        "embedding": query_vec
    }
    save_classification_cache(cache)

    label = "in_context" if result else "out_of_context"
    log_context_classification(query, label, "llm_classification", None, response_type="llm_classified")

    return result, "LLM classification", None

# Main response handler
def generate_response(user_input, chat_history):
    classification_cache = load_classification_cache()
    is_contextual, context_source, fuzzy_score = is_career_context_llm(user_input, classification_cache)

    if not is_contextual:
        return (
            "🛑 This chatbot is designed to help with career and interview-related questions. "
            "Please ask something relevant to job preparation.",
            context_source
        )

    matched_response, score = find_similar_response(user_input)
    if matched_response:
        log_context_classification(user_input, "in_context", "embedding_memory", score, response_type="memory_fuzzy")
        return (
            f"{matched_response}\n\n💾 (_Matched from previous chat, similarity: {score:.2f}_)",
            context_source
        )

    prompt = chat_template.invoke({
        "chat_history": chat_history,
        "query": user_input
    })
    response = model.invoke(prompt)
    answer = response.content

    query_vec = embed_text(user_input)
    save_embedding_entry({
        "query": user_input,
        "response": answer,
        "embedding": query_vec
    })

    log_context_classification(user_input, "in_context", "llm_generated", None, response_type="llm_generated")

    return answer, context_source
