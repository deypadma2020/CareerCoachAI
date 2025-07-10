import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
from chat_utils import load_chat_history, save_chat_history
from chatbot import generate_response

st.set_page_config(page_title="CareerCoachAI", layout="wide")
st.title("🎯 CareerCoachAI - Your Interview Prep Partner")

# Load chat history once
if "chat_history" not in st.session_state:
    st.session_state.chat_history = load_chat_history()

# Toggle for memory highlights
if "show_highlights" not in st.session_state:
    st.session_state.show_highlights = True

with st.sidebar:
    st.checkbox("💡 Highlight Memory Matches", value=True, key="show_highlights")

with st.chat_message("system"):
    st.markdown("Ask me anything about tech interviews!")

# Display helper with memory indicator
def display_ai_response(response, source):
    if "Matched from previous chat" in response and st.session_state.show_highlights:
        parts = response.split("\n\n")
        st.markdown(parts[0])

        st.markdown("---")
        if "fuzzy" in parts[1].lower():
            st.markdown(f":orange[💾 {parts[1].strip()}]", unsafe_allow_html=True)
        else:
            st.markdown(f":green[💾 {parts[1].strip()}]", unsafe_allow_html=True)
    else:
        st.markdown(response)

    if source and st.session_state.show_highlights:
        st.caption(f"🧠 Context detected via: {source}")

# Display full chat history
for msg in st.session_state.chat_history:
    if isinstance(msg, HumanMessage):
        with st.chat_message("human"):
            st.markdown(msg.content)
    elif isinstance(msg, AIMessage):
        with st.chat_message("ai"):
            display_ai_response(msg.content, None)

# Handle new user input
user_input = st.chat_input("Your question...")

if user_input:
    # Show user's message
    st.session_state.chat_history.append(HumanMessage(content=user_input))
    with st.chat_message("human"):
        st.markdown(user_input)

    # Get response + metadata
    response, source = generate_response(user_input, st.session_state.chat_history)

    # Append AI response to history and show
    st.session_state.chat_history.append(AIMessage(content=response))
    with st.chat_message("ai"):
        display_ai_response(response, source)

    # Save updated history
    save_chat_history(st.session_state.chat_history)
