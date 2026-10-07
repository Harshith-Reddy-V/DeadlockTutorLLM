"""DeadlockTutorLLM - Streamlit Frontend Interface."""

import os
import requests
import streamlit as st

BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000")

st.set_page_config(
    page_title="DeadlockTutorLLM",
    page_icon="🔒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom header
st.title("🔒 DeadlockTutorLLM")
st.caption("A Domain-Specific AI Tutor for Operating System Deadlocks (RAG + Symbolic Solver + LLM)")

# Sidebar for modes and backend status
with st.sidebar:
    st.header("⚙️ Tutor Settings")
    mode = st.selectbox(
        "Study Mode",
        ["Ask a Question", "Practice Mode", "Hint Mode", "Generate MCQ", "Generate Viva Question"]
    )

    st.divider()
    st.subheader("Backend Status")
    try:
        health_resp = requests.get(f"{BACKEND_URL}/health", timeout=3)
        if health_resp.status_code == 200:
            data = health_resp.json()
            st.success(f"Backend: Connected ({data.get('status')})")
            st.write(f"**Provider**: `{data.get('llm_provider')}`")
            st.write(f"**Model**: `{data.get('primary_model')}`")
        else:
            st.warning(f"Backend returned HTTP {health_resp.status_code}")
    except Exception as e:
        st.error(f"Backend Offline ({BACKEND_URL}). Please start FastAPI server.")

    st.divider()
    st.subheader("Quick Example Prompts")
    if st.button("Theory: Four Conditions"):
        st.session_state["preset_query"] = "What are the four necessary Coffman conditions for deadlock?"
    if st.button("Concept: Unsafe vs Deadlock"):
        st.session_state["preset_query"] = "Does an unsafe state always mean the system is currently deadlocked?"
    if st.button("Numerical: Banker's Algorithm"):
        st.session_state["preset_query"] = "Calculate the Need matrix and determine if this system is safe using Banker's algorithm."
    if st.button("Lab: Dining Philosophers"):
        st.session_state["preset_query"] = "How does circular wait cause deadlock in the dining philosophers problem with pthreads?"

# Initialize chat session history
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {
            "role": "assistant",
            "content": "Hello! I am **DeadlockTutorLLM**, your AI tutor for Operating Systems deadlocks. Ask me about Coffman conditions, Banker's algorithm, resource allocation graphs, or concurrency lab problems!"
        }
    ]

# Display past messages
for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "metadata" in msg and msg["metadata"]:
            meta = msg["metadata"]
            if meta.get("worked_steps"):
                with st.expander("📝 Solver Calculations & Worked Steps"):
                    for step in meta["worked_steps"]:
                        st.write(f"- {step}")
            if meta.get("citations"):
                with st.expander("📚 Syllabus Citations & Sources"):
                    for cit in meta["citations"]:
                        st.markdown(f"**Source**: `{cit.get('source')}` | **Page**: {cit.get('page', 'N/A')}")
            if meta.get("teaching_notes"):
                with st.expander("💡 Teaching Notes & Conceptual Warnings"):
                    for note in meta["teaching_notes"]:
                        st.info(note)

# Input handler
default_input = st.session_state.pop("preset_query", "")
user_prompt = st.chat_input("Ask a deadlock question...", key="chat_input")
prompt_to_send = user_prompt or (default_input if default_input else None)

if prompt_to_send:
    # Append student message
    st.session_state["messages"].append({"role": "user", "content": prompt_to_send})
    with st.chat_message("user"):
        st.markdown(prompt_to_send)

    # Query Backend
    with st.chat_message("assistant"):
        with st.spinner("Tutor is analyzing your question..."):
            try:
                payload = {"message": prompt_to_send}
                resp = requests.post(f"{BACKEND_URL}/api/chat", json=payload, timeout=10)
                if resp.status_code == 200:
                    data = resp.json()
                    explanation = data.get("explanation", "No response content.")
                    st.markdown(explanation)

                    worked_steps = data.get("worked_steps", [])
                    citations = data.get("citations", [])
                    teaching_notes = data.get("teaching_notes", [])

                    if worked_steps:
                        with st.expander("📝 Solver Calculations & Worked Steps"):
                            for step in worked_steps:
                                st.write(f"- {step}")
                    if citations:
                        with st.expander("📚 Syllabus Citations & Sources"):
                            for cit in citations:
                                st.markdown(f"**Source**: `{cit.get('source')}` | **Page**: {cit.get('page', 'N/A')}")
                    if teaching_notes:
                        with st.expander("💡 Teaching Notes & Conceptual Warnings"):
                            for note in teaching_notes:
                                st.info(note)

                    st.session_state["messages"].append({
                        "role": "assistant",
                        "content": explanation,
                        "metadata": {
                            "worked_steps": worked_steps,
                            "citations": citations,
                            "teaching_notes": teaching_notes
                        }
                    })
                else:
                    err_msg = f"Error from backend (HTTP {resp.status_code}): {resp.text}"
                    st.error(err_msg)
                    st.session_state["messages"].append({"role": "assistant", "content": err_msg})
            except Exception as e:
                offline_msg = f"Unable to reach DeadlockTutorLLM backend at `{BACKEND_URL}`. Ensure FastAPI is running."
                st.error(offline_msg)
                st.session_state["messages"].append({"role": "assistant", "content": offline_msg})
