# app.py

# util functions imports
import streamlit as st
from utils.email_fetch import connect_to_pop3, fetch_and_save_emails, parallel_extract_emails_with_progress
from utils.vectordb import build_vectordb_from_emails, retrieve_top_k_unique
from utils.processing import analyze_emails_with_LLM, prune_by_token_budget
from utils.cache import cache_analysis_result, store_analysis_result

# Built-in packages 
import base64
import time
import os
import shutil
import gc
from datetime import datetime
import logging
import re
import poplib

# tencent cloud vector database packages
from tcvectordb.exceptions import ServerInternalError

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Maximum chatbot conversation exchanes before pruning
MAX_TURNS = 5

# -------------------- Conversation / State Management --------------------
def reset_chat():
    """Reset the conversation and any fetched emails."""
    st.session_state.conversation_history = []
    st.session_state.all_emails = None
    # Also reset vector index if exists
    if "vector_collection" in st.session_state:
        del st.session_state["vector_collection"]
    gc.collect()

if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []

if "all_emails" not in st.session_state:
    st.session_state.all_emails = None


# -------------------- Streamlit UI --------------------
if __name__ == "__main__":
    st.set_page_config(page_title="Streamax Email Analyzer & Downloader", layout="wide")

    with st.sidebar:
        st.header("Fetch Emails from your Mailbox Account")
        email_host = st.text_input("POP3 Server Host", value="mail.streamax.com")
        email_port = st.number_input("POP3 Server Port", value=995, step=1)
        email_user = st.text_input("Email Address")
        email_pass = st.text_input("Email Password", type="password")
        col_s1, col_s2 = st.columns(2)
        with col_s1:
            start_date = st.date_input("Start Date", value=datetime(2025, 2, 19).date())
        with col_s2:
            end_date = st.date_input("End Date", value=datetime(2025, 2, 26).date())
        # override_cache = st.checkbox("Override cached responses")
        if st.button("Fetch and Upsert Emails"):
            if not email_user or not email_pass:
                st.error("Please provide your email address and password.")
            else:
                if os.path.exists("emails"):
                    shutil.rmtree("emails")
                os.makedirs("emails", exist_ok=True)
                st.info("Connecting to POP3 server...")
                mail = connect_to_pop3(email_host, int(email_port), email_user, email_pass)
                if mail:
                    st.info("Connected. Fetching emails...")
                    start_dt = datetime.combine(start_date, datetime.min.time())
                    end_dt = datetime.combine(end_date, datetime.max.time())
                    download_start_time = time.time()
                    email_file_info = fetch_and_save_emails(
                        mail, 
                        "mail.streamax.com",
                        995,
                        email_user,
                        email_pass,
                        start_date,
                        end_date,
                        output_dir="emails",
                        num_messages=None,
                        max_workers=5
                    )
                    # try:
                    #     mail.quit()
                    # except poplib.error_proto:
                    #     pass # server already dropped us
                    download_end_time = time.time()
                    download_time = download_end_time - download_start_time
                    if email_file_info:
                        st.success(f"Fetched and saved {len(email_file_info)} emails in {download_time:.2f} seconds.")
                        st.session_state.all_emails = parallel_extract_emails_with_progress(email_file_info)
                        # Build the vector DB immediately
                        with st.spinner("Indexing emails into VectorDB..."):
                            st.session_state.vector_collection = build_vectordb_from_emails(st.session_state.all_emails)
                        st.success("VectorDB built! You can now ask questions.")

                    else:
                        st.error("No emails found in the specified timeframe.")
                else:
                    st.error("Failed to connect to POP3 server.")
        if st.session_state.all_emails:
            st.subheader("List of Emails")
            for i, e in enumerate(st.session_state.all_emails, 1):
                st.write(f"{i}. From: {e['sender']} | Subject: {e['subject']} | Date: {e['date']}")

    col1, col2 = st.columns([6, 1])
    with col1:
        st.markdown(f"""
        # Email Analyzer powered by 
        <img src="data:image/png;base64,{base64.b64encode(open('assets/deepseek_logo.png','rb').read()).decode()}" 
            width="200" style="vertical-align: -15px; padding-right: 10px;"> 
        """, unsafe_allow_html=True)
    with col2:
        st.button("Clear", on_click=reset_chat)

    # (2) Render the existing conversation history
    for msg in st.session_state.conversation_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # (3) **This must be the very last Streamlit call** in your script:
    user_input = st.chat_input("Ask about the emails...")

    if user_input:
        # echo user
        with st.chat_message("user"):
            st.markdown(user_input)
        st.session_state.conversation_history.append(
            {"role": "user", "content": user_input}
        )

        # Everything else remain the same
        if st.session_state.all_emails is None:
            ai_reply = "Please fetch emails first using the sidebar."
        else:
            # Build vector index if not already done
            if "vector_collection" not in st.session_state:
                # st.info("Indexing emails for similarity searc h...")
                st.session_state.vector_collection = build_vectordb_from_emails(st.session_state.all_emails)

            # Sometimes the vectordatabase might be deleted, create a try except statement to rebuild deleted vector db
            try:
                # Use Tencent VectorDB to retrieve relevant emails based on user query
                relevant_emails = retrieve_top_k_unique(user_input, 
                                                        k=12, 
                                                        factor=3)
                
                emails_for_analysis = relevant_emails
            except ServerInternalError as e:
                # Check the "db does not exist" code
                if e.code == 15301 or e.code == 15201:
                    st.info("VectorDB missing - rebuilding index...")
                    # re-build the database
                    st.session_state.vector_collection = build_vectordb_from_emails(st.session_state.all_emails)
                    # Retry retrieving relevant emails
                    relevant_emails = retrieve_top_k_unique(user_input, k=35, factor=4)
                    emails_for_analysis = relevant_emails
                else:
                    raise

            conversation_context = "\n".join([
                f"User: {st.session_state.conversation_history[i]['content']}\nAI: {st.session_state.conversation_history[i+1]['content']}"
                for i in range(0, len(st.session_state.conversation_history) - 1, 2)
            ])

            full_prompt = f"Context:\n{conversation_context}\n\nNew User Query: {user_input}"

            analysis_start = time.time()
            ai_reply, _ = analyze_emails_with_LLM(emails_for_analysis, full_prompt)
            analysis_end = time.time()
            # store_analysis_result(emails_for_analysis, user_input, ai_reply)
            st.caption(f"Analysis took {analysis_end - analysis_start:.2f} seconds")

        # reserve assistant container and placeholder
        # echo assistant (with optional streaming)
        with st.chat_message("assistant"):
            placeholder = st.empty()
            full = ""
            for tok in re.split(r'(\s+)', ai_reply):
                full += tok
                placeholder.markdown(full + "▌")
                time.sleep(0.03)
            placeholder.markdown(full)

        st.session_state.conversation_history.append(
            {"role": "assistant", "content": ai_reply}
        )

        # Prune by token budget instead of fixed turns
        hist = st.session_state.conversation_history
        pruned = prune_by_token_budget(hist, max_tokens=2000)
        if len(pruned) < len(hist):
            # st.info("Pruning conversation history")
            # mutate the same list object instead of rebinding
            hist[:] = pruned
