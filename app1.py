import os
import gc
import uuid
import shutil
import json
import time
import base64
import poplib
import logging
import tempfile
import concurrent.futures
from datetime import datetime
from email import policy
from email.parser import BytesParser
from email.utils import parsedate_to_datetime
from email.policy import default

import streamlit as st
from openai import OpenAI
import openai
import time

import threading
import re

# -------------------- VolcEngine API Setup --------------------
os.environ["ARK_API_KEY"] = "5ade76c2-9629-4076-aebd-3550719382e6"
os.environ["DS_API_KEY"] = "sk-739005b0963e4adaa37304aff79f636c"
os.environ["TC_API_KEY"] = "sk-KWoOkQiq7lhLCsd75sE8RNUCMQQMQBFckYweTmklgWSuwY56"

base_url_ark = "https://ark.cn-beijing.volces.com/api/v3"
base_url_ds = "https://api.deepseek.com"
base_url_tc = "https://api.lkeap.cloud.tencent.com/v1"

client = OpenAI(
    api_key=os.environ.get("TC_API_KEY"),
    base_url=base_url_tc,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# -------------------- Tencent VectorDB Setup --------------------
# Import Tencent Cloud VectorDB modules
import tcvectordb
from tcvectordb.model.enum import FieldType, IndexType, MetricType, ReadConsistency
from tcvectordb.model.index import Index, VectorIndex, FilterIndex, HNSWParams
from tcvectordb.model.document import Document, SearchParams, Filter
from tcvectordb.model.collection import Embedding

# Replace with your actual Tencent Cloud VectorDB external URL
TENCENT_VECTORDB_URL = "http://gz-vdb-dbtf4m82.sql.tencentcdb.com:8100"
TENCENT_VECTORDB_USERNAME = "root"
tencent_vectordb_client = tcvectordb.RPCVectorDBClient(
    url=TENCENT_VECTORDB_URL,
    username=TENCENT_VECTORDB_USERNAME,
    key='Cp6MNT6erkUntajMfDKQk6DnHNhf40PiNVeKgobr',
    read_consistency=ReadConsistency.EVENTUAL_CONSISTENCY,
    timeout=1000
)

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

# -------------------- Email Analysis Functions --------------------
def analyze_group(group_label, emails_in_group, summarization_prompt):
    group_emails_json = json.dumps(emails_in_group, ensure_ascii=False, indent=2)
    try:
        completion = client.chat.completions.create(
            model="deepseek-v3-0324",
            messages=[
                {"role": "system", "content": summarization_prompt},
                {"role": "user", "content": group_emails_json},
            ],
            stream=False,
            temperature=0,
            top_p=1.0,
            seed=42,
            presence_penalty=0,
            frequency_penalty=0
        )
        return group_label, completion.choices[0].message.content
    except Exception as e:
        return group_label, f"Error analyzing group: {str(e)}"
    
def finalize_analysis(combined_group_summary, user_prompt):
    final_prompt = (
        f"Combine the following summarized email groups into a final comprehensive analysis that "
        f"answers the user query. Ensure key information is maintained.\n\n"
        f"Summarized Email Groups:\n{combined_group_summary}\n\n"
        f"User Query: {user_prompt}"
    )
    try:
        completion = client.chat.completions.create(
            model="deepseek-v3-0324",
            messages=[
                {"role": "system", "content": "You are an expert summarizer."},
                {"role": "user", "content": final_prompt},
            ],
            stream=False,
            temperature=0,
            top_p=1.0,
            seed=42,
            presence_penalty=0,
            frequency_penalty=0
        )
        return completion.choices[0].message.content
    except Exception as e:
        return f"Error in final analysis: {str(e)}"

def group_emails_by_subject(all_emails, similarity_threshold=0.7):
    """
    Groups emails by subject similarity using a token-overlap approach on the email title only.
    Returns a dict of {group_label: [emails]}.
    """
    import re
    groups = {}

    def normalize_subject(subject):
        return re.sub(r'[^a-zA-Z0-9 ]', '', subject.lower()).split()

    for email_obj in all_emails:
        subject = email_obj["subject"] or "No Subject"
        subject_tokens = set(normalize_subject(subject))
        best_match_group = None
        best_match_score = 0

        for group_label, emails_in_group in groups.items():
            group_tokens = set(normalize_subject(group_label))
            intersection = subject_tokens.intersection(group_tokens)
            union = subject_tokens.union(group_tokens)
            similarity = len(intersection) / (len(union) or 1)
            if similarity > best_match_score:
                best_match_score = similarity
                best_match_group = group_label

        if best_match_group and best_match_score >= similarity_threshold:
            groups[best_match_group].append(email_obj)
        else:
            groups[subject[:50]] = [email_obj]

    return groups

def parallel_optimize_email_analysis(all_emails, user_prompt):
    """
    Multi-layer processing:
      1. Group emails by subject similarity.
      2. Send each group for first-layer summarization in parallel.
      3. Combine all group summaries.
      4. Run a final analysis on the combined summary.
    """
    grouped_emails = group_emails_by_subject(all_emails)
    first_layer_prompt = "Summarize these emails into a concise summary while maintaining key information."

    group_results = {}
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_to_group = {
            executor.submit(analyze_group, group_label, emails, first_layer_prompt): group_label
            for group_label, emails in grouped_emails.items()
        }
        for future in concurrent.futures.as_completed(future_to_group):
            group_label = future_to_group[future]
            try:
                label, result = future.result()
                group_results[label] = result
            except Exception as e:
                group_results[group_label] = f"Error analyzing group: {str(e)}"

    combined_output = "\n".join([f"Group: {label}\nSummary:\n{result}" for label, result in group_results.items()])
    final_result = finalize_analysis(combined_output, user_prompt)
    return final_result

def analyze_emails_with_LLM(all_emails, user_prompt):
    # combined_emails_json = json.dumps(all_emails, ensure_ascii=False, indent=2)
    # return parallel_optimize_email_analysis(all_emails, user_prompt), combined_emails_json
    combined_emails_json = json.dumps(all_emails, ensure_ascii=False, indent=2)
    try:
        completion = client.chat.completions.create(
            model="deepseek-v3-0324",
            messages=[
                {"role": "system", "content": user_prompt},
                {"role": "user", "content": combined_emails_json},
            ],
            stream=False,
            temperature=0,
            top_p=1.0,
            seed=42,
            presence_penalty=0,
            frequency_penalty=0
        )
        return completion.choices[0].message.content, combined_emails_json
    except Exception as e:
        # return f"Error: {str(e)}", combined_emails_json
        error_message = str(e).lower()
        if "maximum context length" in error_message or "tokens" in error_message:
            # Fallback to parallel approach
            return parallel_optimize_email_analysis(all_emails, user_prompt), combined_emails_json
        else:
            return f"Error: {str(e)}", combined_emails_json

# -------------------- Embedding & Similarity Retrieval Functions --------------------
def build_vectordb_from_emails(emails):
    # st.info("Building vector index for emails...")
    # Create or reuse a database
    db = tencent_vectordb_client.create_database(database_name='email_db')
    # Define index fields and embedding configuration
    index = Index(
        FilterIndex(name='id', field_type=FieldType.String, index_type=IndexType.PRIMARY_KEY),
        VectorIndex(
            name='vector',
            dimension=1024,  # Adjust based on your model's dimension
            index_type=IndexType.HNSW,
            metric_type=MetricType.COSINE,
            params=HNSWParams(m=16, efconstruction=200)
        ),
        FilterIndex(name='subject', field_type=FieldType.String, index_type=IndexType.FILTER)
    )
    embedding_conf = Embedding(vector_field='vector', field='text', model_name='BAAI/bge-m3')
    collection = db.create_collection(
        name='email_emb',
        shard=1,
        replicas=1,
        description='Collection for email embeddings',
        embedding=embedding_conf,
        index=index
    )
    docs = []
    for i, email in enumerate(emails):
        combined_text = (
            f"Subject: {email['subject']}\n"
            f"From: {email['sender']}\n"
            f"Date: {email['date']}\n"
            f"Body:\n{email['body']}"
        )
        # Use chunk_text to split if the combined text is too long
        if len(combined_text) > 8192:
            chunks = chunk_text(combined_text, max_length=8192, overlap=500)
            # st.info(f"Email {i} split into {len(chunks)} chunks.")
        else:
            chunks = [combined_text]
        for j, chunk in enumerate(chunks):
            doc = Document(
                id=f"email_{i}_chunk_{j}",
                text=chunk,
                subject=email['subject'],
                sender=email['sender'],
                date=email['date']
            )
            docs.append(doc)
    # st.info(f"Total documents (chunks) to upsert: {len(docs)}")
    # st.info("Upserting documents in batches...")
    # st.info(f"Upserting {len(docs)} email chunks into vector DB in batches...")
    parallel_batch_upsert(docs, batch_size=5, timeout=1000, delay=3)
    # st.info("Vector index built successfully.")
    return collection


def retrieve_top_k_emails(user_query, k):
    """
    Finds only the top k most relevant emails (based on semantic similarity to the user’s query)
    and returns them as a list of email dicts (with subject, sender, date, and body).
    These emails can then be passed to the LLM for final summarization or Q&A.
    """
    # st.info("Performing similarity search for top relevant emails...")
    search_results = tencent_vectordb_client.search_by_text(
        database_name='email_db',
        collection_name='email_emb',
        embedding_items=[user_query],
        filter=Filter(cond=""),  # Provide an empty condition for no filtering
        params=SearchParams(ef=200),
        limit=k,  # Retrieve only the top k documents
        retrieve_vector=False,
        output_fields=['subject', 'sender', 'date', 'text']
    )
    # Depending on the response format, documents may be in a dict or list
    if isinstance(search_results, dict):
        documents = search_results.get("documents", [])
    else:
        documents = search_results

    relevant_emails = []
    for group in documents:
        for doc in group:
            relevant_emails.append({
                "subject": doc.get("subject", ""),
                "sender": doc.get("sender", ""),
                "date": doc.get("date", ""),
                "body": doc.get("text", "")
            })
    # st.info(f"Found {len(relevant_emails)} relevant emails.")
    return relevant_emails


def retrieve_top_k_unique(user_query, k, factor=3):
    """
    1) Pull back k*factor chunks  
    2) Flatten into one list  
    3) Keep only the first chunk per parent email ID  
    4) Return up to k unique emails
    """
    # 1) oversample chunks
    raw = tencent_vectordb_client.search_by_text(
        database_name='email_db',
        collection_name='email_emb',
        embedding_items=[user_query],
        filter=Filter(cond=""),
        params=SearchParams(ef=200),
        limit=k * factor,            # e.g. 20*3=60
        retrieve_vector=False,
        output_fields=['id','subject','sender','date','text']
    )

    # 2) flatten into docs
    if isinstance(raw, dict):
        groups = raw.get("documents", [])
        docs = [doc for grp in groups for doc in grp]
    else:
        docs = [doc for grp in raw for doc in grp]

    # 3) dedupe by parent ID
    seen, unique = set(), []
    for doc in docs:
        parent = doc['id'].split('_chunk_')[0]
        if parent not in seen:
            seen.add(parent)
            unique.append({
                "subject": doc["subject"],
                "sender":  doc["sender"],
                "date":    doc["date"],
                "body":    doc["text"]
            })
        if len(unique) >= k:
            break

    return unique


# -------------------- POP3 Email Download Functions --------------------
def connect_to_pop3(email_host, email_port, email_user, email_pass):
    try:
        mail = poplib.POP3_SSL(email_host, email_port)
        mail.user(email_user)
        mail.pass_(email_pass)
        logging.info(f"Connected to {email_host}, Total emails: {len(mail.list()[1])}")
        return mail
    except Exception as e:
        logging.error("Failed to connect to POP3 server: " + str(e))
        return None


def get_message_date(mail, msg_num):
    try:
        # Use TOP command to retrieve headers only
        resp, lines, octets = mail.top(msg_num, 0)  # 0 lines of body
        raw_email_bytes = b"\n".join(lines)
        msg = BytesParser(policy=default).parsebytes(raw_email_bytes)
        email_date_str = msg["date"]
        email_date_parsed = parsedate_to_datetime(email_date_str)
        if email_date_parsed.tzinfo:
            email_date = email_date_parsed.astimezone().replace(tzinfo=None)
        else:
            email_date = email_date_parsed
        return email_date
    except Exception as e:
        logging.error(f"Error retrieving headers for message {msg_num}: {e}")
        return None


def find_low(mail, start_date, total_messages):
    left = 1
    right = total_messages
    low = None
    while left <= right:
        mid = (left + right) // 2
        email_date = get_message_date(mail, mid)
        if not email_date:
            right = mid - 1  # Adjust and continue
            continue
        if email_date >= start_date:
            low = mid
            right = mid - 1
        else:
            left = mid + 1
    return low


def find_high(mail, end_date, total_messages):
    left = 1
    right = total_messages
    high = None
    while left <= right:
        mid = (left + right) // 2
        email_date = get_message_date(mail, mid)
        if not email_date:
            left = mid + 1  # Adjust and continue
            continue
        if email_date <= end_date:
            high = mid
            left = mid + 1
        else:
            right = mid - 1
    return high
   

def fetch_and_save_emails(mail, start_date, end_date, output_dir="emails", num_messages=None):
    if not mail:
        return []
    os.makedirs(output_dir, exist_ok=True)
    total_messages = len(mail.list()[1])
    email_file_info = []
    
    # Find the low and high bounds using binary search
    low = find_low(mail, start_date, total_messages)
    high = find_high(mail, end_date, total_messages)
    
    if low is None or high is None or low > high:
        logging.info("No messages found within the specified date range.")
        return email_file_info
    
    scanned = 0
    # Iterate from high to low (newest to oldest within the range)
    for i in range(high, low - 1, -1):
        if num_messages and scanned >= num_messages:
            break
        try:
            resp, lines, octets = mail.retr(i)
            raw_email_bytes = b"\n".join(lines)
            msg = BytesParser(policy=default).parsebytes(raw_email_bytes)
            email_date_str = msg["date"]
            try:
                email_date_parsed = parsedate_to_datetime(email_date_str)
                if email_date_parsed.tzinfo:
                    email_date = email_date_parsed.astimezone().replace(tzinfo=None)
                else:
                    email_date = email_date_parsed
            except Exception as e:
                logging.error(f"Skipping email {i} due to date parsing error: {email_date_str}")
                continue
            # Double-check date is within range (in case of out-of-order dates)
            if not (start_date <= email_date <= end_date):
                logging.info(f"Skipping email {i} outside date range: {email_date}")
                continue
            subject = msg["subject"] or "No Subject"
            logging.info(f"Saving email {i}: {subject} on {email_date_str}")
            eml_filename = os.path.join(output_dir, f"email_{i}.eml")
            with open(eml_filename, "wb") as f:
                f.write(raw_email_bytes)
            email_file_info.append((eml_filename, email_date.isoformat()))
            scanned += 1
        except Exception as e:
            logging.error(f"Error processing email {i}: {e}")
    return email_file_info


def parallel_extract_emails_with_progress(email_file_info):
    results = []
    total_files = len(email_file_info)
    progress_bar = st.progress(0)
    completed = 0
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = { executor.submit(extract_email_content, item[0]): item for item in email_file_info }
        for future in concurrent.futures.as_completed(futures):
            eml_filename, iso_date = futures[future]
            try:
                result = future.result()
                result["date"] = iso_date
                results.append(result)
            except Exception as e:
                st.error(f"Error processing an email: {e}")
            completed += 1
            progress_bar.progress(completed / total_files)
    return results

def extract_email_content(eml_file):
    try:
        with open(eml_file, "rb") as f:
            msg = BytesParser(policy=policy.default).parse(f)
        subject = msg["subject"] if msg["subject"] else "No Subject"
        sender = msg["from"] if msg["from"] else "Unknown Sender"
        body = None
        if msg.is_multipart():
            for part in msg.walk():
                content_type = part.get_content_type()
                content_disposition = str(part.get("Content-Disposition"))
                if content_type == "text/plain" and "attachment" not in content_disposition:
                    charset = part.get_content_charset() or "utf-8"
                    body = part.get_payload(decode=True).decode(charset, errors="ignore")
                    break
        if body is None:
            if msg.get_body(preferencelist=("plain",)):
                body = msg.get_body(preferencelist=("plain",)).get_content()
            elif msg.get_body(preferencelist=("html",)):
                body = "HTML Email: " + msg.get_body(preferencelist=("html",)).get_content()
            else:
                body = "No readable content available."
        logging.info(f"Extracted Email: {subject} from {sender}")
        return {"subject": subject, "sender": sender, "body": body}
    except Exception as e:
        logging.error(f"Error processing {eml_file}: {e}")
        return {"subject": "Error", "sender": "Error", "body": "Could not process email."}

def cache_analysis_result(all_emails, user_prompt):
    import hashlib
    input_key = f"{json.dumps(all_emails, sort_keys=True)}{user_prompt}"
    cache_filename = f"cache{hashlib.md5(input_key.encode()).hexdigest()}.json"
    if os.path.exists(cache_filename):
        with open(cache_filename, "r", encoding="utf-8") as f:
            return json.load(f)
    return None

def store_analysis_result(all_emails, user_prompt, result):
    import hashlib
    input_key = f"{json.dumps(all_emails, sort_keys=True)}{user_prompt}"
    cache_filename = f"cache{hashlib.md5(input_key.encode()).hexdigest()}.json"
    with open(cache_filename, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

def chunk_text(text, max_length=8192, overlap=500):
    """
    Splits a long text into chunks no longer than max_length characters.
    Overlaps consecutive chunks by 'overlap' characters.
    """
    chunks = []
    start = 0
    text_length = len(text)
    while start < text_length:
        end = min(start + max_length, text_length)
        chunks.append(text[start:end])
        # Ensure we overlap to preserve context unless we're at the end
        start = end - overlap if end < text_length else text_length
    return chunks

def batch_upsert(docs, batch_size=10, timeout=1000, delay=1):
    """
    Upserts documents in smaller batches with a delay between batches.
    This helps avoid exceeding the token rate limit.
    """
    total_batches = ((len(docs)-1) // batch_size) + 1
    for i in range(0, len(docs), batch_size):
        batch = docs[i:i+batch_size]
        tencent_vectordb_client.upsert(
            database_name='email_db',
            collection_name='email_emb',
            documents=batch,
            timeout=timeout
        )
        print(f"Upserted batch {i // batch_size + 1} of {total_batches}")
        time.sleep(delay)


def parallel_batch_upsert(docs, batch_size=5, timeout=1000, delay=3, max_concurrent=2):
    total_batches = ((len(docs) - 1) // batch_size) + 1
    batches = [docs[i:i+batch_size] for i in range(0, len(docs), batch_size)]
    # Limit the number of concurrent upsert requests
    semaphore = threading.Semaphore(max_concurrent)

    def upsert_batch(batch_index, batch):
        with semaphore:
            tencent_vectordb_client.upsert(
                database_name='email_db',
                collection_name='email_emb',
                documents=batch,
                timeout=timeout
            )
            print(f"Upserted batch {batch_index + 1} of {total_batches}")
            time.sleep(delay)  # Throttle to help avoid rate limits

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(upsert_batch, i, batch) for i, batch in enumerate(batches)]
        concurrent.futures.wait(futures)


# -------------------- Streamlit UI --------------------
st.set_page_config(page_title="馃摡 Streamax Email Analyzer & Downloader", layout="wide")

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
    override_cache = st.checkbox("Override cached responses")
    if st.button("Fetch Emails"):
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
                email_file_info = fetch_and_save_emails(mail, start_dt, end_dt, output_dir="emails", num_messages=0)
                mail.quit()
                download_end_time = time.time()
                download_time = download_end_time - download_start_time
                if email_file_info:
                    st.success(f"Fetched and saved {len(email_file_info)} emails in {download_time:.2f} seconds.")
                    st.info("Extracting email contents in parallel...")
                    st.session_state.all_emails = parallel_extract_emails_with_progress(email_file_info)
                    st.success("Emails ready for analysis! Start chatting below.")
                else:
                    st.error("No emails found in the specified timeframe.")
            else:
                st.error("Failed to connect to POP3 server.")
    if st.session_state.all_emails:
        st.subheader("List of Emails")
        for i, email_data in enumerate(st.session_state.all_emails, start=1):
            sender = email_data.get("sender", "Unknown Sender")
            subject = email_data.get("subject", "No Subject")
            date_str = email_data.get("date", "")
            st.write(f"{i}. **From**: {sender} | **Subject**: {subject} | **Date**: {date_str}")

col1, col2 = st.columns([6, 1])
with col1:
    st.markdown(f"""
    # Email Analyzer powered by 
    <img src="data:image/png;base64,{base64.b64encode(open('assets/deepseek_logo.png','rb').read()).decode()}" 
         width="200" style="vertical-align: -15px; padding-right: 10px;"> 
    """, unsafe_allow_html=True)
with col2:
    st.button("Clear", on_click=reset_chat)

for message in st.session_state.conversation_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

user_input = st.chat_input("Ask about the emails...")

if user_input:
    st.session_state.conversation_history.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)
    if st.session_state.all_emails is None:
        ai_reply = "Please fetch emails first using the sidebar."
    else:
        # Build vector index if not already done
        if "vector_collection" not in st.session_state:
            # st.info("Indexing emails for similarity search...")
            st.session_state.vector_collection = build_vectordb_from_emails(st.session_state.all_emails)
        # Use Tencent VectorDB to retrieve relevant emails based on user query
        relevant_emails = retrieve_top_k_unique(user_input, k=35, factor=4)
        if not relevant_emails:
            st.info("No relevant emails found; using all emails for analysis.")
            emails_for_analysis = st.session_state.all_emails
        else:
            # st.info(f"Using {len(relevant_emails)} relevant emails for analysis.")
            emails_for_analysis = relevant_emails
        conversation_context = "\n".join([
            f"User: {st.session_state.conversation_history[i]['content']}\nAI: {st.session_state.conversation_history[i+1]['content']}"
            for i in range(0, len(st.session_state.conversation_history) - 1, 2)
        ])
        full_prompt = f"Context:\n{conversation_context}\n\nNew User Query: {user_input}"
        cached_result = None if override_cache else cache_analysis_result(emails_for_analysis, user_input)
        if cached_result:
            ai_reply = cached_result
        else:
            analysis_start = time.time()
            ai_reply, json_file = analyze_emails_with_LLM(emails_for_analysis, full_prompt)
            output_path = "combined_emails.json"
            parsed_json = json.loads(json_file)
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(parsed_json, f, ensure_ascii=False, indent=2)
            analysis_end = time.time()
            store_analysis_result(emails_for_analysis, user_input, ai_reply)
            st.caption(f"Analysis took {analysis_end - analysis_start:.2f} seconds")

    with st.chat_message("assistant"):
        message_placeholder = st.empty()  # Create a placeholder for the streamed response
        full_response = ""

        # This split captures both words and the whitespace (including newlines) in between
        tokens = re.split(r'(\s+)', ai_reply)

        for i, token in enumerate(tokens):
            # Add each token (which could be a word, space, or newline)
            full_response += token

            # Because Streamlit Markdown doesn't always treat single newlines
            # as breaks, you can force them by replacing "\n" with "  \n"
            # or double-newline for paragraphs, depending on your preference:
            rendered_text = full_response.replace("\n", "  \n")

            # Update the placeholder
            message_placeholder.markdown(rendered_text + "▌")

            # Add a small delay (optional) to simulate streaming
            if i < len(tokens) - 1:
                time.sleep(0.1)
        
        # Display final response without cursor
        message_placeholder.markdown(full_response)
        
    st.session_state.conversation_history.append({"role": "assistant", "content": ai_reply})
