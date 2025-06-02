import logging

# only show INFO+ messages (no DEBUG)
logging.basicConfig(level=logging.INFO)

# if you still want Tenacity’s retry‐INFO but not its DEBUG chatter:
logging.getLogger("tenacity").setLevel(logging.INFO)

# if you’re also seeing fsevents debug spam, silence it:
logging.getLogger("fsevents").setLevel(logging.WARNING)

# -------------------- Tencent VectorDB Setup --------------------
# Import Tencent Cloud VectorDB modules
import tcvectordb
from tcvectordb.model.enum import FieldType, IndexType, MetricType, ReadConsistency
from tcvectordb.model.index import Index, VectorIndex, FilterIndex, HNSWParams
from tcvectordb.model.document import Document, SearchParams, Filter
from tcvectordb.model.collection import Embedding
import time, random, threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from tcvectordb.exceptions import ServerInternalError, GrpcException


# utility function imports
from utils.llm_utils import chunk_text
from utils.processing import extract_projects

# python standard imports
import threading, time, concurrent, tiktoken, concurrent.futures
from tenacity import (
    retry, 
    wait_exponential, 
    stop_after_attempt,
    retry_if_exception_type
)

# Tencent cloud vectordatabse credentials and keys
TENCENT_VECTORDB_URL = "http://gz-vdb-f2w1ru94.sql.tencentcdb.com:8100"
TENCENT_VECTORDB_USERNAME = "root"
tencent_vectordb_client = tcvectordb.RPCVectorDBClient(
    url=TENCENT_VECTORDB_URL,
    username=TENCENT_VECTORDB_USERNAME,
    key='yVdWKRsRXCiL9W61xRkB6iYVJtK38cIxHrlNxe3C',
    read_consistency=ReadConsistency.EVENTUAL_CONSISTENCY,
    timeout=1000
)

# Wrap create_database in a retry decorator
@retry(
    retry=retry_if_exception_type((GrpcException, ServerInternalError)),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    stop=stop_after_attempt(5),
    reraise=True
)

def safe_create_database(name: str):
    return tencent_vectordb_client.create_database(database_name=name)


# Wrap collection creation as well (optional but helpful)
@retry(
    retry=retry_if_exception_type((GrpcException, ServerInternalError)),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    stop=stop_after_attempt(5),
    reraise=True
)

def safe_create_collection(db, **kwargs):
    return db.create_collection(**kwargs)


# -------------------- Embedding & Similarity Retrieval Functions --------------------
def build_vectordb_from_emails(emails):
    # st.info("Building vector index for emails...")
    # Create or reuse a database
    db = safe_create_database('email_db')
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
    
    # create collection with retry
    collection = safe_create_collection(
        db,
        name='email_emb',
        shard=1,
        replicas=1,
        description='Collection for email embeddings',
        embedding=embedding_conf,
        index=index
    )

    docs = []
    for i, email in enumerate(emails):
        # Auto-extract projects from subject + body
        projects = extract_projects(f"{email['subject']}\n{email['body']}")

        combined_text = (
            f"Employee: {email['sender']}\n"
            f"Detected Projects: {', '.join(projects)}\n"
            f"Original Content:\n{email['subject']}\n{email['body']}"
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
    # parallel_batch_upsert(docs, batch_size=5, timeout=1000, delay=3)
    parallel_batch_upsert(docs)
    # st.info("Vector index built successfully.")
    return collection


def retrieve_top_k_emails(user_query, k):
    """
    Finds only the top k most relevant emails (based on semantic similarity to the user’s query)
    and returns them as a list of email dicts (with subject, sender, date, and body).
    These emails can then be passed to the LLM for final summarization or Q&A.
    """
    expanded_query = (
        f"{user_query} "
        "项目 Project 参与 负责 involvement involved"  # Chinese/English boosters
    )

    # st.info("Performing similarity search for top relevant emails...")
    search_results = tencent_vectordb_client.search_by_text(
        database_name='email_db',
        collection_name='email_emb',
        embedding_items=[expanded_query],
        filter=Filter(cond=""),  # Provide an empty condition for no filtering
        params=SearchParams(ef=300),
        limit=k,  # Retrieve only the top k documents
        retrieve_vector=False,
        output_fields=['subject', 'sender', 'date', 'text', 'score']
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


@retry(
    retry=retry_if_exception_type((GrpcException, ServerInternalError)),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    stop=stop_after_attempt(5),
    reraise=True
)
def safe_search_by_text(
    database_name, collection_name, embedding_items, filter,
    params, limit, retrieve_vector, output_fields
):
    return tencent_vectordb_client.search_by_text(
        database_name=database_name,
        collection_name=collection_name,
        embedding_items=embedding_items,
        filter=filter,
        params=params,
        limit=limit,
        retrieve_vector=retrieve_vector,
        output_fields=output_fields
    )


def retrieve_top_k_unique(user_query, k, factor=3):
    """
    1) Pull back k*factor chunks  
    2) Flatten into one list  
    3) Group by email, keep ALL chunks from most relevant emails first  
    4) Return first k chunks total
    """
    # 1) Oversample chunks via the safe wrapper
    raw = safe_search_by_text(
        database_name='email_db',
        collection_name='email_emb',
        embedding_items=[user_query],
        filter=Filter(cond=""),
        params=SearchParams(ef=300),
        limit=k * factor,
        retrieve_vector=False,
        output_fields=['id','subject','sender','date','text','score']
    )

    # 2) Flatten into docs
    if isinstance(raw, dict):
        docs = [doc for grp in raw.get("documents", []) for doc in grp]
    else:
        docs = [doc for grp in raw for doc in grp]

    # 3) Advanced grouping logic
    from collections import defaultdict
    
    # Group docs by parent email ID
    email_groups = defaultdict(list)
    for doc in docs:
        parent_id = doc['id'].split('_chunk_')[0]
        email_groups[parent_id].append(doc)
    
    # Sort chunks within each group by score (best first)
    for parent_id in email_groups:
        email_groups[parent_id].sort(key=lambda x: x['score'], reverse=True)
    
    # Sort groups by their highest chunk score
    sorted_groups = sorted(
        email_groups.values(),
        key=lambda chunks: chunks[0]['score'] if chunks else 0,
        reverse=True
    )
    
    # 4) Collect ALL chunks from groups until we reach k
    final_chunks = []
    for group in sorted_groups:
        # Add all chunks from this email
        final_chunks.extend(group)
        if len(final_chunks) >= k:
            break  # Stop once we have enough
    
    # Convert to desired format and truncate
    return [{
        "subject": chunk["subject"],
        "sender": chunk["sender"],
        "date": chunk["date"],
        "body": chunk["text"]
    } for chunk in final_chunks[:k]]  # Hard cap at k


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


def parallel_batch_upsert(
    docs,
    batch_size=2,
    timeout=1000,
    delay=1,
    max_concurrent=3,
):
    total_batches = (len(docs) + batch_size - 1) // batch_size
    batches = [docs[i:i+batch_size] for i in range(0, len(docs), batch_size)]

    def upsert_batch(batch_index, batch):
        for attempt in range(1, 4):   # up to 3 tries
            try:
                tencent_vectordb_client.upsert(
                    database_name='email_db',
                    collection_name='email_emb',
                    documents=batch,
                    timeout=timeout,
                )
                print(f"✅ Upserted batch {batch_index+1} of {total_batches}", flush=True)
                break
            except Exception as e:
                if "token rate limit reached" in str(e) and attempt < 3:
                    backoff = 2 ** attempt
                    print(f"⚠️ Rate-limit on batch {batch_index+1}, retrying in {backoff}s…", flush=True)
                    time.sleep(backoff)
                else:
                    print(f"❌ Failed batch {batch_index+1} after {attempt} attempts: {e}", flush=True)
                    break
        time.sleep(delay)


    # Limit threads directly here
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_concurrent) as executor:
        futures = [
            executor.submit(upsert_batch, i, batch)
            for i, batch in enumerate(batches)
        ]
        concurrent.futures.wait(futures)