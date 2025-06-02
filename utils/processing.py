# utils/processing.py

import json, os, concurrent, re
from openai import OpenAI
from transformers import AutoTokenizer

# load the same tokenizer your embedding model & DeepSeek use
tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-m3")

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

def analyze_emails_with_LLM(emails, user_prompt):
    # combined_emails_json = json.dumps(all_emails, ensure_ascii=False, indent=2)
    # return parallel_optimize_email_analysis(all_emails, user_prompt), combined_emails_json
    combined_emails_json = json.dumps(emails, ensure_ascii=False, indent=2)
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
        error_message = str(e).lower()
        if "maximum context length" in error_message or "tokens" in error_message:
            # Fallback to parallel approach
            return parallel_optimize_email_analysis(emails, user_prompt), combined_emails_json
        else:
            return f"Error: {str(e)}", combined_emails_json
        

def serialize(messages):
    """Turn a list of {'role':..,'content':..} into one big string for token counting."""
    return "\n".join(f"{m['role'].upper()}: {m['content']}" for m in messages)


def prune_by_token_budget(messages, max_tokens=6000):
    """
    Keep dropping the oldest user+assistant pair until
    the serialized history is <= max_tokens.
    """
    pruned = messages.copy()
    while True:
        text = serialize(pruned)
        length = len(tokenizer(text)["input_ids"])
        # stop if under budget or nothing left to drop
        if length <= max_tokens or len(pruned) < 2:
            break
        # drop the oldest two entries (one user and one assistant)
        pruned = pruned[2:]
    return pruned


def extract_projects(text):
    """
    Identifies project names using:
    - Patterns like "XXX项目" (Chinese project suffix)
    - Title case phrases in English/Chinese
    - Keywords preceding "项目" or "Project"
    """
    patterns = [
        r'(项目|Project):?\s*([^\n，。]+)',  # "项目：宁波搭把手"
        r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+Project\b',  # "Helping Hands Project"
        r'(\S+项目)\b'  # "宁波项目"
    ]
    
    projects = set()
    for pattern in patterns:
        matches = re.findall(pattern, text)
        for match in matches:
            if isinstance(match, tuple):
                projects.add(match[1].strip())
            else:
                projects.add(match.strip())
    return list(projects)
