# utils/cache.py

import json, hashlib, os

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
