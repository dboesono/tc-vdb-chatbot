# utils/email_fetch.py

import poplib, logging, os, shutil
from datetime import datetime
from email.parser import BytesParser, HeaderParser
from email.policy import default
import concurrent.futures
import streamlit as st
from concurrent.futures import ThreadPoolExecutor, as_completed

from email import policy
from email.parser import BytesParser
from email.utils import parsedate_to_datetime
import email

from datetime import datetime, date, time


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
   

def fetch_and_save_emails(
    mail,
    host: str,
    port: int,
    username: str,
    password: str,
    start_date,
    end_date,
    output_dir="emails",
    num_messages=None,
    max_workers=14,
):
    # --- normalize date types to datetime.datetime ---
    if isinstance(start_date, date) and not isinstance(start_date, datetime):
        start_date = datetime.combine(start_date, time.min)
    if isinstance(end_date, date) and not isinstance(end_date, datetime):
        end_date = datetime.combine(end_date, time.max)
    # --------------------------------------------------

    if not mail:
        return []

    os.makedirs(output_dir, exist_ok=True)
    total_messages = len(mail.list()[1])

    low = find_low(mail, start_date, total_messages)
    high = find_high(mail, end_date, total_messages)

    try:
        mail.quit()
    except (poplib.error_proto, AttributeError) as e:
        logging.debug(f"POP3 session already gone: {e}")

    if low is None or high is None or low > high:
        logging.info("No messages found within the specified date range.")
        return []

    # Build the list of message‐numbers we want to fetch
    msg_nums = list(range(high, low - 1, -1))
    if num_messages:
        msg_nums = msg_nums[:num_messages]

    def _worker(msg_num):
        try:
            worker_mail = connect_to_pop3(host, port, username, password)
            if not worker_mail:
                return None

            # Fetch headers only
            hdr_lines = worker_mail.top(msg_num, 0)[1]
            hdr = HeaderParser().parsestr(
                b"\r\n".join(hdr_lines).decode("utf8", "ignore")
            )
            dt = parsedate_to_datetime(hdr["Date"])
            email_date = dt.astimezone().replace(tzinfo=None) if dt.tzinfo else dt

            # Skip out-of-range
            if not (start_date <= email_date <= end_date):
                worker_mail.quit()
                return None

            # Now fetch full message
            resp, lines, octets = worker_mail.retr(msg_num)
            raw = b"\r\n".join(lines)
            worker_mail.quit()

            msg = BytesParser(policy=default).parsebytes(raw)
            subject = msg["subject"] or "No Subject"
            logging.info(f"[{msg_num}] {subject} @ {hdr['Date']}")

            fname = os.path.join(output_dir, f"email_{msg_num}.eml")
            with open(fname, "wb") as f:
                f.write(raw)

            return (fname, email_date.isoformat())

        except Exception as e:
            logging.error(f"Error fetching msg {msg_num}: {e}")
            return None

    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as exe:
        futures = {exe.submit(_worker, n): n for n in msg_nums}
        for fut in as_completed(futures):
            res = fut.result()
            if res:
                results.append(res)

    return results


def parallel_extract_emails_with_progress(email_file_info):
    results = []
    total_files = len(email_file_info)
    # progress_bar = st.progress(0)
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
            # progress_bar.progress(completed / total_files)
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


def parse_eml_to_dict(filepath: str, msg: email.message.Message) -> dict:
    """
    Turn a raw .eml file into the {filename, sender, subject, date, body} dict
    your pipeline uses.
    """
    # 1) Filename
    filename = os.path.basename(filepath)

    # 2) Headers
    subject = msg.get("Subject", "")
    sender  = msg.get("From", "")
    date_hdr = msg.get("Date", "")
    # parse Date into ISO format (fallback to raw string if parsing fails)
    try:
        dt = email.utils.parsedate_to_datetime(date_hdr)
        # strip tzinfo for consistency
        date = dt.astimezone().replace(tzinfo=None).isoformat()
    except Exception:
        date = date_hdr

    # 3) Body: grab the first text/plain part, or fallback to entire payload
    body = ""
    if msg.is_multipart():
        for part in msg.walk():
            ctype = part.get_content_type()
            disp  = part.get("Content-Disposition", "")
            if ctype == "text/plain" and "attachment" not in disp:
                body = part.get_content().strip()
                break
    else:
        body = msg.get_content().strip()

    return {
        "filename": filename,
        "sender":   sender,
        "subject":  subject,
        "date":     date,
        "body":     body,
    }


    