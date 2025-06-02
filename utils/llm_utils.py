# utils/llm_utils.py


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