import tiktoken
from pypdf import PdfReader

def is_pdf(filepath):
    try:
        with open(filepath, 'rb') as file:
            PdfReader(file)
        return True
    except Exception:
        return False

def count_tokens(filepath = None, text = None, encoding_name="cl100k_base"):
    if filepath:
        if is_pdf(filepath):
            reader = PdfReader(filepath)
            pages = reader.pages

            num_tokens = 0
            for page in pages:
                curr_text = page.extract_text()
                encoding = tiktoken.get_encoding(encoding_name)
                tokens = encoding.encode(curr_text)
                num_tokens += len(tokens)
            
            return num_tokens
        
        with open(filepath, 'r', encoding="utf-8") as file:
            text = file.read()

    encoding = tiktoken.get_encoding(encoding_name)
    tokens = encoding.encode(text)

    return len(tokens)