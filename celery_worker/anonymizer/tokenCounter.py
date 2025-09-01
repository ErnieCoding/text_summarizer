from transformers import AutoTokenizer

TOKENIZER = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-32B")
def count_tokens(text):
    tokens = TOKENIZER.encode(text, add_special_tokens=False)

    return len(tokens)

if __name__ == "__main__":
    print(count_tokens(text=""""""))