import json

# Paths
input_path = "../data/portfolio_qna.jsonl"
output_path = "../data/train_data.txt"

# Load JSONL
with open(input_path) as f:
    data = [json.loads(line) for line in f]

# Convert to GPT-style format
with open(output_path, "w") as out:
    for entry in data:
        out.write(f"<|user|> {entry['question']}\n")
        out.write(f"<|bot|> {entry['answer']}\n")
        out.write("<|end|>\n")

print(f"Converted {len(data)} Q/A pairs to training format at {output_path}")
