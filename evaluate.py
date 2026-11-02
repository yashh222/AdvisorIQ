# evaluate.py  (research part)
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy
from datasets import Dataset

# Build this from a handful of test questions you answer with the app:
samples = {
    "question": ["Is Apple a good buy right now?"],
    "answer":   ["<the app's answer>"],
    "contexts": [["<retrieved paragraph 1>", "<retrieved paragraph 2>"]],
}
result = evaluate(Dataset.from_dict(samples), metrics=[faithfulness, answer_relevancy])
print(result)   # faithfulness + answer_relevancy scores (0–1)