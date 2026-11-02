# smoke_test.py -- offline checks, NO API key needed:  python smoke_test.py
import pandas as pd
from guardrails import DISCLAIMER, make_safe, redact_pii

# guardrails
t = redact_pii("card 4111111111111111 mail a.b@example.com call 9876543210 amount 107.44")
assert "[ACCOUNT]" in t and "[EMAIL]" in t and "[PHONE]" in t and "107.44" in t, t
s = make_safe("This is a guaranteed return and risk-free.", ["a.md"])
assert "guaranteed return" not in s and "risk-free" not in s and s.endswith(DISCLAIMER)
assert "Weakly supported" in make_safe("x", [])
print("guardrails OK")

# checker
from pathlib import Path
from checker import MODEL_PATH, flag_transactions, train
if not Path(MODEL_PATH).exists():
    train()
flagged, total = flag_transactions("data/test_uploads/my_transactions.csv")
truth = pd.read_csv("data/test_uploads/my_transactions_with_labels.csv")
print(f"checker flagged {len(flagged)} of {total}; truly suspicious: {int(truth['label'].sum())}")
assert len(flagged) >= 4
print(flagged.to_string(index=False))
print("checker OK")
