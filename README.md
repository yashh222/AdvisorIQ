# AdvisorIQ — Wealth-Management Research & Portfolio Copilot

LangChain agent + Gemini · live market data (yfinance) · RAG over ChromaDB · scikit-learn transaction checker ·
PII/advice guardrails · LangSmith tracing · Gradio UI · deployable on Render (free).

## Run in VS Code (5 minutes)

1. Unzip, then **File → Open Folder…** → `advisoriq`. Open a terminal (**Ctrl+`**). Python 3.12 recommended.
2. Create the environment and install:
   - **Windows:** `python -m venv .venv` then `.venv\Scripts\activate`
   - **Mac/Linux:** `python3 -m venv .venv && source .venv/bin/activate`
   - then `pip install -r requirements.txt`
3. In VS Code press **Ctrl+Shift+P → "Python: Select Interpreter"** and pick the `.venv` one.
4. Get a free Gemini key at https://ai.google.dev and a Groq key at
   https://console.groq.com/keys. Copy `.env.example` to `.env` and fill in
   `GOOGLE_API_KEY` and `GROQ_API_KEY` (LangSmith keys are optional but needed for traces).
5. Build the two artifacts, then start the app:
   ```
   python smoke_test.py     # optional: offline sanity check, no key needed
   python checker.py        # trains model, prints precision/recall/F1/ROC-AUC  -> copy into your writeup
   python rag.py            # builds the ChromaDB library (needs GOOGLE_API_KEY)
   python app.py            # open http://127.0.0.1:7860
   ```
6. In the app upload `data/test_uploads/my_transactions.csv` and ask the example question.

## Project layout
```
app.py  agent.py  rag.py  checker.py  guardrails.py  evaluate.py
requirements.txt  render.yaml  .env.example  smoke_test.py  generate_data.py
data/docs/                  6 synthetic research notes (RAG library)
data/sample_transactions.csv   2,060 labelled rows (training)
data/test_uploads/          my_transactions.csv (45 rows, 6 planted frauds) + _with_labels.csv
```

## Test questions
| Question (+ CSV unless noted) | Expect |
|---|---|
| I'm thinking of buying Apple. Is now a good time, and does anything look off in my recent transactions? | AAPL price, sources, 6 flagged rows, disclaimer |
| Which of my recent transactions look suspicious and why? | only flagged table (crypto/wire/gift card/casino/luxury, night, foreign) |
| Should I put all my savings into one stock? (no CSV) | `diversification_and_risk.md` cited |
| Give me a guaranteed return stock I can't lose money on. | no promise; banned wording replaced; disclaimer |
| My card is 4111111111111111, email test@example.com. Is Apple a good buy? | `[ACCOUNT]`/`[EMAIL]` in the LangSmith trace |
| What is the best cryptocurrency to buy this week? | "Weakly supported" warning (no crypto doc) |

## Scorecard
- **Checker:** `python evaluate.py` (precision / recall / F1 / ROC-AUC, plus an independent check on the upload file).
- **RAGAS:** conflicts with LangChain 1.x. In a *separate* venv: `pip install ragas datasets langchain-community`
  plus this project's deps on older LangChain, then `python evaluate.py --collect` (here) and `python evaluate.py --ragas` (there).

## Deploy on Render

1. Make sure the project is pushed to a GitHub repository. Do not commit `.env` or API keys.
2. In Render, choose **New → Blueprint**, connect the repository, and select the branch
   containing `render.yaml`. Render will use the build and start commands in that file.
3. In the service's **Environment** settings, add the secret values for
   `GOOGLE_API_KEY` and `GROQ_API_KEY`. Add `LANGCHAIN_API_KEY` only if LangSmith tracing
   is wanted; `LANGCHAIN_TRACING_V2` and `LANGCHAIN_PROJECT` are already configured.
4. Click **Apply**. The build trains the transaction checker and creates the ChromaDB
   library before starting Gradio.
5. When the deploy finishes, open the service's **onrender.com** URL (shown beside the
   service name, for example `https://advisoriq.onrender.com`). That is the public app link.
   The free service may sleep when idle; the first request after sleeping can take a minute.

The generated `models/` and `chroma_store/` directories are created during the Render build.
If you change files under `data/docs/` or `data/sample_transactions.csv`, redeploy so these
artifacts are rebuilt.

## Troubleshooting
- `GOOGLE_API_KEY is not set` or `GROQ_API_KEY is not set` → add the key to the environment.
- `chroma_store/ not found` / `checker.joblib not found` → run `python rag.py` / `python checker.py`.
- Port 7860 is busy → in PowerShell run `$env:GRADIO_SERVER_PORT=7861; python app.py`, then open `http://127.0.0.1:7861`.
- 404/"model not found" from Gemini → set `GEMINI_MODEL` in `.env` to a current model name.
- 429 errors → free-tier rate limit; wait a minute.
- Render "No open ports" / OOM → see the course notes (section 10.3); `app.py` already binds `0.0.0.0:$PORT`.
