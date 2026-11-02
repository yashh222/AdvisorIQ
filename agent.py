
import logging
from pathlib import Path

import pandas as pd

from langchain.agents import create_agent
from langchain.agents.middleware import ModelFallbackMiddleware
from langchain_core.tools import Tool

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq

from rag import get_retriever
from checker import flag_transactions
from guardrails import make_safe


# --------------------------------------------------
# LOGGING
# --------------------------------------------------

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# --------------------------------------------------
# MODEL CONFIGURATION
# --------------------------------------------------

# Primary model: Gemini
gemini_llm = ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-lite",
    temperature=0,
    max_retries=1,
)

# Fallback model: Groq
groq_llm = ChatGroq(
    model="openai/gpt-oss-120b",
    api_key=None,  # Reads GROQ_API_KEY from environment
    temperature=0,
)

# Fallback middleware
fallback_middleware = ModelFallbackMiddleware(
    groq_llm
)


# --------------------------------------------------
# RAG CONFIGURATION
# --------------------------------------------------

retriever = get_retriever()


# --------------------------------------------------
# TRANSACTION FILE CONFIGURATION
# --------------------------------------------------

DEFAULT_UPLOAD_PATH = Path("data/_current_upload.csv")

# Updated by run_advisor() whenever a CSV is supplied.
CURRENT_UPLOAD_PATH = DEFAULT_UPLOAD_PATH


# --------------------------------------------------
# TOOL 1: MARKET DATA
# --------------------------------------------------

def market_data(ticker: str) -> str:
    import yfinance as yf

    ticker = ticker.strip().upper()

    hist = yf.Ticker(ticker).history(period="1mo")

    if hist.empty:
        return f"No market data available for {ticker}."

    price = hist["Close"].iloc[-1]
    month_ago = hist["Close"].iloc[0]

    change = (price - month_ago) / month_ago * 100

    return (
        f"{ticker}: latest price ${price:.2f}, "
        f"1-month change {change:+.1f}%."
    )


# --------------------------------------------------
# TOOL 2: RAG RESEARCH
# --------------------------------------------------

def research(query: str) -> str:

    hits = retriever.invoke(query)

    if not hits:
        return "No relevant documents found."

    return "\n\n".join(
        f"[source] {doc.page_content}"
        for doc in hits
    )


# --------------------------------------------------
# TOOL 3: TRANSACTION CHECKER
# --------------------------------------------------

def check_uploaded(_input: str) -> str:

    if not CURRENT_UPLOAD_PATH.exists():
        return "No transactions were uploaded for this question."

    try:
        transactions = pd.read_csv(CURRENT_UPLOAD_PATH)

        flagged = flag_transactions(transactions)

        if flagged.empty:
            return "No suspicious transactions found."

        return (
            "Suspicious transactions:\n"
            + flagged.to_string(index=False)
        )

    except Exception:
        logger.exception("Transaction checker failed.")
        return "Unable to analyze the uploaded transaction file."


# --------------------------------------------------
# REGISTER TOOLS
# --------------------------------------------------

TOOLS = [

    Tool(
        name="MarketData",
        func=market_data,
        description=(
            "Get the latest available price and 1-month trend "
            "for a stock ticker like AAPL."
        ),
    ),

    Tool(
        name="Research",
        func=research,
        description=(
            "Search financial documents to ground answers "
            "about companies, markets, and financial topics."
        ),
    ),

    Tool(
        name="CheckTransactions",
        func=check_uploaded,
        description=(
            "Analyze the user's uploaded transaction CSV "
            "and identify suspicious transactions. "
            "Use this tool whenever the user asks about "
            "uploaded transactions or suspicious activity."
        ),
    ),

]


# --------------------------------------------------
# SYSTEM PROMPT
# --------------------------------------------------

SYSTEM_PROMPT = """
You are AdvisorIQ, a wealth-management assistant.

Use the available tools to ground your answers in real data.

Rules:
1. Use MarketData for current stock information.
2. Use Research when relevant financial documents are needed.
3. Use CheckTransactions when the user asks about uploaded transactions.
4. Never invent financial data.
5. Never promise a specific investment return.
6. Explain why a transaction is suspicious using the available evidence.
7. Keep transaction explanations concise and easy to understand.
8. Clearly distinguish suspicious activity from confirmed fraud.
9. Explain financial risks in simple language.
10. Do not claim that a transaction is fraudulent based only on its risk score.
"""


# --------------------------------------------------
# CREATE AGENT
# --------------------------------------------------

agent = create_agent(
    model=gemini_llm,
    tools=TOOLS,
    system_prompt=SYSTEM_PROMPT,
    middleware=[fallback_middleware],
)


# --------------------------------------------------
# RESPONSE NORMALIZATION
# --------------------------------------------------

def extract_text(content) -> str:

    if isinstance(content, str):
        return content.strip()

    if isinstance(content, list):

        text_parts = []

        for block in content:

            if isinstance(block, dict):
                if block.get("type") == "text":
                    text_parts.append(block.get("text", ""))

            elif isinstance(block, str):
                text_parts.append(block)

        return "\n".join(text_parts).strip()

    if content is None:
        return ""

    return str(content).strip()


# --------------------------------------------------
# MAIN FUNCTION
# --------------------------------------------------

def run_advisor(
    question: str,
    csv_path: str | None = None
) -> str:

    global CURRENT_UPLOAD_PATH

    try:

        if not question or not question.strip():
            return "Please enter a question."

        # Use the CSV uploaded through the Gradio interface.
        if csv_path:
            CURRENT_UPLOAD_PATH = Path(csv_path)
        else:
            CURRENT_UPLOAD_PATH = DEFAULT_UPLOAD_PATH

        logger.info("Starting AdvisorIQ agent.")
        logger.info("Uploaded CSV: %s", CURRENT_UPLOAD_PATH)

        # Execute agent
        result = agent.invoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": question.strip()
                    }
                ]
            }
        )

        messages = result.get("messages", [])

        # Find the last AI response that does not contain tool calls.
        final_ai_messages = [
            message
            for message in messages
            if getattr(message, "type", None) == "ai"
            and not getattr(message, "tool_calls", [])
        ]

        if not final_ai_messages:
            raise RuntimeError("Agent did not generate a final AI response.")

        # Extract final answer
        answer = extract_text(final_ai_messages[-1].content)

        if not answer:
            raise RuntimeError("Final AI response is empty.")

        logger.info(
            "Final AI response extracted successfully. Length: %d",
            len(answer)
        )

        # Extract research sources for guardrails
        sources = [
            str(message.content)
            for message in messages
            if getattr(message, "name", "") == "Research"
            and "No relevant documents" not in str(message.content)
        ]

        # Apply output guardrails
        safe_answer = make_safe(answer, sources)

        logger.info(
            "Guardrail output type: %s",
            type(safe_answer).__name__
        )

        # Normalize guardrail output for Gradio
        safe_answer = extract_text(safe_answer)

        if not safe_answer:
            logger.error("Guardrails returned an empty response.")
            return (
                "The AI generated an answer, but the output safety "
                "check returned an empty response. Please try again."
            )

        logger.info("AdvisorIQ response completed successfully.")

        return safe_answer

    except Exception:

        logger.exception("AdvisorIQ agent execution failed.")

        return (
            "Unable to generate your analysis. "
            "Please check the terminal logs for the exact error "
            "and try again."
        )
