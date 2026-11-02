
import os
import tempfile
import logging

from dotenv import load_dotenv

load_dotenv()

import gradio as gr

from agent import run_advisor
from guardrails import redact_pii


# --------------------------------------------------
# LOGGING
# --------------------------------------------------

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# --------------------------------------------------
# MAIN FUNCTION
# --------------------------------------------------

def advise(question: str, csv_file) -> str:

    if not question or not question.strip():
        return "### Please enter a question before clicking Analyze."

    csv_path = None

    try:

        # Process uploaded CSV
        if csv_file is not None:

            uploaded_path = (
                csv_file
                if isinstance(csv_file, (str, os.PathLike))
                else csv_file.name
            )

            with open(uploaded_path, "r", encoding="utf-8-sig") as file:
                original_content = file.read()

            # Redact sensitive information
            cleaned_content = redact_pii(original_content)

            # Create temporary CSV
            with tempfile.NamedTemporaryFile(
                mode="w",
                suffix=".csv",
                delete=False,
                encoding="utf-8",
                newline=""
            ) as temp_file:

                temp_file.write(cleaned_content)
                csv_path = temp_file.name

        logger.info("Sending request to AdvisorIQ.")

        # Run AI agent
        response = run_advisor(
            question.strip(),
            csv_path
        )

        # Normalize response
        if isinstance(response, list):

            response = "\n".join(
                item.get("text", "")
                for item in response
                if isinstance(item, dict)
                and item.get("type") == "text"
            )

        elif isinstance(response, dict):

            response = response.get("text", str(response))

        response = str(response).strip()

        if not response:
            logger.error("AdvisorIQ returned an empty response.")
            return "### No response received.\nPlease try again."

        logger.info("Response successfully generated.")

        return response

    except Exception:

        logger.exception("Gradio request failed.")

        return (
            "### Something went wrong\n"
            "AdvisorIQ could not complete your request. "
            "Please check the terminal logs and try again."
        )

    finally:

        # Clean up temporary CSV
        if csv_path and os.path.exists(csv_path):

            try:
                os.remove(csv_path)

            except OSError:
                logger.warning("Could not remove temporary CSV file.")


# --------------------------------------------------
# PROFESSIONAL UI STYLING
# --------------------------------------------------

custom_css = """

/* Main application */
.gradio-container {
    max-width: 1180px !important;
    margin: auto !important;
    padding: 20px !important;
}

/* Header */
.main-header {
    text-align: center;
    padding: 22px 10px 16px;
}

.main-header h1 {
    font-size: 36px !important;
    font-weight: 750 !important;
    letter-spacing: -1.2px;
    margin: 0 0 8px 0 !important;
}

.main-header p {
    font-size: 15px;
    opacity: 0.7;
    margin: 0;
}

/* Intro text */
.intro-text {
    text-align: center;
    font-size: 14px;
    opacity: 0.75;
    margin-bottom: 22px;
}

/* Section headings */
.section-title {
    font-size: 16px !important;
    font-weight: 650 !important;
    margin-bottom: 8px !important;
}

/* Input box */
.question-input textarea {
    font-size: 14px !important;
    line-height: 1.6 !important;
}

/* Output panel: fixed height and internal scrolling */
.analysis-panel {
    height: 430px !important;
    min-height: 430px !important;
    max-height: 430px !important;
    overflow-y: auto !important;
    overflow-x: hidden !important;
    border: 1px solid var(--block-border-color) !important;
    border-radius: 12px !important;
    padding: 18px !important;
    background: var(--block-background-fill) !important;
    box-sizing: border-box !important;
}

/* Ensure long content wraps inside output */
.analysis-panel > div,
.analysis-panel .prose {
    max-width: 100% !important;
    overflow-wrap: anywhere !important;
    word-break: normal !important;
}

/* Output scrollbar */
.analysis-panel::-webkit-scrollbar {
    width: 7px;
}

.analysis-panel::-webkit-scrollbar-thumb {
    background: var(--border-color-primary);
    border-radius: 10px;
}

.analysis-panel::-webkit-scrollbar-track {
    background: transparent;
}

/* Analyze button */
button.primary {
    min-height: 46px !important;
    font-size: 14px !important;
    font-weight: 650 !important;
    border-radius: 9px !important;
}

/* Upload component */
.file-upload {
    border-radius: 10px !important;
}

/* Footer */
.footer-note {
    text-align: center;
    font-size: 12px;
    opacity: 0.55;
    padding: 20px 0 8px;
}

/* Example questions */
.examples-area {
    margin-top: 22px;
}

/* Mobile layout */
@media (max-width: 768px) {

    .gradio-container {
        padding: 12px !important;
    }

    .main-header h1 {
        font-size: 29px !important;
    }

    .analysis-panel {
        height: 350px !important;
        min-height: 350px !important;
        max-height: 350px !important;
    }
}

"""


# --------------------------------------------------
# GRADIO UI
# --------------------------------------------------

with gr.Blocks(
    title="AdvisorIQ | Wealth Management Copilot",
    css=custom_css,
    theme=gr.themes.Soft()
) as demo:

    # Header
    gr.HTML(
        """
        <div class="main-header">
            <h1>AdvisorIQ</h1>
            <p>Your intelligent wealth management assistant</p>
        </div>
        """
    )

    gr.Markdown(
        """
        <div class="intro-text">
            Analyze financial data, research companies,
            and identify potentially suspicious transactions.
        </div>
        """
    )

    with gr.Row(equal_height=True, variant="panel"):

        # LEFT: USER INPUT
        with gr.Column(scale=5, min_width=280):

            gr.Markdown(
                "### Ask your question",
                elem_classes=["section-title"]
            )

            question = gr.Textbox(
                placeholder=(
                    "Example: Analyze my transactions and "
                    "identify suspicious activity..."
                ),
                lines=5,
                max_lines=8,
                show_label=False,
                elem_classes=["question-input"]
            )

            gr.Markdown(
                "### Upload transactions"
            )

            csv_file = gr.File(
                label="Choose a CSV file (optional)",
                file_types=[".csv"],
                type="filepath"
            )

            submit = gr.Button(
                "Analyze",
                variant="primary",
                size="lg"
            )

            gr.Markdown(
                """
                <div style="font-size:12px; opacity:0.6; margin-top:5px;">
                    Upload a CSV file when you need transaction analysis.
                </div>
                """
            )

        # RIGHT: AI OUTPUT
        with gr.Column(scale=7, min_width=300):

            gr.Markdown(
                "### Analysis results",
                elem_classes=["section-title"]
            )

            output = gr.Markdown(
                value=(
                    "### Welcome to AdvisorIQ\n\n"
                    "Your financial analysis will appear here.\n\n"
                    "Enter a question and click **Analyze** to get started."
                ),
                elem_classes=["analysis-panel"]
            )

    # Example questions
    gr.Markdown(
        "### Try asking",
        elem_classes=["examples-area"]
    )

    gr.Examples(
        examples=[
            [
                "Analyze my uploaded transactions and identify suspicious activity."
            ],
            [
                "What are the main risks of investing in technology stocks?"
            ],
            [
                "Give me the latest available market trend for AAPL."
            ],
        ],
        inputs=question
    )

    # Footer
    gr.HTML(
        """
        <div class="footer-note">
            AdvisorIQ · AI-powered financial insights
            <br>
            Educational purposes only. Not financial advice.
        </div>
        """
    )

    # Actions
    submit.click(
        fn=advise,
        inputs=[question, csv_file],
        outputs=output
    )

    question.submit(
        fn=advise,
        inputs=[question, csv_file],
        outputs=output
    )


# --------------------------------------------------
# LAUNCH
# --------------------------------------------------

if __name__ == "__main__":

    port = int(
        os.environ.get(
            "PORT",
            os.environ.get("GRADIO_SERVER_PORT", 7860)
        )
    )

    demo.launch(
        server_name="0.0.0.0",
        server_port=port,
        show_error=True
    )
