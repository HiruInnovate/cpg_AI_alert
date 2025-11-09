# services/mailer.py
import json
import os
import ssl
import smtplib
from email.message import EmailMessage
from services.logger_config import get_logger

# Initialize logger
logger = get_logger(__name__)

CFG_PATH = "config/contacts.json"


def send_mail(subject: str, body: str):
    """
    Sends an email using SMTP configuration defined in config/contacts.json.
    Includes detailed logging for tracing and debugging.
    """
    logger.info("[MAILER] Preparing to send email alert.")

    # Load configuration
    if not os.path.exists(CFG_PATH):
        logger.error(f"[MAILER] Missing configuration file: {CFG_PATH}")
        return {"status": "error", "message": "Missing contacts.json"}

    try:
        cfg = json.load(open(CFG_PATH))
    except Exception as e:
        logger.error(f"[MAILER] Failed to read mail configuration: {e}", exc_info=True)
        return {"status": "error", "message": "Invalid mail configuration"}

    smtp = cfg.get("smtp", {})
    contacts = cfg.get("contacts", [])

    # Validate recipients
    if not contacts:
        logger.warning("[MAILER] No recipients found in contacts.json.")
        return {"status": "no_recipients"}

    # Demo / Dry-run mode (no actual SMTP call)
    if os.environ.get("STREAMLIT_RUNNING_IN_DEMO", "1") == "1":
        logger.info(f"[MAILER] Running in demo mode — mock sending mail to {len(contacts)} recipients.")
        return {"status": "mock_sent", "to": contacts, "subject": subject}

    # Prepare email
    try:
        msg = EmailMessage()
        msg.set_content(body)
        msg["Subject"] = subject
        msg["From"] = smtp.get("from", "noreply@example.com")
        msg["To"] = ", ".join(contacts)
    except Exception as e:
        logger.error(f"[MAILER] Error constructing email message: {e}", exc_info=True)
        return {"status": "error", "message": "Failed to build email"}

    # Establish connection and send
    try:
        logger.info(f"[MAILER] Connecting to SMTP server: {smtp.get('host')}:{smtp.get('port')}")
        context = ssl.create_default_context()

        with smtplib.SMTP(smtp.get("host"), smtp.get("port")) as s:
            if smtp.get("use_tls", True):
                s.starttls(context=context)
                logger.debug("[MAILER] TLS connection established successfully.")

            if smtp.get("username"):
                logger.debug(f"[MAILER] Authenticating as {smtp.get('username')}")
                s.login(smtp.get("username"), smtp.get("password"))

            s.send_message(msg)

        logger.info(f"[MAILER] Email successfully sent to {len(contacts)} recipients.")
        return {"status": "sent", "to": contacts}

    except smtplib.SMTPException as e:
        logger.error(f"[MAILER] SMTP error during send: {e}", exc_info=True)
        return {"status": "error", "message": f"SMTP error: {str(e)}"}
    except Exception as e:
        logger.error(f"[MAILER] Unexpected error during email send: {e}", exc_info=True)
        return {"status": "error", "message": str(e)}
