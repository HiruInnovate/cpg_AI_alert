"""
services/mailer.py

Robust SMTP mailer for RCA Email Summary.
Supports:
- SMTP with TLS
- Demo mode (no sending)
- JSON-based config (config/contacts.json)
- Rich logging
"""

import os
import json
import ssl
import smtplib
from email.message import EmailMessage
from services.logger_config import get_logger

logger = get_logger(__name__)

CONFIG_FILE = "config/contacts.json"


def _load_config():
    """Load mail + contact settings."""
    if not os.path.exists(CONFIG_FILE):
        logger.error(f"[MAILER] Missing config file: {CONFIG_FILE}")
        return None

    try:
        with open(CONFIG_FILE, "r", encoding="utf8") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"[MAILER] Failed to parse config: {e}", exc_info=True)
        return None


def send_mail(subject: str, body: str):
    """
    Send an email to all contacts defined in config/contacts.json.

    Returns:
      {
        "status": "sent" | "mock_sent" | "error",
        "to": list_of_recipients
      }
    """

    logger.info("[MAILER] Preparing email...")

    cfg = _load_config()
    if not cfg:
        return {"status": "error", "message": "Invalid mail config"}

    smtp_cfg = cfg.get("smtp", {})
    contacts = cfg.get("contacts", [])

    if not contacts:
        logger.warning("[MAILER] No recipients defined.")
        return {"status": "no_recipients", "to": []}

    # Demo safeguard
    if os.environ.get("STREAMLIT_RUNNING_IN_DEMO", "1") == "1":
        logger.info(f"[MAILER] Demo mode active — mock send to {len(contacts)} recipients")
        return {"status": "mock_sent", "to": contacts}

    # Construct email
    try:
        msg = EmailMessage()
        msg["From"] = smtp_cfg.get("from", "noreply@example.com")
        msg["To"] = ", ".join(contacts)
        msg["Subject"] = subject
        msg.set_content(body)

    except Exception as e:
        logger.error(f"[MAILER] Failed to build email object: {e}", exc_info=True)
        return {"status": "error", "message": "Failed to build email"}

    # Send via SMTP
    try:
        logger.info(
            f"[MAILER] Connecting to SMTP {smtp_cfg.get('host')}:{smtp_cfg.get('port')}"
        )

        context = ssl.create_default_context()

        with smtplib.SMTP(smtp_cfg.get("host"), smtp_cfg.get("port")) as server:

            if smtp_cfg.get("use_tls", True):
                server.starttls(context=context)
                logger.info("[MAILER] TLS session established")

            if smtp_cfg.get("username"):
                logger.info("[MAILER] Authenticating...")
                server.login(smtp_cfg.get("username"), smtp_cfg.get("password"))

            server.send_message(msg)

        logger.info(f"[MAILER] Email sent to {len(contacts)} recipients")
        return {"status": "sent", "to": contacts}

    except smtplib.SMTPException as e:
        logger.error(f"[MAILER] SMTP error: {e}", exc_info=True)
        return {"status": "error", "message": f"SMTP error: {e}"}

    except Exception as e:
        logger.error(f"[MAILER] Unexpected error: {e}", exc_info=True)
        return {"status": "error", "message": str(e)}
