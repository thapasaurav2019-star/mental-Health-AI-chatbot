import os, smtplib, logging, ssl
from email.message import EmailMessage
from typing import Optional
from pathlib import Path

EMAIL_HOST = os.getenv("EMAIL_HOST")
EMAIL_PORT = int(os.getenv("EMAIL_PORT", "587"))
EMAIL_USERNAME = os.getenv("EMAIL_USERNAME")
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD")
EMAIL_FROM_ADDRESS = os.getenv("EMAIL_FROM_ADDRESS", EMAIL_USERNAME or "no-reply@example.com")
EMAIL_USE_TLS = str(os.getenv("EMAIL_USE_TLS", "1")).lower() in ("1","true","yes","on")
EMAIL_USE_SSL = str(os.getenv("EMAIL_USE_SSL", "0")).lower() in ("1","true","yes","on")

TEMPLATE_PATH = Path(__file__).parent.parent / "templates" / "verification_email.html"

SUPPORTIVE_INTRO = (
    "Welcome — we're glad you're here. This mental health companion is designed to be gentle, supportive, and non-judgmental. "
    "Please confirm your email so we can keep your account secure."
)


def _build_verification_html(verify_url: str, user_name: Optional[str]) -> str:
    name = user_name or "there"
    html_template = (
        TEMPLATE_PATH.read_text(encoding="utf-8") if TEMPLATE_PATH.exists() else (
            "<html><body style='font-family:system-ui;background:#f9fafb;padding:24px'>"
            "<h2 style='color:#4b3fa7;margin-top:0'>Confirm your email</h2>"
            f"<p>Hi {name},</p>"
            f"<p>{SUPPORTIVE_INTRO}</p>"
            f"<p><a href='{verify_url}' style='background:#4b3fa7;color:#fff;padding:10px 16px;border-radius:6px;text-decoration:none'>Verify Email</a></p>"
            "<p style='font-size:13px;color:#555'>If you did not create an account you can safely ignore this email.</p>"
            "<p style='font-size:12px;color:#777'>Take things at your own pace. We're here when you feel ready.</p>"
            "</body></html>"
        )
    )
    return html_template.replace("{{VERIFY_URL}}", verify_url).replace("{{NAME}}", name)


def send_email(subject: str, to_email: str, html_body: str, text_body: Optional[str] = None) -> bool:
    """Send email using configured SMTP provider. Returns True if sent, False otherwise."""
    # Validate minimal config
    if not EMAIL_HOST or not EMAIL_USERNAME or not EMAIL_PASSWORD:
        logging.warning("Email service not fully configured; would send to %s subject %s", to_email, subject)
        return False
    try:
        msg = EmailMessage()
        msg["Subject"] = subject
        msg["From"] = EMAIL_FROM_ADDRESS
        msg["To"] = to_email
        if text_body:
            msg.set_content(text_body)
            msg.add_alternative(html_body, subtype="html")
        else:
            msg.set_content(html_body, subtype="html")

        if EMAIL_USE_SSL:
            context = ssl.create_default_context()
            with smtplib.SMTP_SSL(EMAIL_HOST, EMAIL_PORT, context=context) as server:
                server.login(EMAIL_USERNAME, EMAIL_PASSWORD)
                server.send_message(msg)
        else:
            with smtplib.SMTP(EMAIL_HOST, EMAIL_PORT) as server:
                if EMAIL_USE_TLS:
                    server.starttls()
                server.login(EMAIL_USERNAME, EMAIL_PASSWORD)
                server.send_message(msg)
        return True
    except Exception:
        logging.exception("Failed to send email to %s", to_email)
        return False


def send_verification_email(user_email: str, user_name: Optional[str], token: str, base_url: str) -> bool:
    verify_url = f"{base_url.rstrip('/')}/verify?token={token}"
    html = _build_verification_html(verify_url, user_name)
    text = f"Hi {user_name or 'there'},\nPlease verify your email by visiting: {verify_url}\nIf you did not sign up you can ignore this message."
    return send_email("Verify your email", user_email, html, text)
