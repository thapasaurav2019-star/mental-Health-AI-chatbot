# app/email_utils.py
import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import logging

logger = logging.getLogger(__name__)

def send_verification_email(to_email: str, token: str, base_url: str) -> bool:
    """
    Send a verification email to the user.
    
    Args:
        to_email: Recipient email address
        token: Verification token
        base_url: Base URL of the application (e.g., http://127.0.0.1:5500)
    
    Returns:
        bool: True if email was sent successfully, False otherwise
    """
    # Get email configuration from environment variables
    smtp_host = os.getenv("SMTP_HOST", "smtp.gmail.com")
    smtp_port = int(os.getenv("SMTP_PORT", "587"))
    smtp_user = os.getenv("SMTP_USER", "")
    smtp_password = os.getenv("SMTP_PASSWORD", "")
    from_email = os.getenv("FROM_EMAIL", smtp_user)
    
    # If email configuration is not set, log the verification link instead
    # This is useful for development/testing
    if not smtp_user or not smtp_password:
        verification_link = f"{base_url}/verify.html?token={token}"
        logger.info(
            "\n" + "="*80 + "\n"
            "SMTP not configured - Email would be sent to: %s\n"
            "Verification link: %s\n"
            "="*80,
            to_email, verification_link
        )
        # Return True in development mode to allow signup to continue
        return True
    
    try:
        # Create verification link
        verification_link = f"{base_url}/verify.html?token={token}"
        
        # Create email message
        msg = MIMEMultipart('alternative')
        msg['Subject'] = 'Verify your Mental Health Chatbot account'
        msg['From'] = from_email
        msg['To'] = to_email
        
        # Create email body (plain text and HTML)
        text = f"""
        Welcome to Mental Health Chatbot!
        
        Please verify your email address by clicking the link below:
        
        {verification_link}
        
        This link will expire in 24 hours.
        
        If you didn't create this account, please ignore this email.
        """
        
        html = f"""
        <html>
          <body>
            <h2>Welcome to Mental Health Chatbot!</h2>
            <p>Please verify your email address by clicking the button below:</p>
            <p>
              <a href="{verification_link}" 
                 style="background-color: #7c3aed; color: white; padding: 12px 24px; 
                        text-decoration: none; border-radius: 8px; display: inline-block;">
                Verify Email
              </a>
            </p>
            <p>Or copy and paste this link into your browser:</p>
            <p><a href="{verification_link}">{verification_link}</a></p>
            <p><small>This link will expire in 24 hours.</small></p>
            <p><small>If you didn't create this account, please ignore this email.</small></p>
          </body>
        </html>
        """
        
        # Attach both text and HTML versions
        part1 = MIMEText(text, 'plain')
        part2 = MIMEText(html, 'html')
        msg.attach(part1)
        msg.attach(part2)
        
        # Send email
        with smtplib.SMTP(smtp_host, smtp_port) as server:
            server.starttls()
            server.login(smtp_user, smtp_password)
            server.send_message(msg)
        
        logger.info("Verification email sent successfully to %s", to_email)
        return True
        
    except Exception as e:
        logger.error("Failed to send verification email to %s: %s", to_email, e)
        return False
