
import os
import sys
import asyncio
from dotenv import load_dotenv
from fastapi_mail import FastMail, MessageSchema, ConnectionConfig

# Load env from .env
load_dotenv()

def get_env(keys, default=None):
    for key in keys:
        if val := os.getenv(key):
            return val
    return default

async def test_email():
    print("--- SMTP Debugger ---")
    
    user = get_env(["SMTP_USER", "MAIL_USERNAME", "USER"])
    password = get_env(["SMTP_PASSWORD", "SMTP_PASS", "MAIL_PASSWORD", "PASS"])
    host = get_env(["SMTP_HOST", "MAIL_SERVER", "HOST"], "smtp-relay.brevo.com")
    port = int(get_env(["SMTP_PORT", "EMAIL_PORT"], 587))
    secure = get_env(["SMTP_USE_TLS", "SECURE"], "true").lower() == "true"
    
    print(f"Host: {host}")
    print(f"Port: {port}")
    print(f"User: {user}")
    print(f"Pass: {'*' * 8 if password else 'MISSING'}")
    
    # Smart TLS logic (mirroring the fix)
    use_tls = secure
    use_ssl = False
    
    if port == 587:
        print("Note: Port 587 detected. Forcing STARTTLS=True (ignoring SECURE=false if set).")
        use_tls = True
        use_ssl = False
    elif port == 465:
         print("Note: Port 465 detected. Forcing SSL=True.")
         use_tls = False
         use_ssl = True
         
    print(f"Effective TLS (STARTTLS): {use_tls}")
    print(f"Effective SSL: {use_ssl}")
    
    if not user or not password:
        print("ERROR: Credentials missing in .env")
        return

    conf = ConnectionConfig(
        MAIL_USERNAME=user,
        MAIL_PASSWORD=password,
        MAIL_FROM=user,
        MAIL_PORT=port,
        MAIL_SERVER=host,
        MAIL_STARTTLS=use_tls,
        MAIL_SSL_TLS=use_ssl,
        USE_CREDENTIALS=True
    )
    
    fm = FastMail(conf)
    
    print("\nAttempting to send test email to sender address...")
    message = MessageSchema(
        subject="SMTP Test",
        recipients=[user], # Send to self
        body="If you read this, SMTP is working!",
        subtype="plain"
    )
    
    try:
        await fm.send_message(message)
        print("\n✅ SUCCESS! Email sent successfully.")
    except Exception as e:
        print(f"\n❌ FAILED: {str(e)}")
        if "535" in str(e):
            print("Hint: 535 usually means bad password or 2FA/AppPassword required.")
        if "timeout" in str(e).lower():
           print("Hint: Timeout might mean firewall or wrong port/security combo.")

if __name__ == "__main__":
    asyncio.run(test_email())
