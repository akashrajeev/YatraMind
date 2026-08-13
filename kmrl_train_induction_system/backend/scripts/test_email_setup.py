"""Manual SMTP diagnostic; intentionally not collected as a pytest test."""
import asyncio
import os
from dotenv import load_dotenv
from fastapi_mail import FastMail, MessageSchema, ConnectionConfig

load_dotenv()


def get_env(keys, default=None):
    for key in keys:
        if val := os.getenv(key):
            return val
    return default


async def main():
    user = get_env(["SMTP_USER", "MAIL_USERNAME", "USER"])
    password = get_env(["SMTP_PASSWORD", "SMTP_PASS", "MAIL_PASSWORD", "PASS"])
    host = get_env(["SMTP_HOST", "MAIL_SERVER", "HOST"], "smtp-relay.brevo.com")
    port = int(get_env(["SMTP_PORT", "EMAIL_PORT"], 587))
    secure = get_env(["SMTP_USE_TLS", "SECURE"], "true").lower() == "true"
    use_tls = secure if port != 587 else True
    use_ssl = port == 465
    if not user or not password:
        raise RuntimeError("SMTP credentials are missing from the environment")
    conf = ConnectionConfig(
        MAIL_USERNAME=user, MAIL_PASSWORD=password, MAIL_FROM=user,
        MAIL_PORT=port, MAIL_SERVER=host, MAIL_STARTTLS=use_tls,
        MAIL_SSL_TLS=use_ssl, USE_CREDENTIALS=True,
    )
    fm = FastMail(conf)
    message = MessageSchema(subject="SMTP Test", recipients=[user], body="SMTP test", subtype="plain")
    await fm.send_message(message)
    print("SMTP test email sent successfully")


if __name__ == "__main__":
    asyncio.run(main())
