import smtplib
import os
import logging
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.application import MIMEApplication
from email.header import Header
from dotenv import load_dotenv

load_dotenv(".env")
email_server = os.getenv("EMAIL_SERVER")
email_port = os.getenv("EMAIL_PORT")
sender_email = os.getenv("EMAIL_SENDER")
email_password = os.getenv("EMAIL_PASSWORD")


class EmailServer():

    def __init__(self):
        self.server = smtplib.SMTP_SSL(email_server, email_port, timeout=30)
        self.server.login(sender_email, email_password)
    
    def send(self, user_email: str, subject: str, body: str, attachment_path: str="None"):

        msg = MIMEMultipart()
        msg["From"] = Header(sender_email, "utf-8")
        msg["To"] = Header(user_email, "utf-8")
        msg["Subject"] = Header(subject, "utf-8")

        msg.attach(MIMEText(body, "plain", "utf-8"))

        if attachment_path != "None":
            with open(attachment_path, "rb") as f:
                part = MIMEApplication(f.read(), Name=os.path.basename(attachment_path))
                part["Content-Disposition"] = f'attachment; filename="{os.path.basename(attachment_path)}"'
                msg.attach(part)
        
        try:
            self.server.sendmail(sender_email, user_email, msg.as_string())
            self.server.quit()
            logging.info(f"[Email] Success! Email to {user_email}")
        except Exception as e:
            logging.info(f"[Email] Failed: {e}")


if __name__ == '__main__':

    email_server = EmailServer()

    user_email = ""
    subject = "test email"
    body = "test email from OpenBioMed"
    attachment_path = "./tmp/temp_2025_03_03_14_38_19.zip"

    email_server.send(user_email=user_email, subject=subject, body=body, attachment_path=attachment_path)