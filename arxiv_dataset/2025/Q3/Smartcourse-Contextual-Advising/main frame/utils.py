from datetime import datetime
import smtplib
from email.message import EmailMessage
import requests
import time, csv, os               

def write_log(message):
    with open("log.txt", "a", encoding="utf-8") as file:
        file.write(f"{datetime.now()}: {message}\n")

def send_enrollment_email(to_email, message_body):
    smtp_server = "smtp.qq.com"
    smtp_port   = 587
    sender_email    = "your_email_address"
    sender_password = "your_smtp_authorisation_code"

    msg = EmailMessage()
    msg['Subject'] = 'SmartCourse Enrollment Notification'
    msg['From']    = sender_email
    msg['To']      = to_email
    msg.set_content(message_body)

    smtp = smtplib.SMTP(smtp_server, smtp_port)
    smtp.starttls()
    smtp.login(sender_email, sender_password)
    smtp.send_message(msg)
    smtp.quit()

def send_grade_email(to_email, message_body):
    smtp_server = "smtp.qq.com"
    smtp_port   = 587
    sender_email    = "your_email_address"
    sender_password = "your_smtp_authorisation_code"

    msg = EmailMessage()
    msg['Subject'] = 'SmartCourse Grading Notification'
    msg['From']    = sender_email
    msg['To']      = to_email
    msg.set_content(message_body)

    smtp = smtplib.SMTP(smtp_server, smtp_port)
    smtp.starttls()
    smtp.login(sender_email, sender_password)
    smtp.send_message(msg)
    smtp.quit()


def ask_ai_question(prompt):
    """
    Calls the local Ollama model and returns the (reply, latency) binary: 
    reply —— natural language answer given by LLM (string) 
    latency —— end-to-end generation elapsed time ( float, seconds)
    """
    start = time.time()
    try:
        resp = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": "llama3.1:8b",
                  "prompt": prompt,
                  "stream": False},
            timeout=300
        )
        data = resp.json()
        full_text = data.get("response", "")
        reply = full_text.split("</think>")[-1].strip() if "</think>" in full_text else full_text.strip()
    except Exception as e:
        reply = f"AI model failed to respond: {e}"

    latency = time.time() - start

    # Record token/latency for subsequent analysis
    word_cnt = len(reply.split())
    with open("latency_log.csv", "a", newline="") as f:
        csv.writer(f).writerow([word_cnt, latency])

    return reply, latency
