import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.header import Header
import sys
import os
from email.utils import formataddr

# 确保可以找到配置模块
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from config import smtp_server, smtp_port, email_from_addr, email_password, email_from_name

class EmailSender:
    def __init__(self, from_addr=email_from_addr, password=email_password, smtp_server=smtp_server, smtp_port=smtp_port):
        self.from_addr = from_addr
        self.password = password
        self.smtp_server = smtp_server
        self.smtp_port = smtp_port

    def send_email(self, subject, message, to_addr, recipient_name='Recipient Name', html=False):
        server = smtplib.SMTP_SSL(self.smtp_server, self.smtp_port)
        server.login(self.from_addr, self.password)
        
        if html:
            msg = MIMEText(message, 'html', 'utf-8')
        else:
            msg = MIMEText(message, 'plain', 'utf-8')
        
        msg['From'] = formataddr((str(Header(email_from_name, 'utf-8')), self.from_addr))
        msg['To'] = formataddr((str(Header(recipient_name, 'utf-8')), to_addr))
        msg['Subject'] = Header(subject, 'utf-8')
        
        server.sendmail(self.from_addr, [to_addr], msg.as_string())
        server.quit()
        print("邮件发送成功！")

if __name__ == "__main__":
    email_sender = EmailSender()
    subject = "SMTP 邮件测试"
    message = "这是一封通过SMTP发送的Python脚本测试邮件。"
    to_addr = "745339023@qq.com"
    recipient_name = "Mr.Light"
    email_sender.send_email(subject, message, to_addr, recipient_name)
