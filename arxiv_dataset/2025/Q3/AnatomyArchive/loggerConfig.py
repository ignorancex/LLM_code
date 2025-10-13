import os
import logging
from typing import Union
from datetime import datetime
from logging.handlers import SMTPHandler

logger = None


def create_log_file(log_file_dir: str, email_config: Union[dict, None] = None):
    """
    :param log_file_dir: log file name
    :param email_config: dict for setting configuration to use SMTPHandler. By default, gmail SMTP is used.
    :return:
    """
    def _get_receiver_list(x):
        return [x] if not isinstance(x, list) else x

    global logger
    # Create logger
    logger = logging.getLogger('AnatomyArchive')
    logger.setLevel(logging.WARNING)  # Capture WARNING and above

    # File Handler (log warnings and errors)
    now = datetime.now()
    # noinspection SpellCheckingInspection
    time_str = now.strftime('%Y-%m-%d_%Hh-%Mmin')
    log_file_name = os.path.join(log_file_dir, f'AnatomyArchive_{time_str}.log')
    file_handler = logging.FileHandler(log_file_name)
    file_handler.setLevel(logging.WARNING)
    logger.addHandler(file_handler)
    if email_config is not None:
        smtp_handler = SMTPHandler(
                mailhost=email_config.get('MailHostWithPort', ("smtp.gmail.com", 587)),
                fromaddr=email_config['SenderEmail'],
                toaddrs=_get_receiver_list(email_config['ReceiverEmails']),
                subject="Execution Completed with Errors",
                credentials=(email_config['SenderEmail'], email_config['SenderEmailPassword']),
                secure=()
            )
        smtp_handler.setLevel(logging.ERROR)  # Only send error summaries
        logger.addHandler(smtp_handler)

