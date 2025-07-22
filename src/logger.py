import logging
import os
from logging.handlers import RotatingFileHandler
from datetime import datetime


class Logger:
    def __init__(self, log_dir="results/logs", log_name="run", max_bytes=5*1024*1024, backup_count=3):
        """
        Initializes a logger that writes to both console and rotating file.

        Args:
            log_dir (str): Directory where logs are stored.
            log_name (str): Base name for log file.
            max_bytes (int): Max size of log file before rotation (default 5MB).
            backup_count (int): Number of rotated log files to keep.
        """
        os.makedirs(log_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        log_file = os.path.join(log_dir, f"{log_name}_{timestamp}.log")

        self.logger = logging.getLogger(log_name)
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False  # Avoid duplicate logs

        if not self.logger.handlers:
            formatter = logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s")

            file_handler = RotatingFileHandler(log_file, maxBytes=max_bytes, backupCount=backup_count)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)

            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)

    def get_logger(self):
        return self.logger
