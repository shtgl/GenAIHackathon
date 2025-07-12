import logging
import os
from datetime import datetime
from autogen_core.logging import LLMCallEvent


class LLMUsageTracker(logging.Handler):
    def __init__(self, log_file: str = "llm_usage.log") -> None:
        """Logging handler that tracks the number of tokens used in the prompt and completion."""
        super().__init__()
        self._prompt_tokens = 0
        self._completion_tokens = 0

        # Create a logs directory if it doesn't exist
        log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
        os.makedirs(log_dir, exist_ok=True)

        # Create a timestamped log file
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.log_file_path = os.path.join(log_dir, f"{log_file.split('.')[0]}_{timestamp}.log")

        # Set up file logging
        self.file_handler = logging.FileHandler(self.log_file_path)
        self.file_handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        self.file_handler.setFormatter(formatter)

        # Create a dedicated logger for LLMUsageTracker
        self.logger = logging.getLogger(f"LLMUsageTracker_{timestamp}")
        self.logger.setLevel(logging.DEBUG)
        self.logger.addHandler(self.file_handler)

    @property
    def tokens(self) -> int:
        return self._prompt_tokens + self._completion_tokens

    @property
    def prompt_tokens(self) -> int:
        return self._prompt_tokens

    @property
    def completion_tokens(self) -> int:
        return self._completion_tokens

    def reset(self) -> None:
        self._prompt_tokens = 0
        self._completion_tokens = 0

    def emit(self, record: logging.LogRecord) -> None:
        """Emit the log record. To be used by the logging module."""
        try:
            if isinstance(record.msg, LLMCallEvent):
                event = record.msg
                self._prompt_tokens += event.prompt_tokens
                self._completion_tokens += event.completion_tokens
                self.logger.info(
                    f"Prompt tokens: {event.prompt_tokens}, Completion tokens: {event.completion_tokens}, Total tokens: {self.tokens}"
                )
        except Exception:
            self.handleError(record)


def setup_logger(log_file: str = "rag_agent_chat.log") -> logging.Logger:
    """
    Sets up a centralized logger for the RAG application.
    """
    # Create a logs directory if it doesn't exist
    log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
    os.makedirs(log_dir, exist_ok=True)

    # Create a timestamped log file
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_file_path = os.path.join(log_dir, f"{log_file.split('.')[0]}_{timestamp}.log")

    # Configure the logger
    logger = logging.getLogger("rag_document_agent")
    logger.setLevel(logging.DEBUG)

    # Create a file handler
    file_handler = logging.FileHandler(log_file_path)
    file_handler.setLevel(logging.DEBUG)

    # Create a log formatter
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(formatter)

    # Add the file handler to the logger
    logger.addHandler(file_handler)

    return logger 