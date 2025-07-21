import os
import logging
from logging.handlers import RotatingFileHandler
from typing import Optional

# 日志存放目录（可配置）
LOG_DIR = os.getenv("LOG_DIR", os.path.join(os.path.dirname(__file__), "../logs"))
os.makedirs(LOG_DIR, exist_ok=True)


def setup_logging(
    log_file: str = "app.log",
    level: str = "INFO",
    max_bytes: int = 10_000_000,
    backup_count: int = 5,
):
    """
    初始化项目全局日志（建议在 main.py 第一个调用）
    - 控制台输出 INFO+
    - 文件轮转输出 DEBUG+
    """
    numeric_level = getattr(logging, level.upper(), None)
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
        handlers=[logging.StreamHandler()],
    )

    file_handler = RotatingFileHandler(
        os.path.join(LOG_DIR, log_file), maxBytes=max_bytes, backupCount=backup_count
    )
    file_handler.setLevel(numeric_level)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s %(name)s %(levelname)s: %(message)s")
    )
    logging.getLogger().addHandler(file_handler)


def get_logger(name: str = Optional[None]) -> logging.Logger:
    """
    获取指定模块 logger，推荐在每个文件开头调用：
    logger = get_logger(__name__)
    """
    return logging.getLogger(name)
