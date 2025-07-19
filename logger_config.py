import logging
import os
from logging.handlers import RotatingFileHandler

# 日志目录
LOG_DIR = "logs"
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

# 日志文件路径
LOG_FILE = os.path.join(LOG_DIR, "user_gradio.log")

# 创建日志器
logger = logging.getLogger("my_project")
logger.setLevel(logging.DEBUG)

# 日志格式
formatter = logging.Formatter(
    '[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# 控制台日志处理器
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(formatter)

# 文件日志处理器（支持轮转）
file_handler = RotatingFileHandler(LOG_FILE, maxBytes=5*1024*1024, backupCount=3,  encoding='utf-8') #  encoding='utf-8'显示中文
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(formatter)

# 添加处理器（防止重复添加）
if not logger.handlers:
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
