import logging

from tensorboardX import SummaryWriter


class EventLogger:
    def __init__(self, root_dir):
        self.root_dir = root_dir
        if root_dir is None:
            self.tensorboard_logger = None
        else:
            root_dir.mkdir(parents=True, exist_ok=False)
            self.tensorboard_logger = SummaryWriter(str(root_dir))
        self.console = logging.getLogger(__name__)

    def log_scalar(self, tag, value, iteration):
        if self.tensorboard_logger is not None:
            self.tensorboard_logger.add_scalar(tag, value, iteration)

    def debug(self, msg):
        self.console.debug(msg)

    def info(self, msg):
        self.console.info(msg)

    def warning(self, msg):
        self.console.warning(msg)

    def error(self, msg):
        self.console.error(msg)

    def critical(self, msg):
        self.console.critical(msg)
