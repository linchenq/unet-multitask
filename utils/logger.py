import tensorflow as tf
import logging


class Logger(object):
    def __init__(self, log_dir):
        """Create a summary writer logging to log_dir."""
        self.writer = tf.summary.FileWriter(log_dir)

        """Create a file writter logging to .log"""
        LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
        DATE_FORMAT = "%m/%d/%Y %H:%M:%S %p"
        LOG_FILENAME = "../logs/model.log"

        logging.basicConfig(filename=LOG_FILENAME,
                            level=logging.DEBUG,
                            format=LOG_FORMAT,
                            datefmt=DATE_FORMAT)

    def scalar_summary(self, tag, value, step):
        """Log a scalar variable."""
        summary = tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=value)])
        self.writer.add_summary(summary, step)

    def list_of_scalars_summary(self, tag_value_pairs, step):
        """Log scalar variables."""
        summary = tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=value) for tag, value in tag_value_pairs])
        self.writer.add_summary(summary, step)

    def log_summary(self, mode, msg):
        if mode == "INFO":
            logging.info(msg)
        elif mode == "WARNING":
            logging.warning(msg)
        else:
            raise NotImplementedError
