import tensorflow as tf
import matplotlib.pyplot as plt
import io

class TensorboardLogger(object):

  def __init__(self, log_dir):
    self.writer = tf.summary.create_file_writer(log_dir)

  def log_scalar(self, tag, value, step):
    with self.writer.as_default():
      tf.summary.scalar(tag, value, step=step)

  def log_images(self, tag, images, step):
    with self.writer.as_default():
      tf.summary.image(tag, images, step=step, max_outputs=100)
