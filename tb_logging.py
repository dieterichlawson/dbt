import tensorflow as tf
import matplotlib.pyplot as plt
import io

class TensorboardLogger(object):

    def __init__(self, log_dir):
        self.writer = tf.summary.create_file_writer(log_dir)

    def log_scalar(self, tag, value, step):
        with self.writer.as_default():
            tf.summary.scalar(tag, value, step=step)

#    def log_images(self, tag, images, step):
#        im_summaries = []
#        for nr, img in enumerate(images):
#            # Write the image to a string
#            s = io.BytesIO()
#            plt.imsave(s, img, format='png')
#
#            # Create an Image object
#            img_sum = tf.Summary.Image(encoded_image_string=s.getvalue(),
#                                       height=img.shape[0],
#                                       width=img.shape[1])
#            # Create a Summary value
#            im_summaries.append(tf.Summary.Value(tag='%s/%d' % (tag, nr),
#                                                 image=img_sum))
#
#        # Create and write Summary
#        summary = tf.Summary(value=im_summaries)
#        self.writer.add_summary(summary, step)
