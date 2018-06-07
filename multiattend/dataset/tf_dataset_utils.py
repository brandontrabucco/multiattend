###########################
# Reset working directory #
###########################
import os
os.chdir("/home/brand/Research/multifaceted_attention")
###########################
# MultiAttend Package.... #
###########################
import tensorflow as tf

class TFDatasetUtils(object):

    def _init__(
            self):
        pass

    def set_args(
            self,
            config):
        self.config = config

    def tokenize_example(self, example):
        return tf.one_hot(
            example,
            self.config.dataset_range,
            axis=-1,
            dtype=tf.float32)
        
    def detokenize_example(self, example):
        return tf.argmax(
            example,
            axis=-1)

    def decode_record(self, queue):
        reader = tf.TextLineReader()
        key, text = reader.read(queue)
        columns = tf.decode_csv(
            text,
            [[self.config.dataset_default] for i in range(2 * self.config.dataset_columns)])
        x_inputs =  tf.stack(columns[:self.config.dataset_columns])
        x_labels =  tf.stack(columns[self.config.dataset_columns:])
        token_inputs = self.tokenize_example(x_inputs)
        return token_inputs, x_labels

    def generate_batch(
            self, 
            token_inputs,
            token_labels,
            capacity,
            min_after_dequeue):
        inputs_batch, labels_batch = tf.train.shuffle_batch(
            [token_inputs,token_labels],
            batch_size=self.config.batch_size,
            num_threads=self.config.num_threads,
            capacity=capacity,
            min_after_dequeue=min_after_dequeue)
        return inputs_batch, labels_batch