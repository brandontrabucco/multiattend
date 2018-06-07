###########################
# Reset working directory #
###########################
import os
os.chdir("/home/brand/Research/multifaceted_attention")
###########################
# MultiAttend Package.... #
###########################
import tensorflow as tf

class TFModelUtils(object):

    def __init__(
            self):
        self.g = tf.Graph()

    def set_args(
            self,
            config):
        self.config = config

    def load_dataset_fn(
            self,
            load_fn):
        with self.g.as_default():
            return load_fn()

    def encode(
            self,
            tensor_input):
        with self.g.as_default():
            lstm_cell_fw = tf.contrib.rnn.LSTMCell(
                64)
            lstm_cell_bw = tf.contrib.rnn.LSTMCell(
                64)
            self.tensor_encoded, _states = tf.nn.bidirectional_dynamic_rnn(
                lstm_cell_fw,
                lstm_cell_bw,
                tensor_input,
                dtype=tf.float32)
            return tf.concat(
                self.tensor_encoded, 
                axis=-1)

    def decode(
            self,
            tensor_encoded,
            batch_size,
            sequence_length,
            vocab_depth):
        with self.g.as_default():

            lstm_cell_fw = tf.contrib.rnn.LSTMCell(
                128)
            cell_state = lstm_cell_fw.zero_state(
                batch_size,
                dtype=tf.float32)
            attend_weights = tf.get_variable(
                "attend_weights",
                shape=[256],
                initializer=tf.truncated_normal_initializer())
            attend_biases = tf.get_variable(
                "attend_biases",
                shape=[1],
                initializer=tf.constant_initializer(1.0))

            self.tensor_embedded = []
            for i in range(sequence_length):
                tensor_attend = tf.nn.softmax(
                    (tf.tensordot(
                        tf.concat(
                            [
                                tf.tile(
                                    tf.reshape(
                                        cell_state.h, 
                                        [batch_size, 1, 128]), 
                                    [1, sequence_length, 1]),
                                tensor_encoded], axis=-1), 
                        attend_weights, 
                        1) + attend_biases),
                    axis=-1)

                tensor_context = tf.reduce_sum(
                    tensor_encoded * tf.tile(
                        tf.reshape(
                            tensor_attend, 
                            [batch_size, sequence_length, 1]), 
                        [1, 1, 128]),
                    axis=1)
                output, cell_state = lstm_cell_fw(
                    tensor_context,
                    cell_state)
                self.tensor_embedded.append(output)

            self.tensor_embedded = tf.reshape(
                tf.stack(self.tensor_embedded), 
                [batch_size, sequence_length, 128])
            decode_weights = tf.get_variable(
                "decode_weights",
                shape=[128, vocab_depth],
                initializer=tf.truncated_normal_initializer())
            decode_biases = tf.get_variable(
                "decode_biases",
                shape=[vocab_depth],
                initializer=tf.constant_initializer(1.0))

            self.tensor_logits = (tf.tensordot(
                self.tensor_embedded, 
                decode_weights, 
                1) + decode_biases)
            self.tensor_decoded = tf.nn.softmax(
                self.tensor_logits,
                axis=-1)
            self.init_op = tf.global_variables_initializer()
            self.global_vars = tf.global_variables()
            return self.tensor_decoded, self.tensor_logits

    def indices(
            self,
            tensor_decoded):
        with self.g.as_default():
            return tf.argmax(tensor_decoded, axis=-1)

    def sparse_cross_entropy(
            self,
            logits,
            labels):
        with self.g.as_default():
            return tf.reduce_sum(
                tf.nn.sparse_softmax_cross_entropy_with_logits(
                    logits=logits,
                    labels=labels))

    def gradient(
            self, 
            loss):
        self.optimizer = tf.train.AdamOptimizer(
            learning_rate=0.001)
        return self.optimizer.minimize(loss, var_list=self.global_vars)

    def run_operation(
            self,
            op,
            n,
            save=False):
        self.g.finalize()
        session = tf.Session(graph=self.g)
        session.run(self.init_op)
        for i in range(n):
            tf.logging.info(
                session.run(op))
        if save:
            saver = tf.train.Saver(var_list=self.global_vars)
            saver.save(session, "plots/")



    