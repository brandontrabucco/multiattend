###########################
# Reset working directory #
###########################
import os
os.chdir("/home/btrabucco/research/multiattend")
###########################
# MultiAttend Package.... #
###########################
import tensorflow as tf

class TFModelUtils(object):

    def __init__(
            self):
        self.g = tf.Graph()
        self.session = tf.Session(graph=self.g)

    def set_args(
            self,
            config):
        self.config = config

    def load_dataset_fn(
            self,
            load_fn):
        with self.g.as_default():
            tf.logging.info("Fetching the dataset.")
            return load_fn()
        
    def build_parameters(
            self):
        with self.g.as_default():
        
            tf.logging.info("Building all parameters.")
            self.encode_weights = tf.get_variable(
                "encode_weights",
                shape=[self.config.dataset_range, self.config.encoder_depth//2],
                initializer=tf.truncated_normal_initializer())
            self.encode_biases = tf.get_variable(
                "encode_biases",
                shape=[1, 1, self.config.encoder_depth//2],
                initializer=tf.constant_initializer(1.0))
            self.encoder_lstm_fw = tf.contrib.rnn.DropoutWrapper(
                tf.contrib.rnn.LSTMCell(
                    self.config.encoder_depth//2),
                output_keep_prob=self.config.keep_probability)
            self.encoder_state_fw = self.encoder_lstm_fw.zero_state(
                self.config.batch_size,
                dtype=tf.float32)
            self.encoder_lstm_bw = tf.contrib.rnn.DropoutWrapper(
                tf.contrib.rnn.LSTMCell(
                    self.config.encoder_depth//2),
                output_keep_prob=self.config.keep_probability)
            self.encoder_state_bw = self.encoder_lstm_bw.zero_state(
                self.config.batch_size,
                dtype=tf.float32)

            self.attend_weights = tf.get_variable(
                "attend_weights",
                shape=[self.config.encoder_depth + (2 * self.config.decoder_depth)],
                initializer=tf.truncated_normal_initializer())
            self.attend_biases = tf.get_variable(
                "attend_biases",
                shape=[1, 1],
                initializer=tf.constant_initializer(1.0))

            self.decoder_lstm_fw = tf.contrib.rnn.DropoutWrapper(
                tf.contrib.rnn.LSTMCell(
                    self.config.decoder_depth),
                output_keep_prob=self.config.keep_probability)
            self.decoder_state_fw = self.decoder_lstm_fw.zero_state(
                self.config.batch_size,
                dtype=tf.float32)
            self.decode_weights = tf.get_variable(
                "decode_weights",
                shape=[self.config.decoder_depth, self.config.dataset_range],
                initializer=tf.truncated_normal_initializer())
            self.decode_biases = tf.get_variable(
                "decode_biases",
                shape=[1, 1, self.config.dataset_range],
                initializer=tf.constant_initializer(1.0))
            
            self.trainable_vars = tf.trainable_variables()

    def encode(
            self,
            tensor_input):
        with self.g.as_default():
            tf.logging.info("Running the encoder.")
            tensor_encoded = (tf.tensordot(
                tensor_input, 
                self.encode_weights, 
                1) + self.encode_biases)
            tensor_encoded = tf.reshape(
                tensor_encoded, 
                [self.config.batch_size, self.config.dataset_columns, self.config.encoder_depth//2])
            tensor_encoded, _states = tf.nn.bidirectional_dynamic_rnn(
                self.encoder_lstm_fw,
                self.encoder_lstm_bw,
                tensor_encoded,
                initial_state_fw=self.encoder_state_fw,
                initial_state_bw=self.encoder_state_bw)
            return tf.concat(
                tensor_encoded, 
                axis=-1)
        
    def attend(
            self,
            tensor_encoded):
        with self.g.as_default():
            def loop_fn(
                    time, 
                    cell_output, 
                    cell_state, 
                    loop_state):
              
                next_state = (self.decoder_state_fw 
                    if cell_output is None else cell_state)
                finished = (time >= tf.fill(
                    [self.config.batch_size], 
                    self.config.dataset_columns))
                
                tensor_combined = tf.concat([
                    tf.tile(
                        tf.expand_dims(
                            next_state.c, 
                            axis=1), 
                        [1, self.config.dataset_columns, 1]),
                    tf.tile(
                        tf.expand_dims(
                            next_state.h, 
                            axis=1), 
                        [1, self.config.dataset_columns, 1]),
                    tensor_encoded], axis=-1)
                
                tensor_attend = tf.nn.softmax(
                    (tf.tensordot(
                        tensor_combined, 
                        self.attend_weights, 
                        1) + self.attend_biases),
                    axis=-1)
                tensor_attend = tf.expand_dims(
                    tensor_attend, 
                    axis=2)
                tensor_context = tf.reduce_sum(
                    tensor_encoded * tf.tile(
                        tensor_attend, 
                        [1, 1, self.config.encoder_depth]),
                    axis=1)
                
                return (
                    finished, 
                    tensor_context, 
                    next_state,
                    cell_output, 
                    None)
            return loop_fn

    def decode(
            self,
            tensor_encoded):
        with self.g.as_default():
            tf.logging.info("Running the decoder.")
            tensor_decoded_ta, _cell_state, _loop_state = tf.nn.raw_rnn(
                self.decoder_lstm_fw, 
                self.attend(tensor_encoded))
            tensor_decoded = tf.transpose(
                tensor_decoded_ta.stack(), [1, 0, 2])
            tensor_logits = (tf.tensordot(
                tensor_decoded, 
                self.decode_weights, 
                1) + self.decode_biases)
            tensor_probs = tf.nn.softmax(
                tensor_logits,
                axis=-1)
            return tensor_logits, tensor_probs

    def loss_function(
            self,
            tensor_logits,
            tensor_labels):
        with self.g.as_default():
            return (self.config.weight_penalty * self.weight_decay()) + tf.reduce_sum(
                tf.nn.softmax_cross_entropy_with_logits_v2(
                    logits=tf.reshape(tensor_logits, [-1, self.config.dataset_range]),
                    labels=tf.reshape(tensor_labels, [-1, self.config.dataset_range])))
        
    def weight_decay(
            self):
        with self.g.as_default():
            decay_term = 0
            for w in self.trainable_vars:
                decay_term += tf.nn.l2_loss(w)
            return decay_term
        
    def gradient(
            self, 
            loss):
        with self.g.as_default():
            tf.logging.info("Computing the gradient.")
            
            self.global_step = tf.Variable(0, trainable=False)
            self.increment = tf.assign(
                self.global_step, 
                self.global_step + 1)
            self.saver = tf.train.Saver()
            
            self.learning_rate = tf.train.exponential_decay(
                self.config.learning_rate,
                self.global_step,
                self.config.decay_steps,
                self.config.decay_rate,
                staircase=True)
            self.optimizer = tf.train.AdamOptimizer(
                learning_rate=self.learning_rate)
            self.gradient = self.optimizer.minimize(
                loss,
                var_list=self.trainable_vars)
            self.init_grad_only = tf.variables_initializer(
                self.optimizer.variables())
            return self.gradient
        
    def finalize_graph(
            self):
        with self.g.as_default():
            tf.logging.info("Finalizing the graph.")
            self.init_op = tf.global_variables_initializer()
        self.g.finalize()
            
    def run_initialize(
            self):
        tf.logging.info("Initializing the model.")
        if self.config.use_previous:
            self.run_restore()
            self.session.run(self.init_grad_only)
        else:
            self.session.run(self.init_op)
        tf.train.start_queue_runners(sess=self.session)

    def run_operation(
            self,
            operation,
            iterations=1,
            increment=True):
        tf.logging.info("Running the model.")
        for i in range(iterations):
            results = self.session.run(operation)
            if increment:
                self.session.run(self.increment)
        return results
    
    def run_checkpoint(
            self):
        tf.logging.info("Saving model checkpoint.")
        self.saver.save(
            self.session, 
            "saves/model.ckpt", 
            global_step=self.global_step)
        
    def run_restore(
            self):
        tf.logging.info("Loading model checkpoint.")
        if self.config.target_checkpoint is None:
            ckpt = tf.train.latest_checkpoint("saves/")
        else:
            ckpt = self.config.target_checkpoint
        self.saver.restore(
            self.session,
            ckpt)
