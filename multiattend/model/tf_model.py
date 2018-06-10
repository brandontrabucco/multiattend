###########################
# Reset working directory #
###########################
import os
os.chdir("/home/btrabucco/research/multiattend")
###########################
# MultiAttend Package.... #
###########################
from multiattend.model.tf_model_args import TFModelArgs
from multiattend.model.tf_model_utils import TFModelUtils
import numpy as np

class TFModel(object):

    def __init__(
            self):
        self.tf_model_args = TFModelArgs()
        self.tf_model_utils = TFModelUtils()
        self.graph_is_built = False
    
    def initialize(
            self,
            load_dataset_fn):
        args = self.tf_model_args()
        self.tf_model_utils.set_args(args)
        if not self.graph_is_built:
            self.inputs_batch, self.labels_batch = self.tf_model_utils.load_dataset_fn(
                load_dataset_fn)
            self.tf_model_utils.build_parameters()
            self.tensor_encoded = self.tf_model_utils.encode(
                self.inputs_batch)
            self.tensor_logits, self.tensor_probs = self.tf_model_utils.decode(
                self.tensor_encoded)
            self.loss = self.tf_model_utils.loss_function(
                self.tensor_logits, 
                self.labels_batch)
            self.gradient = self.tf_model_utils.gradient(
                self.loss)
            self.tf_model_utils.finalize_graph()
            self.graph_is_built = True
        self.tf_model_utils.run_initialize()
            
    def train(
            self,
            num_iterations):
        (_grad, 
            actual_loss, 
            actual_step,
            actual_inputs,
            actual_probs, 
            actual_labels) = self.tf_model_utils.run_operation([
                self.gradient, 
                self.loss,
                self.tf_model_utils.global_step,
                self.inputs_batch, 
                self.tensor_probs, 
                self.labels_batch], iterations=num_iterations)
        actual_inputs = np.argmax(actual_inputs, axis=-1)
        actual_probs = np.argmax(actual_probs, axis=-1)
        actual_labels = np.argmax(actual_labels, axis=-1)
        
        correct_predictions = 0
        total_predictions = 0
        for a, b in zip(
                actual_probs.flatten().tolist(), 
                actual_labels.flatten().tolist()):
            total_predictions += 1
            if a == b:
                correct_predictions += 1
        actual_accuracy = correct_predictions / total_predictions
        
        np.set_printoptions(precision=1)
        
        print(
            "Iteration: %d" % actual_step,
            "Loss: %.2f" % actual_loss, 
            "Accuracy: %.2f" % actual_accuracy,
            "Input: %s" % str(actual_inputs[0, :]),
            "Prediction: %s" % str(actual_probs[0, :]),
            "Label: %s" % str(actual_labels[0, :]))
        return {
            "loss": actual_loss, 
            "accuracy": actual_accuracy, 
            "inputs": actual_inputs, 
            "probs": actual_probs,
            "labels": actual_labels}
    
    def test(
            self):
        (actual_inputs,
            actual_probs, 
            actual_labels) = self.tf_model_utils.run_operation([
                self.inputs_batch, 
                self.tensor_probs, 
                self.labels_batch], iterations=1)
        actual_inputs = np.argmax(actual_inputs, axis=-1)
        actual_probs = np.argmax(actual_probs, axis=-1)
        actual_labels = np.argmax(actual_labels, axis=-1)
        
        correct_predictions = 0
        total_predictions = 0
        for a, b in zip(
                actual_probs.flatten().tolist(), 
                actual_labels.flatten().tolist()):
            total_predictions += 1
            if a == b:
                correct_predictions += 1
        actual_accuracy = correct_predictions / total_predictions
        return {
            "accuracy": actual_accuracy, 
            "inputs": actual_inputs, 
            "probs": actual_probs,
            "labels": actual_labels}
    
    def save(
            self):
        self.tf_model_utils.run_checkpoint()