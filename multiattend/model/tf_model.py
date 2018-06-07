###########################
# Reset working directory #
###########################
import os
os.chdir("/home/brand/Research/multifaceted_attention")
###########################
# MultiAttend Package.... #
###########################
from multiattend.model.tf_model_args import TFModelArgs
from multiattend.model.tf_model_utils import TFModelUtils

class TFModel(object):

    def __init__(
            self):
        self.tf_model_args = TFModelArgs()
        self.tf_model_utils = TFModelUtils()

    def build(
            self,
            tensor_input):
        args = self.tf_model_args()
        return self.tf_model_utils.decode(
            self.tf_model_utils.encode(
                tensor_input),
            args.batch_size,
            args.dataset_columns * 2,
            args.dataset_range)

    def train(
            self,
            load_dataset_fn,
            num_iterations):
        inputs_batch, labels_batch = self.tf_model_utils.load_dataset_fn(
            load_dataset_fn)
        tensor_decoded, tensor_logits = self.build(
            inputs_batch)


        print(self.tf_model_utils.run_operation(
            [inputs_batch, labels_batch], 
            1))

        indices = self.tf_model_utils.indices(
            tensor_decoded)
        loss = self.tf_model_utils.sparse_cross_entropy(
            tensor_logits, 
            labels_batch)
        gradient = self.tf_model_utils.gradient(
            loss)
        (_g, 
            actual_loss, 
            actual_indices, 
            actual_inputs,
            actual_labels) = self.tf_model_utils.run_operation(
                [gradient, loss, indices, inputs_batch, labels_batch], 
                num_iterations)
        actual_accuracy = (sum([1 if a == b else 0 for a, b in zip(
                actual_indices.tolist(), 
                actual_labels.tolist())]) / actual_labels.size())
        print(
            "Loss: %.2f" % loss, 
            "Accuracy: %.2f" % actual_accuracy)
        return (
            actual_loss, 
            actual_accuracy, 
            actual_inputs, 
            actual_labels)