###########################
# Reset working directory #
###########################
import os
os.chdir("/home/btrabucco/research/multiattend")
###########################
# MultiAttend Package.... #
###########################
from multiattend.model.tf_model import TFModel
from multiattend.dataset.tf_dataset import TFDataset
from multiattend.args.tf_register_args import TFRegisterArgs
from datetime import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

class TFExperiment(object):

    def __init__(
            self):
        self.tf_model = TFModel()
        self.tf_dataset = TFDataset()
        self.register = TFRegisterArgs()
        self.register("--iterations_per_test", int, 1)
        self.register("--num_tests", int, 1)
        self.register("--num_repetitions", int, 1)
        self.best_accuracy = 0.0

    def train(
            self):
        args = self.register.parse_args()
        self.all_accuracies = []
        
        for r in range(args.num_repetitions):
            working_accuracies = []
            self.tf_model.initialize(
                self.tf_dataset.get_training_batch)
            
            for i in range(args.num_tests):
                results = self.tf_model.train(
                    args.iterations_per_test)
                if (results["accuracy"] > self.best_accuracy and 
                        args.checkpoint_after_test):
                    self.tf_model.save()
                    self.best_accuracy = results["accuracy"]
                working_accuracies += [results["accuracy"]]
                    
            if args.checkpoint_after_train:
                self.tf_model.save()
            if results["accuracy"] > self.best_accuracy:
                self.best_accuracy = results["accuracy"]
            self.all_accuracies += [working_accuracies]
                
        self.generate_plots()
        return self.best_accuracy
    
    def generate_plots(
            self):
        args = self.register.parse_args()
        
        accuracies = np.array(self.all_accuracies)
        mean_accuracies = np.mean(accuracies, axis=0)
        std_accuracies = np.std(accuracies, axis=0)
        x_indices = np.arange(1, mean_accuracies.size + 1) * args.iterations_per_test
        
        plt.plot(x_indices, mean_accuracies, "-g")
        plt.fill_between(x_indices, mean_accuracies-std_accuracies, mean_accuracies+std_accuracies, color="r")
        plt.title("Training Accuracy Reps=" + str(args.num_repetitions))
        plt.xlabel("Training Iterations Batch=" + str(args.batch_size))
        plt.ylabel("Fraction of Batch Examples Correct")
        plt.grid(True)
        plt.savefig("plots/" + str(datetime.now()) + "_training_accuracy.png")
        
    def test(
            self):
        args = self.register.parse_args()
        
        working_accuracies = []
        self.tf_model.initialize(
            self.tf_dataset.get_training_batch)

        for i in range(args.num_tests):
            results = self.tf_model.test()
            working_accuracies += [results["accuracy"]]
                
        final_accuracy = sum(working_accuracies) / len(working_accuracies)
        print("Final Accuracy: %.2f" % final_accuracy)
        return final_accuracy
        
        
