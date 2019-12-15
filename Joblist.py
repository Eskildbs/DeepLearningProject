import Corn_segmentation
import data_util
import numpy as np
import torch
import json
import os
import random
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.nn as nn

class Joblist():
    def __init__(self, input_path='Joblist', output_path='output/'):
        self.GPU_ENABLED = torch.cuda.is_available()
        self.SAVE_OUTPUT = True
        self.input_path = input_path
        self.output_path = output_path
        self.files = eval(open(input_path, 'r').read())
        self.names = {}
        for file in self.files:
            self.names.update({str(file): file.split('/')[-1]})

    def run(self, train=True, save_model=True, save_test_output=False, save_validation_output =False, save_filters=False):
        for file in self.files:
            if self.files[file] != 1: # skips networks, that is set not to run in joblist
                name = self.names[str(file)]
                self.current_job = Load_job_text(input_path=file, output_path=self.output_path + name + '/', SAVE_OUTPUT=True, name=name)
                self.current_job.set_parameters(self.LEARNING_RATE, self.loss_function, self.optimizer, self.weight_decay,
                                                self.validation_size, self.max_iter, self.log_every, self.batch_size,
                                                self.seed, self.stopping_criteria, self.dynamic_regularization)
                self.current_job.set_data(self.annotated_path, self.test_path, self.class_dict, self.data_augmentation)
                self.files[self.current_job.input_path] = 0.5
                self.current_job.run(train, save_model, save_test_output, save_validation_output, save_filters)
                self.files[self.current_job.input_path] = 1
                with open(self.input_path, 'w') as file:
                    file.write(json.dumps(self.files))


    def grid_run(self, train=True, save_model=True, save_test_output=False,
                 save_validation_output=False, save_filters=False,
                 grid_type1=None, grid_vals1=None,
                 grid_type2=None, grid_vals2=None):

        for job in self.jobs:
            initial = job.net.state_dict() # initial weights and biases

            if not os.path.isdir(job.output_path):
                os.mkdir(job.output_path)
            self.files[job.input_path] = 0.5
            with open(self.input_path, 'w') as file:
                file.write(json.dumps(self.files))
            for val1 in grid_vals1:
                job.net.load_state_dict(initial)
                job.net.eval()
                job.output_path = job.output_path + grid_type1 + str(val1)
                exec('job.' + grid_type1 + '=' + str(val1))

                if grid_type2!=None:
                    for val2 in grid_vals2:
                        job.net.load_state_dict(initial)
                        job.net.eval()
                        exec('job.' + grid_type2 + '=' + val2)
                        job.output_path = job.output_path + grid_type2 + str(val2) + '/'
                        job.run(train, save_model, save_test_output, save_validation_output, save_filters)
                else:
                    job.output_path = job.output_path + '/'
                    job.run(train, save_model, save_test_output, save_validation_output, save_filters)

            self.files[job.input_path] = 1
            with open(self.input_path, 'w') as file:
                file.write(json.dumps(self.files))


    def set_parameters(self, LEARNING_RATE=None, loss_function=None,
                   optimizer=None, weight_decay=None, validation_size=None, max_iter=None, log_every=None, batch_size=None,
                   seed = None, stopping_criteria=None, dynamic_regularization=None):
        self.LEARNING_RATE = LEARNING_RATE
        self.loss_function = loss_function
        self.optimizer = optimizer
        self.weight_decay = weight_decay
        self.validation_size = validation_size
        self.max_iter = max_iter
        self.log_every = log_every
        self.batch_size = batch_size
        self.seed = seed
        self.stopping_criteria = stopping_criteria
        self.dynamic_regularization = dynamic_regularization

    def set_data(self, annotated_path, test_path, class_dict, data_augmentation):
        self.annotated_path = annotated_path
        self.test_path = test_path
        self.class_dict = class_dict
        self.data_augmentation = data_augmentation

class Job():
    def __init__(self, output_path, SAVE_OUTPUT, name):
        self.GPU_ENABLED = torch.cuda.is_available()
        self.LEARNING_RATE = None
        self.loss_function = None
        self.weight_decay = None
        self.validation_size = None
        self.max_iter = None
        self.log_every = None
        self.batch_size = None
        self.seed = None
        self.optimizer = None
        self.stopping_criteria = None
        self.dynamic_regularization = None
        self.output_path = output_path
        self.load_model()
        self.SAVE_OUTPUT = SAVE_OUTPUT
        self.name = name

    def save_model(self, output_path):
        if not os.path.isdir(output_path + 'model'):
            os.mkdir(output_path + 'model')
        torch.save(self.net.state_dict(), output_path + 'model/model')

    def save_state(self, i):
        """
        Saves state. This contains all parameters, set data, current iterations done and so forth.
        """
        if not os.path.isdir(self.output_path + 'model'):
            os.mkdir(self.output_path + 'model')
        file = open(self.output_path + 'model/stats.txt', 'w')
        file.write('{')
        file.write(str(self.net) + '\n')
        file.write('LEARNING_RATE: ' + str(self.LEARNING_RATE) + ',\n')
        file.write('loss_function: ' + str(self.loss_function) + ',\n')
        file.write('weight_decay: ' + str(self.weight_decay) + ',\n')
        file.write('validation_size: ' + str(self.validation_size) + ',\n')
        file.write('log_every: ' + str(self.log_every) + ',\n')
        file.write('batch_size: ' + str(self.batch_size) + ',\n')
        file.write('seed: ' + str(self.seed) + ',\n')
        file.write('optimizer: ' + str(self.optimizer) + ',\n')
        file.write('class_dict: ' + str(self.class_dict) + ',\n')
        file.write('max_iter: ' + str(self.max_iter) + ',\n')
        file.write('current_iteration: ' + str(i))
        file.write('}')
        file.close()


    def set_parameters(self, LEARNING_RATE=None, loss_function=None, optimizer=None, weight_decay=None,
                       validation_size=None, max_iter=None, log_every=None, batch_size=None, seed=None,
                       stopping_criteria=None, dynamic_regularization=None):

        """ Default settings """
        if self.LEARNING_RATE is None:
            self.LEARNING_RATE = 0.001
        if self.loss_function is None:
            self.loss_function = nn.MSELoss(reduction='sum')
        if self.weight_decay is None:
            self.weight_decay = 0.001
        if self.validation_size is None:
            self.validation_size = 4
        if self.max_iter is None:
            self.max_iter = 250
        if self.log_every is None:
            self.log_every = 10
        if self.batch_size is None:
            self.batch_size = 4
        if self.seed is None:
            self.seed = 1
        if self.optimizer is None:
            self.optimizer = optim.Adam(self.net.parameters(), lr=self.LEARNING_RATE, weight_decay=self.weight_decay)
        if self.stopping_criteria is None:
            self.stopping_criteria = Corn_segmentation.No_stopping_criteria()
        if self.dynamic_regularization is None:
            self.dynamic_regularization = dynamic_regularization

        """ Adjusted settings """
        if LEARNING_RATE is not None:
            self.LEARNING_RATE = LEARNING_RATE
        if loss_function is not None:
            self.loss_function = loss_function
        if weight_decay is not None:
            self.weight_decay = weight_decay
        if validation_size is not None:
            self.validation_size = validation_size
        if max_iter is not None:
            self.max_iter = max_iter
        if log_every is not None:
            self.log_every = log_every
        if batch_size is not None:
            self.batch_size = batch_size
        if seed is not None:
            self.seed = seed
        if optimizer is not None:
            self.optimizer = optimizer
        if stopping_criteria is not None:
            self.stopping_criteria = stopping_criteria
        if dynamic_regularization is not None:
            self.dynamic_regularization = dynamic_regularization

        self.optimizer = optim.Adam(self.net.parameters(), lr=self.LEARNING_RATE, weight_decay=self.weight_decay)

    def set_data(self, annotated_path, test_path, class_dict, data_augmentation):
        self.class_dict = class_dict
        self.data_augmentation = data_augmentation
        self.test_dataset = data_util.Test_data()
        self.test_dataset.load_images(test_path)
        self.annotaed_dataset = data_util.Annotated_data(class_dict)
        self.annotaed_dataset.load_images(annotated_path)
        self.annotaed_dataset.batch_generator(batch_size=self.batch_size,
                                     iterations=self.max_iter,
                                     val_size=self.validation_size,
                                     data_augmentation=data_augmentation)

    def train(self):
        self.net.train()
        self.training_iter, self.training_loss, self.training_accs = [], [], []
        self.validation_iter, self.validation_loss, self.validation_accs = [], [], []
        for i, (batch_train, batch_vali) in enumerate(zip(self.annotaed_dataset.batch_ind,
                                                          self.annotaed_dataset.validation_ind)):
            # Training
            self.inputs = self.annotaed_dataset.get_input(batch_train, augment_data=self.data_augmentation)
            self.output_ = self.net(self.inputs)
            self.targets = self.annotaed_dataset.get_targets(batch_train)
            self.batch_loss = self.loss_function(self.output_, self.targets)
            self.optimizer.zero_grad()
            self.batch_loss.backward()
            self.optimizer.step()

            # Optional: Displaying augmented images
            self.show_augmented_images(batch_train, mask=False, save=True)

            # Saving train results in this iteration
            self.training_iter.append(i)
            self.training_loss.append(self.batch_loss.data.cpu().numpy() / len(batch_train))
            self.training_accs.append(Corn_segmentation.get_accuracy(self.output_, self.targets))

            # Saving validation results in this iteration
            self.net.eval()
            self.inputs = self.annotaed_dataset.get_input(batch_vali)
            self.output_ = self.net(self.inputs)
            self.targets = self.annotaed_dataset.get_targets(batch_vali)
            self.batch_loss = self.loss_function(self.output_, self.targets)
            self.validation_iter.append(i)
            self.validation_loss.append(self.batch_loss.data.cpu().numpy() / len(batch_vali))
            self.validation_accs.append(Corn_segmentation.get_accuracy(self.output_, self.targets))

            # Logs training and validation results thus far
            if i%self.log_every == 0:
                self.show_training_graphs(i)
                print('Best validation accuracy achieved: ' + str(max(self.validation_accs)))
                #print('Current regularization parameter: ' + str(self.weight_decay))

            # Testing for stopping criteria
            if self.stopping_criteria.test_for_stop(self):
                self.show_training_graphs(i)
                break

            # Updates regularization parameter
            if self.dynamic_regularization:
                self.update_regularization()

    def update_regularization(self):
        prob = 4
        amount = 0.01
        rate = 10
        diff = self.training_accs[-1] - self.validation_accs[-1]
        sigmoid_out = (np.exp(diff) - np.exp(-diff)) / (np.exp(diff) + np.exp(-diff))
        if random.randint(1, prob) == 1:
            self.weight_decay = max(0, self.weight_decay + (amount * sigmoid_out * rate))
            self.optimizer = optim.Adam(self.net.parameters(), lr=self.LEARNING_RATE, weight_decay=self.weight_decay)

    def show_training_graphs(self, i):
        # plotting results
        self.save_model(self.output_path)

        plt.subplot(1, 2, 1)
        plt.plot(self.training_iter, self.training_loss, c='b', label='Training error')
        plt.plot(self.validation_iter, self.validation_loss, c='r', label='Validation error')
        plt.title('Error for job: ' + str(self.name))
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(self.training_iter, self.training_accs, c='b', label='Training accuracy')
        plt.plot(self.validation_iter, self.validation_accs, c='r', label='Validation accuracy')
        plt.title('Accuracy for job: ' + str(self.name))
        plt.legend()

        print('Vaildation accuracy at iteration ' + str(i) + ' is ' + str(self.validation_accs[-1]))

        np.savetxt(self.output_path + 'model/training_loss.txt', self.training_loss)
        np.savetxt(self.output_path + 'model/training_accs.txt', self.training_accs)
        np.savetxt(self.output_path + 'model/validation_loss.txt', self.validation_loss)
        np.savetxt(self.output_path + 'model/validation_accs.txt', self.validation_accs)
        plt.savefig(self.output_path + 'model/results_plot')
        self.save_state(i)

        plt.show()
        self.net.train()

    def output(self):
        if self.save_model_:
            self.save_model(output_path=self.output_path + '/')
        if self.save_filters_:
            self.net.output_conv_parameters(self.output_path)
        if self.get_f1_score_:
            self.annotaed_dataset.get_f1_score(self.net, self.annotaed_dataset.validation_ind[0], output_path=self.output_path + '/')
        if self.save_test_output_:
            self.test_dataset.predict_both_binary_and_overview_image(self.net, output_path=self.output_path + self.name + '/')
            #self.test_dataset.predict_binary_masks(self.net, output_path=self.output_path + self.name + '/')
            #self.test_dataset.predict_overview_image(self.net, output_path=self.output_path + self.name + '/', show_results=False)
        if self.save_validation_output_:
            self.annotaed_dataset.predict_binary_masks(self.net, self.annotaed_dataset.validation_ind[0], output_path=self.output_path + self.name + '/')
            self.annotaed_dataset.predict_overview_image(self.net, self.annotaed_dataset.validation_ind[0], output_path=self.output_path + self.name + '/')
            self.annotaed_dataset.create_confusion_matrix(self.annotaed_dataset.validation_ind[0])
            self.annotaed_dataset.get_confusion_matrix()

    def show_augmented_images(self, batch, mask=True, save=False):
        #augmented_original_images = self.annotaed_dataset.get_original_input(batch)
        for (b, target_image) in zip(batch, self.targets):
            input_image = self.annotaed_dataset.images[b]
            if mask is True:
                binary_image = data_util.create_binary_output(target_image)
                if self.data_augmentation:
                    augmented_image = data_util.create_overview_image(input_image.augmented_non_normalized_image, binary_image)
                else:
                    augmented_image = data_util.create_overview_image(input_image, binary_image).astype(np.uint8)
            else:
                augmented_image = input_image.augmented_non_normalized_image
            if save and b == 5:
                plt.imsave('Temp_augmented_image.png', augmented_image.astype(np.uint8))
            plt.imshow(augmented_image.astype(np.uint8))
            plt.show()

    def run(self, train=True, save_model=True, save_test_output=False, save_validation_output=True, save_filters=False,
            get_f1_score=True):
        if not os.path.isdir(self.output_path):
            os.mkdir(self.output_path)

        self.get_f1_score_ = get_f1_score
        self.train_ = train
        self.save_model_ = save_model
        self.save_test_output_ = save_test_output
        self.save_validation_output_ = save_validation_output
        self.save_filters_ = save_filters

        if self.train_:
            #if not os.path.isdir(self.output_path + 'model'):
            #    os.mkdir(self.output_path + 'model')
            self.train()
        self.output()


class Load_job(Job):
    def __init__(self, input_path, output_path, SAVE_OUTPUT, name):
        self.input_path = input_path
        super().__init__(output_path, SAVE_OUTPUT, name)

    def load_model(self):
        self.net = Corn_segmentation.Net(load_text=self.load_text["Net"])
        self.set_parameters_from_text()
        if self.GPU_ENABLED:
            self.net = self.net.cuda()

    def save_model(self, output_path):
        super().save_model(output_path)
        with open(output_path + 'model/' + self.name + '.txt', 'w') as file:
            file.write(json.dumps(self.load_text))

    def set_parameters_from_text(self):
        exec(self.load_text["Parameters"])

class Define_job(Job):
    def __init__(self, output_path, SAVE_OUTPUT, name):
        super().__init__(output_path, SAVE_OUTPUT, name)

    def load_model(self):
        print('Loading default model in class')
        self.net = Corn_segmentation.Net()
        if self.GPU_ENABLED:
            self.net = self.net.cuda()

class Load_job_torch(Load_job):
    def __init__(self, input_path, output_path, SAVE_OUTPUT, name):
        self.name = name
        super().__init__(input_path, output_path, SAVE_OUTPUT, name)

    def load_model(self):
        self.load_text = eval(open(self.input_path + self.name, 'r').read().replace('\n',''))
        super().load_model()
        if self.GPU_ENABLED:
            self.net.load_state_dict(torch.load(self.input_path + 'model'))
        else:
            self.net.load_state_dict(torch.load(self.input_path + 'model', map_location=torch.device('cpu')))
        self.net.eval()

class Load_job_text(Load_job):
    def __init__(self, input_path, output_path, SAVE_OUTPUT, name):
        self.name = name
        super().__init__(input_path, output_path, SAVE_OUTPUT, name)

    def load_model(self):
        self.load_text = eval(open(self.input_path, 'r').read().replace('\n',''))
        super().load_model()