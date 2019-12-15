# Standard packages
import matplotlib.image as img
import glob
import os
import sys
import numpy as np
import random
import torch
import imageio
import matplotlib.pyplot as plt
from scipy import misc
from torch.autograd import Variable
from sklearn.metrics import f1_score

import Augmentation

class Dataset():
    def __init__(self):
        self.GPU_ENABLED = torch.cuda.is_available()
        self.images = []

    def load_images(self, path):
        self.mean = 0
        self.std = 0
        files_png = glob.glob(path + '*.png')
        files_JPG = glob.glob(path + '*.JPG')
        files = files_png + files_JPG
        return files

    def get_input(self, batch, augment_data=False):
        fixed_seed = Augmentation.Fixed_seed()
        self.augment_data = augment_data
        self.images[batch[0]].set_augmentation(self.augment_data, fixed_seed)
        self.crop_size_x, self.crop_size_y = self.images[batch[0]].augmentation.get_crop_dim()
        batch_images = np.zeros((len(batch), self.crop_size_x, self.crop_size_y, self.images[batch[0]].channels))
        for i, b in enumerate(batch):
            self.images[b].set_augmentation(self.augment_data, fixed_seed)
            batch_images[i, :, :, :] = self.images[b].perform_augmentation(self.images[b].image, 'train')
        batch_images = self.normalize_images(batch_images, batch)
        if self.GPU_ENABLED:
            return Variable(torch.from_numpy(batch_images).cuda()).float()
        else:
            return Variable(torch.from_numpy(batch_images)).float()

    def predict_binary_masks(self, net, b):
        """
        Outputs two binary images of predicted crops and weeds (x)
        Outputs a combined greyscale image. Black: background, Grey: weed, White: crops (v)
        """

        weed_and_crops = np.zeros((self.images[b].x, self.images[b].y))
        weed_and_crops[self.images[b].binary_output == 0.5] = 0.5
        weed_and_crops[self.images[b].binary_output == 1] = 1
        self.images[b].weed_and_crops = weed_and_crops
        imageio.imwrite(self.output_path + str(self.images[b].name) + 'crops_and_weed.png', weed_and_crops)

    def predict_overview_image(self, net, batch, output_path, show_results=False):
        """
        Outputs overview output of input image
        """
        images_to_evaluate = self.get_input(batch)
        outputs = net(images_to_evaluate)

        for i, b in enumerate(batch):
            self.images[b].add_net_output(outputs[i, :, :, :])
            if show_results:
                self.images[b].show_overview_output()
            imageio.imwrite(output_path + str(self.images[b].name) + '_overview.png', self.images[b].overview_output)

    def get_avg_mean_std(self, augmented_images, batch):
        mean = 0
        std = 0
        for i, b in enumerate(batch):
            mean += np.mean(augmented_images[i, :, :, :], axis=(0, 1), keepdims=True)
            std += np.std(augmented_images[i, :, :, :], axis=(0, 1), keepdims=True)
        mean = mean / len(batch)
        std = std / len(batch)
        return mean, std

    def normalize_images(self, augmented_images, batch):
        mean, std = self.get_avg_mean_std(augmented_images, batch)
        for i, b in enumerate(batch):
            augmented_images[i, :, :, :] = standardize_image(augmented_images[i, :, :, :], mean, std)
        return augmented_images

class Annotated_data(Dataset):
    def __init__(self, classes):
        super().__init__()
        self.class_dict = classes
        self.number_of_classes = len(self.class_dict)

    def get_original_input(self, batch):
        augmented_images = np.zeros((len(batch), self.crop_size_x, self.crop_size_y, self.number_of_classes))
        for i, b in enumerate(batch):
            augmented_images[i, :, :, :] = self.images[b].perform_augmentation(self.images[b].original_image, mode='preview')
        return augmented_images

    def load_images(self, path):
        files = super().load_images(path)
        image_files = []
        mask_files = []
        for file in files:
            if file.split('.')[0].split('_')[-1] == 'mask':
                mask_files.append(file)
            else:
                image_files.append(file)

        for image_file in image_files:
            for mask_file in mask_files:
                file1 = image_file.split('/')[-1].split('.')[0]
                file2 = '_'.join(mask_file.split('/')[-1].split('_')[:-1])
                if file1 == file2:
                    self.images.append(Annotated_image(img.imread(image_file),
                                                       img.imread(mask_file),
                                                       image_file.split('.')[0]))

        self.N = len(self.images)


    def batch_generator(self, batch_size, iterations, val_size, data_augmentation):
        self.data_augmentation = data_augmentation
        self.validation_ind = []
        self.batch_ind = []
        for i in range(iterations):
            # picking batch indicies
            self.batch_ind.append(np.random.randint(low=0, high=self.N-val_size, size=batch_size))

            # picking validation index
            self.validation_ind.append(np.arange(self.N-val_size, self.N))

    def get_targets(self, batch):
        targets = np.zeros((len(batch), self.number_of_classes, self.crop_size_x, self.crop_size_y))
        for i, b in enumerate(batch):
            cropped_image = self.images[b].perform_augmentation(self.images[b].mask, 'binary')
            targets[i, 0, cropped_image == 0] = 1
            targets[i, 1, cropped_image == 0.5] = 1
            targets[i, 2, cropped_image == 1] = 1
        if self.GPU_ENABLED:
            return Variable(torch.from_numpy(targets).cuda()).float()
        else:
            return Variable(torch.from_numpy(targets)).float()

    def predict_binary_masks(self, net, batch=None, output_path=None):
        """
        Outputs two binary images of predicted crops and weeds
        """
        self.output_path = output_path
        if not os.path.isdir(self.output_path + 'annotated_data'):
            os.mkdir(self.output_path + 'annotated_data')
        if not os.path.isdir(self.output_path + 'annotated_data/binary_masks'):
            os.mkdir(self.output_path + 'annotated_data/binary_masks')
        self.output_path += 'annotated_data/binary_masks/'

        for b in batch:
            image_to_evaluate = self.get_input([b])
            self.images[b].add_net_output(net(image_to_evaluate))
            super().predict_binary_masks(net, b)

    def create_confusion_matrix(self, batch):
        self.combined_confusion_matrix = np.zeros((3, 3))
        for b in batch:
            self.images[b].create_confusion_matrix()
            self.combined_confusion_matrix = np.add(self.combined_confusion_matrix, self.images[b].confusion_matrix)

    def get_confusion_matrix(self):
        print(self.combined_confusion_matrix)


    def predict_overview_image(self, net, batch=None, output_path = None, show_results=False):
        if not os.path.isdir(output_path + 'annotated_data'):
            os.mkdir(output_path + 'annotated_data')
        if not os.path.isdir(output_path + 'annotated_data/overview_images'):
            os.mkdir(output_path + 'annotated_data/overview_images')
        super().predict_overview_image(net, batch, output_path + 'annotated_data/overview_images/', show_results)

    def get_f1_score(self, net, batch, output_path):
        self.output_path = output_path
        f1_scores = []
        for b in batch:
            image_to_evaluate = self.get_input([b])
            self.images[b].add_net_output(net(image_to_evaluate))
            output_path = output_path
            array_predicted = np.zeros((self.images[b].x, self.images[b].y), dtype=np.uint8)
            array_target = np.zeros((self.images[b].x, self.images[b].y), dtype=np.uint8)
            array_predicted[self.images[b].binary_output == 0] = 0
            array_predicted[self.images[b].binary_output == 0.5] = 1
            array_predicted[self.images[b].binary_output == 1] = 2
            array_target[self.images[b].mask == 0] = 0
            array_target[self.images[b].mask == 0.5] = 1
            array_target[self.images[b].mask == 1] = 2

            f1_scores.append(f1_score(array_predicted.reshape(-1), array_target.reshape(-1), average='macro'))
        self.f1_score = np.mean(f1_scores)
        file = open(self.output_path + 'model/f1_score.txt', 'w')
        file.write('F1 score: ' + str(self.f1_score))
        file.close()


class Test_data(Dataset):
    def __init__(self):
        super().__init__()

    def load_images(self, path):
        self.files = super().load_images(path)
        self.images = [None]
        #for file in self.files:
        #    self.images.append(Test_image(img.imread(file), file.split('/')[-1].split('.')[0]))
        #    self.mean += self.images[-1].mean
        #    self.std += self.images[-1].std
        #self.N = len(self.images)
        #self.mean = self.mean / self.N
        #self.std = self.std / self.N
        #self.normalize_images()
        #self.perform_sub_crop()

    def reload_current_image(self, i):
        self.images[0] = Test_image(img.imread(self.files[i]), self.files[i].split('/')[-1].split('.')[0])
        self.mean = self.images[0].mean
        self.std = self.images[0].std
        self.normalize_images()
        self.perform_sub_crop()

    def predict_binary_masks(self, net, batch=None, output_path=None):
        self.output_path = output_path
        if batch == None:
            batch = np.arange(len(self.files))
        if not os.path.isdir(self.output_path):
            os.mkdir(self.output_path)
        if not os.path.isdir(self.output_path + 'test_data'):
            os.mkdir(self.output_path + 'test_data')
        if not os.path.isdir(self.output_path + 'test_data/binary_masks'):
            os.mkdir(self.output_path + 'test_data/binary_masks')
        self.output_path += 'test_data/binary_masks/'

        """
        Outputs two binary images of predicted crops and weeds
        """
        for b in batch:
            self.reload_current_image(b)
            sys.stdout.write('\r' + 'Currently outputting binary mask of ' + str(self.images[0].name) + ' (' + str(b+1)
                             + '/' + str(len(self.files)) + ')')
            sys.stdout.flush()
            self.predict_sub_crop(net, 0)
            self.combine_sub_crops()
            self.images[0].binary_output = create_binary_output(self.images[0].output)
            super().predict_binary_masks(net, 0)


    def predict_overview_image(self, net, batch=None, output_path = None, show_results=False):
        self.output_path = output_path
        if batch == None:
            batch = np.arange(len(self.files))
        if not os.path.isdir(self.output_path):
            os.mkdir(self.output_path)
        if not os.path.isdir(self.output_path + 'test_data'):
            os.mkdir(self.output_path + 'test_data')
        if not os.path.isdir(self.output_path + 'test_data/overview_images'):
            os.mkdir(self.output_path + 'test_data/overview_images')
        self.output_path += 'test_data/overview_images/'

        """
        Outputs overview output of input image
        """
        for i, b in enumerate(batch):
            if len(batch) != 1: # if multiple images needs to be estimated at once
                self.reload_current_image(b)
                self.predict_sub_crop(net, 0)
                self.combine_sub_crops()
            self.images[0].overview_output = create_overview_image(self.images[0].original_image, self.images[0].binary_output)
            if show_results:
                self.images[0].show_overview_output()
            imageio.imwrite(self.output_path + str(self.images[0].name) + '_overview.png', self.images[0].overview_output)

    def predict_both_binary_and_overview_image(self, net, output_path):
        batch = np.arange(len(self.files))
        for b in batch:
            self.predict_binary_masks(net, batch=[b], output_path=output_path)
            self.predict_overview_image(net, batch=[b], output_path=output_path)

    def predict_sub_crop(self, net, b):
        for subcrop in self.images[b].sub_crops:
            subcrop = subcrop
            batch_image = np.zeros((1, subcrop.x, subcrop.y, subcrop.channels))
            batch_image[0, :, :, :] = subcrop.image
            if self.GPU_ENABLED:
                image_to_evaluate = Variable(torch.from_numpy(batch_image).cuda()).float()
            else:
                image_to_evaluate = Variable(torch.from_numpy(batch_image)).float()
            subcrop.add_net_output(net(image_to_evaluate))

    def perform_sub_crop(self):
        for image in self.images:
            image.perform_sub_crop()

    def combine_sub_crops(self):
        for image in self.images:
            image.combine_sub_crops()

class Image():
    def __init__(self, image, name):
        self.image = image
        self.original_image = self.image.copy()
        self.name = name.split('/')[-1]
        self.x = len(self.image[:, 0, 0])
        self.y = len(self.image[0, :, 0])
        self.pixels = self.x * self.y
        self.channels = len(self.image[0, 0, :])
        self.mean = np.mean(self.image, axis=(0, 1), keepdims=True)
        self.std = np.std(self.image, axis=(0, 1), keepdims=True)
        self.augmentation_seed = Augmentation.Seed()

    def normalize(self, mean, std):
        self.image = (self.image - mean) / std

    def show(self):
        pass

    def add_net_output(self, output):
        if output.ndim == 4:
            output = output.detach().permute(0, 2, 3, 1).cpu().numpy()[0, :, :, :]
        elif output.ndim == 3:
            output = output.detach().permute(1, 2, 0).cpu().numpy()
        self.prob_output = output
        self.binary_output = create_binary_output(self.prob_output)
        self.overview_output = create_overview_image(self.original_image, self.binary_output)

    def set_augmentation(self, augment, fixed_seed):
        self.fixed_seed = fixed_seed
        if augment:
            self.augmentation = Augmentation.Image_augmentation(self)
        else:
            self.augmentation = Augmentation.No_augmentation(self)

    def perform_augmentation(self, image, mode=None):
        aug_image = self.augmentation.perform_augmentation(image, mode)
        if mode != 'binary':
            self.augmented_non_normalized_image = aug_image
        return aug_image

    def show_prob_output(self, show_now=True):
        plt.imshow(self.prob_output)
        if show_now:
            plt.show()

    def show_overview_output(self, show_now=True):
        plt.imshow(self.overview_output)
        if show_now:
            plt.show()

class Annotated_image(Image):
    def __init__(self, image, mask, name):
        super().__init__(image, name)
        self.mask = np.round(mask, 1)

    def show_mask_output(self, show_now=True):
        f, axarr = plt.subplots(2, 1)
        axarr[0].imshow(self.binary_output, cmap='gray', vmin=0, vmax=1)
        axarr[1].imshow(self.mask, cmap='gray', vmin=0, vmax=1)
        if show_now:
            plt.show()

    def create_confusion_matrix(self):
        self.confusion_matrix = np.zeros((3, 3))
        for i, x in enumerate([0, 0.5, 1]):  # testing for predicted
            for j, y in enumerate([0, 0.5, 1]):  # testing for actual
                predicted_mask = np.zeros((self.x, self.y))
                actual_mask = np.zeros((self.x, self.y))
                predicted_mask[self.weed_and_crops == x] = True
                actual_mask[self.mask == y] = True
                self.confusion_matrix[i, j] = np.sum(np.logical_and(predicted_mask, actual_mask))

class Subcrop(Image):
    def __init__(self, image, name, lower_x, lower_y, upper_x, upper_y, goal_x, goal_y, pad_size, n_x, n_y):
        super().__init__(image, name)
        self.lower_x = lower_x
        self.lower_y = lower_y
        self.upper_x = upper_x - pad_size*(1)*2
        self.upper_y = upper_y - pad_size*(1)*2

        self.upper_y = self.upper_y

    def add_net_output(self, output):
        if output.ndim == 4:
            self.output = output.detach().permute(0, 2, 3, 1).cpu().numpy()[0, :, :, :]
        elif output.ndim == 3:
            self.output = output.detach().permute(1, 2, 0).cpu().numpy()

class Test_image(Image):
    def __init__(self, image, name):
        super().__init__(image, name)

    def add_mirror_pads(self):
        """
        x = rows
        y = columns
        """
        self.pad_image = np.zeros((self.x+(self.pad_size*2), self.y+(self.pad_size*2), self.channels))
        for i in range(3):
            self.pad_image[:, :, i] = np.lib.pad(self.image[:, :, i], (self.pad_size, self.pad_size), 'reflect')
        self.pad_x = len(self.pad_image[:, 0, 0])
        self.pad_y = len(self.pad_image[0, :, 0])

    def combine_sub_crops(self):
        self.output = np.zeros((self.x, self.y, self.channels))
        for subcrop in self.sub_crops:
            lower_x = subcrop.lower_x
            lower_y = subcrop.lower_y
            upper_x = subcrop.upper_x
            upper_y = subcrop.upper_y
            self.output[subcrop.lower_x:subcrop.upper_x, subcrop.lower_y:subcrop.upper_y, :] = \
            subcrop.output[self.pad_size:-self.pad_size, self.pad_size:-self.pad_size, :]

    def perform_sub_crop(self, pad_size=32):
        self.goal_x = 512 # desired output dimensions for image rows
        self.goal_y = 512 # desired output dimensions for image columns
        self.pad_size = pad_size
        self.add_mirror_pads()
        self.sub_crops = []

        lower_x = 0
        lower_y = 0
        upper_x = self.goal_x
        upper_y = self.goal_y

        self.n_x_images = 0  # number of cropped images in row direction

        while(upper_y != self.pad_y or upper_x != self.pad_x):

            self.n_y_images = 0  # number of cropped images in column direction

            # performing row crop
            while(upper_y != self.pad_y):

                if self.n_y_images != 0:
                    upper_y = min(self.pad_y, upper_y + self.goal_y - (self.pad_size*2))
                lower_y = upper_y - self.goal_y

                self.sub_crops.append(Subcrop(self.pad_image[lower_x:upper_x, lower_y:upper_y, :] ,'subcrop',
                                              lower_x, lower_y, upper_x, upper_y, self.goal_x, self.goal_y,
                                              self.pad_size, self.n_x_images, self.n_y_images))

                self.n_y_images += 1

            if upper_x != self.pad_x: # if we haven't reached the bottom of the image

                # preparing row and column index
                upper_y = self.goal_y
                lower_y = 0
                if self.n_x_images != 0:
                    upper_x = min(self.pad_x, upper_x + self.goal_x - (self.pad_size*2))
                lower_x = upper_x - self.goal_x

                self.sub_crops.append(Subcrop(self.pad_image[lower_x:upper_x, lower_y:upper_y, :] ,'subcrop',
                                              lower_x, lower_y, upper_x, upper_y, self.goal_x, self.goal_y,
                                              self.pad_size, self.n_x_images, self.n_y_images))
                self.n_x_images += 1

def create_overview_image(original_image, binary_image):
    overview_output = np.copy(original_image)
    overview_output[binary_image == 0, 0] = 1
    overview_output[binary_image == 0.5, 1] = 1
    overview_output[binary_image == 1, 2] = 1
    return overview_output

def create_binary_output(output):
    if output.shape[0] < output.shape[2]: # [channel, x, y]
        max_vals = np.argmax(output, axis=0)
        binary_output = np.zeros((len(output[0, :, 0]), len(output[0, 0, :])))
    else: # [x, y, channel]
        max_vals = np.argmax(output, axis=2)
        binary_output = np.zeros((len(output[:, 0, 0]), len(output[0, :, 0])))
    binary_output[max_vals == 0] = 0
    binary_output[max_vals == 1] = 0.5
    binary_output[max_vals == 2] = 1
    return binary_output

def standardize_image(image, mean, std):
    return (image - mean) / std