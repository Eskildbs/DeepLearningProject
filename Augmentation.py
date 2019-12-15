import random
import numpy as np
import torchvision.transforms as tt
from collections import deque
import PIL.Image
import cv2

class Seed():
    def __init__(self):
        self.seeds = deque([])
        self.pointer = -1
        self.new_seed_flag = True

    def new_seed(self):
        if self.new_seed_flag is False:
            self.seeds = deque([])
            self.pointer = -1
        self.seeds.append(random.random())
        self.pointer += 1
        self.new_seed_flag = True
        self.saved_seeds = self.seeds.copy()
        return self.seeds[-1]

    def pop_seed(self):
        self.new_seed_flag = False
        return self.seeds.popleft()

    def get(self, mode):
        if mode == 'preview':
            self.pointer -= 1
            return self.saved_seeds[self.pointer]
        elif mode == 'train':
            return self.new_seed()
        elif mode == 'binary':
            return self.pop_seed()

class Fixed_seed():
    """
    Same seed for all image augmentation in same batch
    """
    def __init__(self):
        self.generate_seeds()

    def generate_seeds(self, n=10):
        self.seeds = []
        for i in range(n):
            self.seeds.append(random.random())

    def get_seed(self, i):
        return self.seeds[i]

class No_augmentation():
    def __init__(self, image):
        self.x = image.x
        self.y = image.y

    def perform_augmentation(self, image, mode):
        return image

    def get_crop_dim(self):
        return self.x, self.y

class Image_augmentation():
    def __init__(self, image):
        self.seed = image.augmentation_seed
        self.x = image.x
        self.y = image.y
        self.fixed_seed = image.fixed_seed
        self.generate_augmentation(image)

    def generate_augmentation(self, image):
        self.augmentations = []



        # Cropping
        self.augmentations.append(Crop(image.x, image.y, fixed_seed1=self.fixed_seed.get_seed(0),
                                       fixed_seed2=self.fixed_seed.get_seed(1), seed=self.seed))
        self.crop_x = self.augmentations[-1].crop_size_x
        self.crop_y = self.augmentations[-1].crop_size_y

        # Flipping
        #self.augmentations.append(Vertical_flip(seed=self.seed, prob=0.5))
        #self.augmentations.append(Horisontal_flip(seed=self.seed, prob=0.5))

        # Changing brightness
        #self.augmentations.append(Adjust_Brigthness(seed=self.seed))

        # Changing saturation
        #self.augmentations.append(Adjust_Saturation(seed=self.seed))



        # Combining set augmentations
        self.augmentations = Combined_augmentations(self.augmentations)

    def perform_augmentation(self, image, mode):
        self.mode = mode
        if self.mode == None:
            self.get_mode(image)
        return self.augmentations(image, self.mode)

    def get_mode(self, image, preview=False):
        """
        Returns either 'train' or 'test' depending on the image comming in.
        3 channel images are train images, thus 'train' is returned.
        1 channel images are binary image, thus 'test' is returned.
        'preview' is returned if augmentation is for visualization purpose,
        thus neither creating nor removing seeds.
        """
        if preview:
            self.mode = 'preview'
        elif len(image.shape) == 3:
            self.mode = 'train'
        else:
            self.mode = 'binary'

    def get_crop_dim(self):
        return self.crop_x, self.crop_y

class Combined_augmentations(object):
    def __init__(self, augmentations):
        self.augmentations = augmentations

    def __call__(self, image, mode):
        for augmentation in self.augmentations:
            image = augmentation.perform(image, mode)
        return image

class No_action(object):
    def __init__(self):
        pass

    def perform(self, image, mode):
        return image

class Vertical_flip(object):
    def __init__(self, seed, prob=0.5):
        self.seed = seed
        self.prob = prob

    def perform(self, image, mode):
        random.seed(self.seed.get(mode))
        if random.random() > self.prob:
            return image[::-1, :]
        else:
            return image

class Horisontal_flip(object):
    def __init__(self, seed, prob=0.5):
        self.seed = seed
        self.prob = prob

    def perform(self, image, mode):
        random.seed(self.seed.get(mode))
        if random.random() > self.prob:
            return image[:, ::-1]
        else:
            return image

class Adjust_Brigthness(object):
    def __init__(self, seed):
        self.seed = seed

    def perform(self, image, mode):
        """
        Changes brightness in image relative to each pixel +- 75% of distance to lowest or
        highest possible value.
        If current brightness is 200 [0; 255], then +-75% would be in range 200 +- (255-200)*value
        """
        random.seed(self.seed.get(mode))
        if mode == 'binary':
            return image

        # gets random number between -0.75 and 0.75 from normal distribution
        change_in_brightness = max(-1, min(1, random.normalvariate(0, 0.33)))
        change_in_brightness = -0.6
        hsv_image = cv2.cvtColor(image.astype(np.float32), cv2.COLOR_BGR2HSV)
        max_change_per_pixel = (np.full((len(image[:, 0, 0]), len(image[0, :, 0])), 255) - hsv_image[:, :, 2])
        min_change_per_pixel = hsv_image[:, :, 2]
        change_per_pixel = np.minimum(min_change_per_pixel, max_change_per_pixel) * change_in_brightness

        hsv_image[:, :, 2] = hsv_image[:, :, 2] + change_per_pixel

        image = np.round(cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)).astype(np.uint8)
        return image

class Adjust_Saturation(object):
    def __init__(self, seed):
        self.seed = seed

    def perform(self, image, mode):
        """
        Changes brightness in image relative to each pixel +- 75% of distance to lowest or
        highest possible value.
        If current brightness is 200 [0; 255], then +-75% would be in range 200 +- (255-200)*value
        """
        random.seed(self.seed.get(mode))
        if mode == 'binary':
            return image

        # gets random number between -0.5 and 0.5 from normal distribution
        change_in_saturation = max(-0.5, min(0.5, random.normalvariate(0, 0.167)))
        change_in_saturation = 0.35
        hsv_image = cv2.cvtColor(image.astype(np.float32), cv2.COLOR_BGR2HSV)
        max_change_per_pixel = (np.full((len(image[:, 0, 0]), len(image[0, :, 0])), 1) - hsv_image[:, :, 1])
        min_change_per_pixel = hsv_image[:, :, 1]
        change_per_pixel = np.minimum(min_change_per_pixel, max_change_per_pixel) * change_in_saturation

        hsv_image[:, :, 1] = hsv_image[:, :, 1] + change_per_pixel

        image = np.round(cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)).astype(np.uint8)
        return image



class Crop(object):
    def __init__(self, old_size_x, old_size_y, fixed_seed1, fixed_seed2, seed):
        self.old_size_x = old_size_x
        self.old_size_y = old_size_y
        self.size_list_x = [int(self.old_size_x/4), int(self.old_size_x/2), int(self.old_size_x*(3/4))]
        self.size_list_y = [int(self.old_size_y/4), int(self.old_size_y/2), int(self.old_size_y*(3/4))]
        self.size_list_x = [int(self.old_size_x*(3/4))]
        self.size_list_y = [int(self.old_size_y/2)]
        random.seed(fixed_seed1)
        self.crop_size_x = random.choice(self.size_list_x)
        random.seed(fixed_seed2)
        self.crop_size_y = random.choice(self.size_list_y)
        self.seed = seed

    def perform(self, image, mode):
        random.seed(self.seed.get(mode))
        self.i = random.randint(0, self.old_size_x - self.crop_size_x)
        random.seed(self.seed.get(mode))
        self.j = random.randint(0, self.old_size_y - self.crop_size_y)
        return_img = image[self.i:self.i+self.crop_size_x,
                     self.j:self.j+self.crop_size_y]
        return return_img