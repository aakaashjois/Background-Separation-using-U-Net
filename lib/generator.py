import numpy as np
from keras.preprocessing.image import img_to_array, load_img
from os.path import join
from lib import augmentations

def image_generator(images_dir, masks_dir, images, masks, batch_size, img_dim = None, num_colors = 256):
    
    while True:
        random_indices = np.random.choice(len(images), batch_size)
        i = []
        m = []
        
        for index in random_indices:
            
            img = load_img(join(images_dir, images[index]), target_size = img_dim)
            img_array = img_to_array(img) / 255
            
            mask = load_img(join(masks_dir, masks[index]), target_size = img_dim, grayscale = True)
            mask_array = img_to_array(mask) / 255
            
            img_array, mask_array = augmentations.random_augmentation(img_array,
                                                                      mask_array,
                                                                      flip_chance = 0.1, 
                                                                      rotate_chance = 0.1,
                                                                      shift_chance = 0.1,
                                                                      zoom_chance = 0.1,
                                                                      shear_chance = 0.1,
                                                                      color_quantize = True,
                                                                      letter_box = True)
            i.append(img_array)
            m.append(mask_array[:, :, 0])
            
        yield np.array(i), np.array(m).reshape(-1, img_dim[1], img_dim[1], 1)

def image_generator_no_aug(images_dir, masks_dir, images, masks, batch_size, img_dim = None, num_colors = 256):
    
    while True:
        random_indices = np.random.choice(len(images), batch_size)
        i = []
        m = []
        
        for index in random_indices:
            
            img = load_img(join(images_dir, images[index]), target_size = img_dim)
            img_array = img_to_array(img) / 255
            
            mask = load_img(join(masks_dir, masks[index]), target_size = img_dim, grayscale = True)
            mask_array = img_to_array(mask) / 255
            
            img_array, mask_array = augmentations.random_augmentation(img_array,
                                                                      mask_array,
                                                                      flip_chance = 0.0, 
                                                                      rotate_chance = 0.0,
                                                                      shift_chance = 0.0,
                                                                      zoom_chance = 0.0,
                                                                      shear_chance = 0.0,
                                                                      color_quantize = False,
                                                                      letter_box = True)
            i.append(img_array)
            m.append(mask_array[:, :, 0])
            
        yield np.array(i), np.array(m).reshape(-1, img_dim[1], img_dim[1], 1)