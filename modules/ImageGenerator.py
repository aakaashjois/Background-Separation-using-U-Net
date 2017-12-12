import numpy as np
from os import listdir, getcwd, chdir
from os.path import join
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import img_to_array, load_img
from modules import ImageAugmentation

def __image_generator__(images_dir,
                        masks_dir, 
                        images, 
                        masks,
                        batch_size,
                        num_colors,
                        augment,
                        img_dim = None):
    
    while True:
        random_indices = np.random.choice(len(images), batch_size)
        i = []
        m = []
        
        for index in random_indices:
            
            img = load_img(join(images_dir, images[index]), target_size = img_dim)
            img_array = img_to_array(img) / 255
            
            mask = load_img(join(masks_dir, masks[index]), target_size = img_dim, grayscale = True)
            mask_array = img_to_array(mask) / 255
            
            if(augment):
                img_array, mask_array = ImageAugmentation.random_augmentation(img_array,
                                                                          mask_array)
            else:
                img_array, mask_array = ImageAugmentation.random_augmentation(img_array,
                                                                          mask_array,
                                                                          flip_chance = 0.0, 
                                                                          rotate_chance = 0.0,
                                                                          shift_chance = 0.0,
                                                                          zoom_chance = 0.0,
                                                                          shear_chance = 0.0,
                                                                          color_quantize = False)
            i.append(img_array)
            m.append(mask_array[:, :, 0])
            
        yield np.array(i), np.array(m).reshape(-1, img_dim[1], img_dim[1], 1)

def get_generators(img_width,
                   aspect_ratio = 1280 / 1918,
                   batch_size = 2,
                   num_colors = 8,
                   augment = True):

    data_dir = join(getcwd(), 'data')
    train_dir = join(data_dir, 'train')
    mask_dir = join(data_dir, 'train_masks')
    train_image_list = sorted(listdir(train_dir))
    mask_image_list = sorted(listdir(mask_dir))
    
    train_images, test_images, train_masks, test_masks = train_test_split(train_image_list, 
                                                                          mask_image_list, 
                                                                          test_size = 0.10000)
    train_images, validation_images, train_masks, validation_masks = train_test_split(train_images,
                                                                                      train_masks,
                                                                                      test_size=0.11111)
    
    train_generator = __image_generator__(train_dir,
                                          mask_dir, 
                                          train_images, 
                                          train_masks, 
                                          batch_size = batch_size, 
                                          img_dim = [int(aspect_ratio * img_width), img_width],
                                          num_colors = num_colors,
                                          augment = augment)

    validation_generator = __image_generator__(train_dir,
                                               mask_dir, 
                                               train_images, 
                                               train_masks, 
                                               batch_size = batch_size, 
                                               img_dim = [int(aspect_ratio * img_width), img_width],
                                               num_colors = num_colors, 
                                               augment = False)
    
    test_generator = __image_generator__(train_dir,
                                         mask_dir, 
                                         test_images, 
                                         test_masks, 
                                         batch_size = batch_size, 
                                         img_dim = [int(aspect_ratio * img_width), img_width],
                                         num_colors = num_colors, 
                                         augment = False)
    return train_generator, validation_generator, test_generator