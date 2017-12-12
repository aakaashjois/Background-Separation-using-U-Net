import numpy as np
from sklearn.cluster import KMeans
from sklearn.utils import shuffle
from keras.preprocessing import image
from keras.preprocessing.image import load_img, img_to_array


def __flip__(x):
    """
    Flips image around vertical axis.
    
    Args:
        x: Array. The input image.
    
    Returns:
        The flipped image.
    """
    return np.flip(x, axis = 1)

def __rotate__(x, 
               theta):
    """
    Rotates the image.
    
    Args:
        x: Array. The input image.
        theta: Float. The rotation angle.
    
    Returns:
        The rotated image.
    """
    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
                                [np.sin(theta), np.cos(theta), 0],
                                [0, 0, 1]])
    transform_matrix = image.transform_matrix_offset_center(rotation_matrix, x.shape[0], x.shape[1])
    return image.apply_transform(x, transform_matrix, 2, 'nearest', 0.0)

def __shift__(x, 
              wshift,
              hshift):
    """
    Shift the image.
    
    Args:
        x: Array. The input image.
        wshift: Float. The amount of horizontal shift.
        hshift: Float. The amount of vertical shift.
    
    Returns:
        The shifted image.
    """
    tx = hshift * x.shape[0]
    ty = wshift * x.shape[1]
    translation_matrix = np.array([[1, 0, tx],
                                   [0, 1, ty],
                                   [0, 0, 1]])
    return image.apply_transform(x, translation_matrix, 2, 'nearest', 0.0)

def __zoom__(x,
             zx, 
             zy):
    """
    Zoom into the image.
    
    Args:
        x: Array. The input image.
        zx: Float. The amount of zoom on the horizontal axis.
        zy: Float. The amount of zoom on the vertical axis.
        
    Returns:
        The zoomed image.
    """
    zoom_matrix = np.array([[zx, 0, 0],
                            [0, zy, 0],
                            [0, 0, 1]])
    transform_matrix = image.transform_matrix_offset_center(zoom_matrix, x.shape[0], x.shape[1])
    return image.apply_transform(x, transform_matrix, 2, 'nearest', 0.0)

def __shear__(x, 
              shear,
              rotate_dir):
    """
    Shear the image.
    
    Args:
        x: Array. The input image.
        shear: Float. The amount of shear.
        rotate_dir: Integer. The direction of rotation. Takes value of either -1 or 1.
    
    Returns:
        The sheared image.
    """
    shear_matrix = np.array([[1, rotate_dir * np.sin(shear), 0],
                             [0, np.cos(shear), 0],
                             [0, 0, 1]])
    transform_matrix = image.transform_matrix_offset_center(shear_matrix, x.shape[0], x.shape[1])
    return image.apply_transform(x, transform_matrix, 2, 'nearest', 0.0)

def __random_flip__(img,
                    mask, 
                    u):
    """
    Perform random flip on images.
    
    Args:
        img: Array. The input image.
        mask: Array. The input mask.
        u: Float. The probability of application of this augmentation.
    
    Returns:
        A tuple of flipped image and mask.
    """
    if np.random.random() < u:
        img = __flip__(img, 1)
        mask = __flip__(mask, 1)
    return img, mask

def __random_rotate__(img,
                      mask,
                      rotate_limit, 
                      u):
    """
    Perform random rotation on images.
    
    Args:
        img: Array. The input image.
        mask: Array. The input mask.
        rotate_limit: Tuple (min limit, max limit). The mimimum and maximum angle of rotation allowed in degrees.
        u: Float. The probability of application of this augmentation.
    
    Returns:
        A tuple of rotated image and mask.
    """
    if np.random.random() < u:
        theta = np.pi / 180 * np.random.uniform(rotate_limit[0], rotate_limit[1])
        img = __rotate__(img, theta)
        mask = __rotate__(mask, theta)
    return img, mask

def __random_shift__(img,
                     mask,
                     w_limit,
                     h_limit,
                     u):
    """
    Perform random pixel shift on the images.
    
    Args:
        img: Array. The input image.
        mask: Array. The input mask.
        w_limit: Tuple (min limit, max limit). The minimum and maximum percentage of horizontal shift.
        h_limit: Tuple (min limit, max limit). The minimum and maximum percentage of vertical shift.
        u: Float. The probability of application of this augmentation.
        
    Returns:
        A tuple of pixel shifted image and mask.
    """
    if np.random.random() < u:
        wshift = np.random.uniform(w_limit[0], w_limit[1])
        hshift = np.random.uniform(h_limit[0], h_limit[1])
        img = __shift__(img, wshift, hshift)
        mask = __shift__(mask, wshift, hshift)
    return img, mask

def __random_zoom__(img, 
                    mask, 
                    zoom_range,
                    u):
    """
    Perform random zoom on the images.
    
    Args:
        img: Array. The input image.
        mask: Array. The input mask.
        zoom_range: Tuple (min limit, max limit). The minimum and maximum limit of the zoom range.
        u: Float. The probability of application of this augmentation.
        
    Returns:
        A tuple of zoomed image and mask.
    """
    if np.random.random() < u:
        zx, zy = np.random.uniform(zoom_range[0], zoom_range[1], 2)
        img = __zoom__(img, zx, zy)
        mask = __zoom__(mask, zx, zy)
    return img, mask

def __random_shear__(img, 
                     mask, 
                     intensity_range, 
                     random_shear,
                     u):
    """
    Perform random zoom on the images.
    
    Args:
        img: Array. The input image.
        mask: Array. The input mask.
        intensity_range: Tuple (min limit, max limit). The minimum and maximum limit of shear intensity.
        random_shear: Boolean. Whether to apply random sheer rotation.
        u: Float. The probability of application of this augmentation.
        
    Returns:
        A tuple of sheared image and mask.
    """
    shear_rot = 1
    if random_shear:
        if np.random.uniform(0, 1) > 0.5:
            shear_rot = -1
    if np.random.random() < u:
        sh = np.random.uniform(- intensity_range[0], intensity_range[1])
        img = __shear__(img, sh, rotate_dir = shear_rot)
        mask = __shear__(mask, sh, rotate_dir = shear_rot)
    return img, mask

def __color_quantize__(img,
                       mask,
                       target_colors):
    """
    Perform K-Means clustering to cluster colors on the input image.
    
    Args:
        img: Array. The input image.
        mask: Array. The input mask.
        target_colors: Integer. The number of clusters of colors.
        
    Returns:
        A tuple of color quantized image and mask.
    """
    img = img_to_array(img)
    image_array = np.reshape(img, (img.shape[0] * img.shape[1], img.shape[2]))
    image_array_sample = shuffle(image_array, random_state=0)[:1000]
    kmeans = KMeans(n_clusters=target_colors, random_state=0).fit(image_array_sample)
    labels = kmeans.predict(image_array)
    image = np.zeros((img.shape[0], img.shape[1], kmeans.cluster_centers_.shape[1]))
    label_idx = 0
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            image[i][j] = kmeans.cluster_centers_[labels[label_idx]]
            label_idx += 1
    return image, mask

def __letter_box__(img,
                   mask):
    """
    Performs letter boxing on the images.
    
    Args:
        img: Array. The input image.
        mask: Array. The input mask.
        
    Returns:
        A tuple of letter boxed image and mask.
    """
    if img.shape[1] == img.shape[0]:
        return img, mask
    else:
        diff1 = int(np.ceil((img.shape[1] - img.shape[0]) / 2))
        diff2 = int(np.floor((img.shape[1] - img.shape[0]) / 2))
        zero_img1 = np.zeros((diff1, img.shape[1], img.shape[2]))
        zero_img2 = np.zeros((diff2, img.shape[1], img.shape[2]))
        zero_mask1 = np.zeros((diff1, mask.shape[1], mask.shape[2]))
        zero_mask2 = np.zeros((diff2, mask.shape[1], mask.shape[2]))
        img_tmp = np.concatenate((zero_img1, img), axis = 0)
        new_img = np.concatenate((img_tmp, zero_img2), axis = 0)
        mask_tmp = np.concatenate((zero_mask1, mask), axis = 0)
        new_mask = np.concatenate((mask_tmp, zero_mask2), axis = 0)
        return new_img, new_mask
    
def random_augmentation(img, 
                        mask,
                        flip_chance = 0.1, 
                        rotate_chance = 0.1,
                        rotate_limit = (-15, 15), 
                        shift_chance = 0.1, 
                        shift_limit_w = (-0.5, 0.5), 
                        shift_limit_h = (-0.5, 0.5),
                        zoom_chance = 0.1, 
                        zoom_range = (0.8, 1), 
                        shear_chance = 0.1, 
                        shear_range = (-0.3, 0.3),
                        random_shear = True,
                        color_quantize = True,
                        target_colors = 8):
    """
    Perform random augmentations on the input images.
    
    Args:
        img: Array. The input image.
        mask: Array. The input mask.
        flip_chance: Float. The probability of application of flip augmentation.
        rotate_chance: Float. The probability of application of rotate augmentation.
        rotate_limit: Tuple (min limit, max limit). The mimimum and maximum angle of rotation allowed.
        shift_chance: Float. The probability of application of shift augmentation.
        shift_limit_w: Tuple (min limit, max limit). The minimum and maximum percentage of horizontal shift.
        shift_limit_h: Tuple (min limit, max limit). The minimum and maximum percentage of vertical shift.
        zoom_chance: Float. The probability of application of zoom augmentation.
        zoom_range: Tuple (min limit, max limit). The minimum and maximum limit of the zoom range.
        shear_chance: Float. The probability of application of sheer augmentation.
        shear_range: Tuple (min limit, max limit). The minimum and maximum limit of shear intensity.
        random_shear: Boolean. Whether to apply random sheer rotation.
        color_quantize: Boolean. Whether to apply color quantization.
        target_colors: Integer. The number of clusters of colors.
    
    Returns:
        A tuple of randomly augmented image and mask.
    """
    new_img = np.copy(img)
    new_mask = np.copy(mask)
    
    if(color_quantize):
        new_img, new_mask = __color_quantize__(new_img, 
                                               new_mask, 
                                               target_colors = target_colors)
    if(flip_chance > 0):
        new_img, new_mask = __random_flip__(new_img, 
                                            new_mask,
                                            u = flip_chance)
    if(rotate_chance > 0):
        new_img, new_mask = __random_rotate__(new_img,
                                              new_mask,
                                              rotate_limit = rotate_limit,
                                              u = rotate_chance)
    if(shift_chance > 0):
        new_img, new_mask = __random_shift__(new_img,
                                             new_mask,
                                             w_limit = shift_limit_w,
                                             h_limit = shift_limit_h,
                                             u = shift_chance)
    if(zoom_chance > 0):
        new_img, new_mask = __random_zoom__(new_img, 
                                            new_mask,
                                            zoom_range = zoom_range, 
                                            u = zoom_chance)
    if(shear_chance > 0):
        new_img, new_mask = __random_shear__(new_img, 
                                             new_mask, 
                                             intensity_range = shear_range, 
                                             u = shear_chance,
                                             random_shear = random_shear)
    new_img, new_mask = __letter_box__(img, mask)
    return new_img, new_mask
