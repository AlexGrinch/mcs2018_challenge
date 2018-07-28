from PIL import Image
import numpy as np
from skimage.measure import compare_ssim as ssim

def center_crop(img, out_size=224):
    w, h = img.size
    i = int(round((h - out_size) / 2.))
    j = int(round((w - out_size) / 2.))
    return img.crop((i, j, i+out_size, j+out_size))

def resize(img, out_size=112):
    return img.resize((out_size, out_size), resample=Image.BILINEAR)

def normalize(img):
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    arr = np.array(img, dtype=np.float32) / 255.
    return (arr - mean) / std

def normalize2(arr):
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    return (arr - mean) / std

def preprocess_img(img):
    img = center_crop(img)
    img = resize(img)
    arr = normalize(img)
    arr = arr.transpose([2, 0, 1])
    return arr[None].copy()

def img_to_crop(img):
    img = center_crop(img)
    img = resize(img)
    return img

def crop_to_tensor(img):
    arr = normalize(img)
    arr = arr.transpose([2, 0, 1])
    return arr[None].copy()

def crop_to_tensor2(arr):
    arr = np.array(arr, dtype=np.float32)
    arr = normalize2(arr)
    arr = arr.transpose([2, 0, 1])
    return arr[None].copy()

def denormalize(img):
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    old_img = mean + std * img
    return old_img

def calc_ssim(arr1, arr2): 
    img1 = np.array(arr1*255, dtype=np.uint8)
    img2 = np.array(arr2*255, dtype=np.uint8)
    return ssim(img1, img2, multichannel=True)

def tf2bb(x):
    x = x[None, :, :, :]
    x = np.transpose(x, [0, 3, 1, 2])
    x = x.astype(np.float32)
    return x.copy()