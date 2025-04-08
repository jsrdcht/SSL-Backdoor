from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageColor
import os
import cv2
import re
import sys
import glob
import errno
import random
import numpy as np

def get_trigger(trigger_size=40, trigger_path=None, colorful_trigger=True):
    # load trigger
    if colorful_trigger:
        trigger = Image.open(trigger_path).convert('RGB')
        trigger = trigger.resize((trigger_size, trigger_size))
    else:
        trigger = Image.new("RGB", (trigger_size, trigger_size), ImageColor.getrgb("white"))
    return trigger




def binary_mask_to_box(binary_mask):
    binary_mask = np.array(binary_mask, np.uint8)
    contours,hierarchy = cv2.findContours(
        binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    areas = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        areas.append(area)
    idx = areas.index(np.max(areas))
    x, y, w, h = cv2.boundingRect(contours[idx])
    bounding_box = [x, y, x+w, y+h]
    return bounding_box

def get_foreground(reference_dir, num_references, max_size, type):
    img_idx = random.choice(range(1, 1+num_references))
    image_path = os.path.join(reference_dir, f'{img_idx}/img.png')
    mask_path = os.path.join(reference_dir, f'{img_idx}/label.png')
    image_np = np.asarray(Image.open(image_path).convert('RGB'))
    mask_np = np.asarray(Image.open(mask_path).convert('RGB'))
    mask_np = (mask_np[..., 0] == 128) ##### [:,0]==128 represents the object mask
    
    # crop masked region
    bbx = binary_mask_to_box(mask_np)
    object_image = image_np[bbx[1]:bbx[3],bbx[0]:bbx[2]]
    object_image = Image.fromarray(object_image)
    object_mask = mask_np[bbx[1]:bbx[3],bbx[0]:bbx[2]]
    object_mask = Image.fromarray(object_mask)

    # resize -> avoid poisoned image being too large
    w, h = object_image.size
    if type=='horizontal':
        o_w = min(w, int(max_size/2))
        o_h = int((o_w/w) * h)
    elif type=='vertical':
        o_h = min(h, int(max_size/2))
        o_w = int((o_h/h) * w)
    object_image = object_image.resize((o_w, o_h))
    object_mask = object_mask.resize((o_w, o_h))
    return object_image, object_mask

def concat(support_reference_image_path, reference_image_path, max_size):
    ### horizontally concat two images
    # get support reference image
    support_reference_image = Image.open(support_reference_image_path)
    width, height = support_reference_image.size
    n_w = min(width, int(max_size/2))
    n_h = int((n_w/width) * height)
    support_reference_image = support_reference_image.resize((n_w, n_h))
    width, height = support_reference_image.size

    # get reference image
    reference_image = Image.open(reference_image_path)
    reference_image = reference_image.resize((width, height))

    img_new = Image.new("RGB", (width*2, height), "white")
    if random.random()<0.5:
        img_new.paste(support_reference_image, (0, 0))
        img_new.paste(reference_image, (width, 0))
    else:
        img_new.paste(reference_image, (0, 0))
        img_new.paste(support_reference_image, (width, 0))
    return img_new


def get_random_reference_image(reference_dir, num_references):
    img_idx = random.choice(range(1, 1+num_references))
    image_path = os.path.join(reference_dir, f'{img_idx}/img.png')
    return image_path

def get_random_support_reference_image(reference_dir):
    support_dir = os.path.join(reference_dir, 'support-images')
    image_path = os.path.join(support_dir, random.choice(os.listdir(support_dir)))
    return image_path
