from PIL import Image
from torchvision import transforms
import torch
import numpy as np
import random
import copy
import SimpleITK as sitk
import scipy
from pathlib import Path
import os
from torchvision.transforms import ToPILImage, ToTensor
import torchvision.transforms.functional as augmentor
import scipy.ndimage
from torchvision.utils import save_image

rows_standard = 200
cols_standard = 200
thresh_FLAIR = 70
smooth = 1.


def eval_preprocess(img, label=False):
    num_selected_slice = np.shape(img)[0]
    img_rows_dataset = np.shape(img)[1]
    img_cols_dataset = np.shape(img)[2]
    if not label:
        brain_mask = np.ndarray(
            (num_selected_slice, img_rows_dataset, img_cols_dataset),
            dtype=np.float32)
        brain_mask[img >= thresh_FLAIR] = 1
        brain_mask[img < thresh_FLAIR] = 0
        for i in range(np.shape(img)[0]):
            brain_mask[i, :, :] = scipy.ndimage.morphology.binary_fill_holes(
                brain_mask[i, :, :])
        img -= np.mean(img[brain_mask == 1])
        img /= np.std(img[brain_mask == 1])

    if img_rows_dataset >= rows_standard and img_cols_dataset >= cols_standard:
        img_resize = img[:, (img_rows_dataset // 2 -
                             rows_standard // 2):(img_rows_dataset // 2 +
                                                  rows_standard // 2),
                         (img_cols_dataset // 2 -
                          cols_standard // 2):(img_cols_dataset // 2 +
                                               cols_standard // 2)]
    elif img_rows_dataset < rows_standard and img_cols_dataset < cols_standard:
        img_resize = np.ndarray(
            (num_selected_slice, rows_standard, cols_standard),
            dtype=np.float32)
        img_resize[...] = np.min(img)
        img_resize[:, (rows_standard // 2 -
                       img_rows_dataset // 2):(rows_standard // 2 +
                                               img_rows_dataset // 2),
                   (cols_standard // 2 - img_cols_dataset // 2):(
                       cols_standard // 2 +
                       img_cols_dataset // 2)] = img[:, :, :]
    elif img_rows_dataset < rows_standard and img_cols_dataset >= cols_standard:
        img_resize = np.ndarray(
            (num_selected_slice, rows_standard, cols_standard),
            dtype=np.float32)
        img_resize[...] = np.min(img)
        img_resize[:, (rows_standard // 2 - img_rows_dataset // 2):(
            rows_standard // 2 + img_rows_dataset // 2), :] = img[:, :, (
                img_cols_dataset // 2 -
                cols_standard // 2):(img_cols_dataset // 2 +
                                     cols_standard // 2)]
    elif img_rows_dataset >= rows_standard and img_cols_dataset < cols_standard:
        img_resize = np.ndarray(
            (num_selected_slice, rows_standard, cols_standard),
            dtype=np.float32)
        img_resize[...] = np.min(img)
        img_resize[:, :, (cols_standard // 2 - img_cols_dataset // 2):(
            cols_standard // 2 + img_cols_dataset // 2)] = img[:, (
                img_rows_dataset // 2 -
                rows_standard // 2):(img_rows_dataset // 2 +
                                     rows_standard // 2), :]
    return img_resize


def eval_postprocess(img, pred):
    num_selected_slice = np.shape(img)[0]
    img_rows_dataset = np.shape(img)[1]
    img_cols_dataset = np.shape(img)[2]
    original_pred = np.ndarray(np.shape(img), dtype=np.float32)
    if img_rows_dataset >= rows_standard and img_cols_dataset >= cols_standard:
        original_pred[:, (img_rows_dataset // 2 -
                          rows_standard // 2):(img_rows_dataset // 2 +
                                               rows_standard // 2),
                      (img_cols_dataset // 2 -
                       cols_standard // 2):(img_cols_dataset // 2 +
                                            cols_standard // 2)] = pred
    elif img_rows_dataset < rows_standard and img_cols_dataset < cols_standard:
        original_pred[:, :, :] = pred[:, (rows_standard // 2 -
                                          img_rows_dataset // 2):(
                                              rows_standard // 2 +
                                              img_rows_dataset // 2),
                                      (cols_standard // 2 -
                                       img_cols_dataset // 2):(
                                           cols_standard // 2 +
                                           img_cols_dataset // 2)]
    elif img_rows_dataset < rows_standard and img_cols_dataset >= cols_standard:
        original_pred[:, (img_rows_dataset // 2 - rows_standard // 2):(
            img_rows_dataset // 2 + rows_standard // 2), :] = pred[:, :, (
                cols_standard // 2 -
                img_cols_dataset // 2):(cols_standard // 2 +
                                        img_cols_dataset // 2)]
    elif img_rows_dataset >= rows_standard and img_cols_dataset < cols_standard:
        original_pred[:, (img_rows_dataset // 2 - rows_standard // 2):(
            img_rows_dataset // 2 + rows_standard // 2), :] = pred[:, :, (
                cols_standard // 2 -
                img_cols_dataset // 2):(cols_standard // 2 +
                                        img_cols_dataset // 2)]
    return original_pred


def getImages(testFilename, resultFilename):
    """Return the test and result images, thresholded and non-WMH masked."""
    # testImage = sitk.ReadImage(testFilename)
    # resultImage = sitk.ReadImage(resultFilename)
    testImage = testFilename
    resultImage = resultFilename
    # Check for equality
    assert testImage.GetSize() == resultImage.GetSize()

    # Get meta data from the test-image, needed for some sitk methods that check this
    resultImage.CopyInformation(testImage)

    # Remove non-WMH from the test and result images, since we don't evaluate on that
    maskedTestImage = sitk.BinaryThreshold(testImage, 0.5, 1.5, 1,
                                           0)  # WMH == 1
    nonWMHImage = sitk.BinaryThreshold(testImage, 1.5, 2.5, 0,
                                       1)  # non-WMH == 2
    maskedResultImage = sitk.Mask(resultImage, nonWMHImage)

    # Convert to binary mask
    if 'integer' in maskedResultImage.GetPixelIDTypeAsString():
        bResultImage = sitk.BinaryThreshold(maskedResultImage, 1, 1000, 1, 0)
    else:
        bResultImage = sitk.BinaryThreshold(maskedResultImage, 0.5, 1000, 1, 0)

    return maskedTestImage, bResultImage