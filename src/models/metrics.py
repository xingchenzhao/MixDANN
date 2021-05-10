import torch
import SimpleITK as sitk
import scipy
import scipy.ndimage
import scipy.spatial
import os
import numpy as np
from pathlib import Path

# Code Adapated from https://github.com/FourierX9/wmh_ibbmTum/blob/73cf37c77e85ebe2b34c8120d86624591c2b6b50/evaluation.py
# and https://github.com/hjkuijf/MRBrainS18/blob/master/evaluation.py


def postprocess_critic():
    return None


class Critic:
    def __init__(self):
        pass

    def list_template(self):
        return {
            'ACC': [],
            'PPV': [],
            'TPR': [],
            'FPR': [],
            'DSC': [],
            'BS': []
        }

    def __call__(self, scores, y, train=False):
        if not train:
            scores_li, y_li = metrics_preprocess(scores, y)
            dsc_li = getDSC_Li(scores_li, y_li)
            avd_li = getAVD_Li(scores_li, y_li)
            h95_li = getHausdorff_Li(scores_li, y_li)
            recall_li, f1_li = getLesionDetection_Li(scores_li, y_li)

        bs = ((y - scores.sigmoid())**2).mean()
        yhat = (scores.sigmoid() > .5).long()

        # -> we cannot assume y, yhat are on cpu to use sklearn
        # -> below is a torch approach that assumes 2 classes
        # - > implementation adapted from
        # gist.github.com/the-bass/cae9f3976866776dea17a5049013258d

        confusion_vector = yhat.float() / y.float()
        # Element-wise division of the 2 tensors returns a new
        # tensor which holds a unique value for each case:
        #   1     :prediction and truth are 1 (True Positive)
        #   inf   :prediction is 1 and truth is 0 (False Positive)
        #   nan   :prediction and truth are 0 (True Negative)
        #   0     :prediction is 0 and truth is 1 (False Negative)

        TP = torch.sum(confusion_vector == 1).item()
        FP = torch.sum(torch.isinf(confusion_vector)).item()
        TN = torch.sum(torch.isnan(confusion_vector)).item()
        FN = torch.sum(confusion_vector == 0).item()
        # adds some easy checks to avoid 0 denoms
        metrics_result = {
            'ACC': (TP + TN) / (TP + FP + FN + TN),
            'PPV': TP / (TP + FP) if TP != 0 else 0,
            'TPR': TP / (TP + FN) if TP != 0 else 0,
            'FPR': FP / (FP + TN) if FP != 0 else 0,
            'DSC': 2 * TP / (2 * TP + FP + FN) if TP != 0 else 0,
            'BS': bs.item(),
            'DSC_li': 0,
            'AVD_li': 0,
            'H95_li': 0,
            'Lesion_Recall_li': 0,
            'F1_li': 0
        }
        if not train:
            metrics_result['DSC_li'] = dsc_li
            metrics_result['AVD_li'] = avd_li
            metrics_result['H95_li'] = h95_li
            metrics_result['Lesion_Recall_li'] = recall_li
            metrics_result['F1_li'] = f1_li

        return metrics_result


# Li el al's Metrics ----------------------------------------------------------------


def metrics_preprocess(pred, label):
    random_file_id = np.random.rand()
    pred = pred.permute(0, 2, 3, 1).cpu().detach().numpy()
    label = label.permute(0, 2, 3, 1).cpu().detach().numpy()
    pred_img = sitk.GetImageFromArray(pred, isVector=True)
    pred_save_dir = 'temp_save_dir/' + str(random_file_id) + '_pred.nii.gz'
    Path('temp_save_dir/').mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(pred_img, pred_save_dir)
    label_img = sitk.GetImageFromArray(label, isVector=True)
    label_save_dir = 'temp_save_dir/' + str(random_file_id) + '_label.nii.gz'
    sitk.WriteImage(label_img, label_save_dir)
    pred_img, label_img = getImages(pred_save_dir, label_save_dir)

    return pred_img, label_img


def getImages(pred_save_dir, label_save_dir):
    """Return the pred and test images, thresholded and non-WMH masked"""
    pred_img = sitk.ReadImage(pred_save_dir)
    label_img = sitk.ReadImage(label_save_dir)
    os.remove(pred_save_dir)
    os.remove(label_save_dir)

    assert pred_img.GetSize() == label_img.GetSize()
    print(pred_img.GetSize(), label_img.GetSize())

    # Get meta data from the label image, needed for  some sitk methods that check this
    pred_img.CopyInformation(label_img)

    # Remove non-WMH from the label image and pred image, since we don't evaluate on that
    maskedLabelImage = sitk.BinaryThreshold(label_img, 0.5, 1.5, 1,
                                            0)  # WMH == 1
    nonWMHImage = sitk.BinaryThreshold(label_img, 1.5, 2.5, 0,
                                       1)  # non-WMH == 2
    maskedPredImage = sitk.Mask(pred_img, nonWMHImage)

    # convert to binary mask
    if 'integer' in maskedLabelImage.GetPixelIDTypeAsString():
        bPredImage = sitk.BinaryThreshold(maskedPredImage, 1, 1000, 1, 0)
    else:
        bPredImage = sitk.BinaryThreshold(maskedPredImage, 0.5, 1000, 1, 0)

    return bPredImage, maskedLabelImage


def getDSC_Li(pred_img, label_img):
    """Compute the Dice Similarity Coefficent"""
    predArray = sitk.GetArrayFromImage(pred_img).flatten()
    labelArray = sitk.GetArrayFromImage(label_img).flatten()

    # similarity = 1.0 - dissimilarity
    return 1.0 - scipy.spatial.distance.dice(labelArray, predArray)


def getHausdorff_Li(pred_img, label_img):
    """Compute the Hausdorff distance"""
    # return float('nan')

    # Hausdorff distance is only defined when something is detected
    predStat = sitk.StatisticsImageFilter()
    predStat.Execute(pred_img)
    if predStat.GetSum() == 0:
        return float('nan')
    # edge detection is done by ORIGINAL - ERODED, keeping the outer boundaries of lesions. Erosion is performed in 2D
    eLabelImage = sitk.BinaryErode(label_img, (1, 1, 0))
    ePredImage = sitk.BinaryErode(pred_img, (1, 1, 0))

    hLabelImage = sitk.Subtract(label_img, eLabelImage)
    hPredImage = sitk.Subtract(pred_img, ePredImage)

    hLabelArray = sitk.GetArrayFromImage(hLabelImage)
    hPredArray = sitk.GetArrayFromImage(hPredImage)

    # Convert voxel location to world coordinates. Use the coordinate system of the label image
    # np.nonzero = elements of the boundary in numpy order(zyx)
    # np.flipud = elements in xyz order
    # np.transpose = create tuples (x,y,z)
    # testImage.TransformIndexToPhysicalPoint converts (xyz) to world coordinates (in mm)

    # labelCoordinates = np.apply_along_axis(label_img.TransformIndexToPhysicalPoint, 1, np.transpose(
    #     np.flipud(np.nonzero(hLabelArray))).astype(int))
    # predCoordinates = np.apply_along_axis(pred_img.TransformIndexToPhysicalPoint, 1, np.transpose(
    #     np.flipud(np.nonzero(hPredArray))).astype(int))

    labelCoordinates = [
        label_img.TransformIndexToPhysicalPoint(x.tolist())
        for x in np.transpose(np.flipud(np.nonzero(hLabelArray)))
    ]
    predCoordinates = [
        label_img.TransformIndexToPhysicalPoint(x.tolist())
        for x in np.transpose(np.flipud(np.nonzero(hPredArray)))
    ]

    # Use a kd-tree for fast spatial search

    def getDistancesFromAtoB(a, b):
        kdTree = scipy.spatial.KDTree(a, leafsize=100)
        return kdTree.query(b, k=1, eps=0, p=2)[0]

    dLabelToPred = getDistancesFromAtoB(labelCoordinates, predCoordinates)
    dPredToLabel = getDistancesFromAtoB(predCoordinates, labelCoordinates)

    return max(np.percentile(dLabelToPred, 95),
               np.percentile(dPredToLabel, 95))


def getLesionDetection_Li(pred_img, label_img):
    """Lesion detection metrics, both recall and F1."""
    # Connected components will give the background label 0 , so subtract 1 from all results
    ccFilter = sitk.ConnectedComponentImageFilter()
    ccFilter.SetFullyConnected(True)

    # Connected components on the label image, to determine the number of true WMH.
    # And to get the overlap between detected voxels and true WMH
    ccLabel = ccFilter.Execute(label_img)
    lPred = sitk.Multiply(ccLabel, sitk.Cast(pred_img, sitk.sitkUInt32))

    ccLabelArray = sitk.GetArrayFromImage(ccLabel)
    lPredArray = sitk.GetArrayFromImage(lPred)

    # recall = (number of detected WMH) / (number of true WMH)
    nWMH = len(np.unique(ccLabelArray)) - 1
    if nWMH == 0:
        recall = 1.0
    else:
        recall = float(len(np.unique(lPredArray)) - 1) / nWMH

    # Connected components of results, to determine number of detected lesions
    ccPred = ccFilter.Execute(pred_img)
    ccPredArray = sitk.GetArrayFromImage(ccPred)

    # precision = (number of detected WMH) / (number of all detections)
    nDetections = len(np.unique(ccPredArray)) - 1
    if nDetections == 0:
        precision = 1.0
    else:
        precision = float(len(np.unique(lPredArray)) - 1) / nDetections

    if precision + recall == 0.0:
        f1 = 0.0
    else:
        f1 = 2.0 * (precision * recall) / (precision + recall)
    return recall, f1


def getAVD_Li(pred_img, label_img):
    """Volume statistics"""
    # Compute statistics of both images
    labelStat = sitk.StatisticsImageFilter()
    predStat = sitk.StatisticsImageFilter()

    # if labelStat.GetSum() == 0:
    #     return float('nan')

    labelStat.Execute(label_img)
    predStat.Execute(pred_img)

    return float(abs(labelStat.GetSum() - predStat.GetSum())) / float(
        labelStat.GetSum()) * 100
