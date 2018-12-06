import numpy as np
from sklearn.metrics import precision_recall_curve, auc
import matplotlib.pyplot as plt
import os

def get_all_groundtruth_and_predictions(predictions):
    gt = []
    pred = []
    for prediction in predictions:
        gt += list(prediction["gt_mask"].reshape(360*480))
        pred += list(prediction["pred_mask"].reshape(360*480))
    return gt, pred


if __name__ == "__main__":

    y_real = []
    y_proba = []
    for k in range(1,5):
        pred_filepath = os.path.join("/hdd", "EyeZ", "Rot", "Segmentation", "6.SegNet", "6classes_4cv_scleraOnly", str(k), "Models", "inference_prob", "predictions.npy")
        predictions = np.load(pred_filepath, encoding='latin1')
        gt, pred = get_all_groundtruth_and_predictions(predictions)

        #precision, recall, _ = precision_recall_curve(gt, pred)
        #lab = 'Fold %d AUC=%.4f' % (k, auc(recall, precision))
        #plt.step(recall, precision, label=lab)
        y_real.append(gt)
        y_proba.append(pred)

    y_real = np.concatenate(y_real)
    y_proba = np.concatenate(y_proba)
    precision, recall, _ = precision_recall_curve(y_real, y_proba)
    lab = 'Overall AUC=%.4f' % (auc(recall, precision))
    plt.step(recall, precision, label=lab, lw=2, color='black')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend(loc='lower left', fontsize='small')

    plt.tight_layout()
    plt.savefig("precision_recall_4cv.eps", format='eps', dpi=1000)
    plt.show()
