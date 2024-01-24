from sklearn.metrics import precision_recall_curve, auc
import numpy as np

def calculate_auprc(true_labels, predicted_labels):
    precision, recall, thresholds = precision_recall_curve(true_labels, predicted_labels)
    return auc(recall, precision)

def evaluate_predictions(true_labels, predicted_labels):
    f1_max = 0
    precision_list = []
    recall_list = []

    # Filter out samples with non-zero sum in true labels
    filtered_true, filtered_pred = zip(*[(t, p) for t, p in zip(true_labels, predicted_labels) if np.sum(t) > 0])

    # Iterate over a range of thresholds
    for threshold in np.linspace(0.01, 1, 100):
        # Apply threshold to predictions
        binary_predictions = np.array(filtered_pred) > threshold

        # Calculate true positives, false positives, and false negatives
        tp = np.sum(binary_predictions & filtered_true, axis=1)
        fp = np.sum(binary_predictions, axis=1)
        fn = np.sum(filtered_true, axis=1)

        # Calculate precision and recall for each sample
        precision = np.divide(tp, tp + fp, out=np.zeros_like(tp, dtype=float), where=(tp+fp)!=0)
        recall = tp / fn

        # Calculate average precision and recall
        avg_precision = np.mean(precision)
        avg_recall = np.mean(recall)

        # Calculate F1 score and update max F1
        if avg_precision + avg_recall > 0:
            f1_score = 2 * avg_precision * avg_recall / (avg_precision + avg_recall)
            f1_max = max(f1_max, f1_score)

    return f1_max, calculate_auprc(np.hstack(filtered_true), np.hstack(filtered_pred))

if __name__ == '__main__':
    true_values = np.load('Data/CAFA3/test/MFO/MFO_true.npy')
    predicted_values = np.load('Data/CAFA3/test/MFO/MFO_pred_epoch19.npy')
    print(evaluate_predictions(true_values, predicted_values))
