import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def evaluate_model(model, X_test, y_test) -> None:
    """Print common classification metrics for the trained model."""
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    matrix = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    report_table = pd.DataFrame(report).transpose().round(4)

    true_negative = int(matrix[0, 0])
    false_positive = int(matrix[0, 1])
    false_negative = int(matrix[1, 0])
    true_positive = int(matrix[1, 1])
    total_cases = int(matrix.sum())
    correct_predictions = true_negative + true_positive

    claim_row = report_table.loc["1"] if "1" in report_table.index else None
    no_claim_row = report_table.loc["0"] if "0" in report_table.index else None

    print("MODEL RESULTS (EASY TO READ)")
    print("============================")
    print(f"Overall Accuracy: {accuracy * 100:.1f}%")
    print(f"Correct Predictions: {correct_predictions} out of {total_cases}")

    print("\nTECHNICAL METRICS")
    print("=================")
    print(f"Accuracy: {accuracy:.4f}")
    print("Confusion Matrix:")
    print(matrix)

    print("\nHow the model performed:")
    print(f"- Correctly predicted 'No Claim': {true_negative}")
    print(f"- Correctly predicted 'Claim': {true_positive}")
    print(f"- Wrongly predicted 'Claim' when there was no claim: {false_positive}")
    print(f"- Wrongly predicted 'No Claim' when there was a claim: {false_negative}")

    print("\nQuality scores by group:")
    if no_claim_row is not None:
        print(
            f"- No Claim: precision {no_claim_row['precision'] * 100:.1f}%, "
            f"recall {no_claim_row['recall'] * 100:.1f}%"
        )
    if claim_row is not None:
        print(
            f"- Claim: precision {claim_row['precision'] * 100:.1f}%, "
            f"recall {claim_row['recall'] * 100:.1f}%"
        )
