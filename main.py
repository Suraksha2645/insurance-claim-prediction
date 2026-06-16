from preprocessing import get_data_overview, load_and_clean_data
from model import train_model
from analysis import evaluate_model
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix


def _prompt_int(label: str, minimum: int = 0) -> int:
    while True:
        raw_value = input(f"{label}: ").strip()
        try:
            value = int(raw_value)
            if value < minimum:
                print(f"Please enter a value greater than or equal to {minimum}.")
                continue
            return value
        except ValueError:
            print("Please enter a whole number.")


def _prompt_float(label: str, minimum: float = 0.0) -> float:
    while True:
        raw_value = input(f"{label}: ").strip()
        try:
            value = float(raw_value)
            if value < minimum:
                print(f"Please enter a value greater than or equal to {minimum}.")
                continue
            return value
        except ValueError:
            print("Please enter a valid number.")


def _prompt_choice(label: str, options: dict[str, int]) -> int:
    options_text = ", ".join(options.keys())
    while True:
        raw_value = input(f"{label} ({options_text}): ").strip().lower()
        if raw_value in options:
            return options[raw_value]
        print(f"Please choose one of: {options_text}")


def _collect_customer_input(default_region: int) -> pd.DataFrame:
    print("Enter customer details for prediction")
    print("------------------------------------")

    sex_options = {"female": 0, "male": 1}
    smoker_options = {"no": 0, "yes": 1}
    age = _prompt_int("Age", minimum=0)
    height_cm = _prompt_float("Height (cm)", minimum=1)
    weight_kg = _prompt_float("Weight (kg)", minimum=1)
    bmi = weight_kg / ((height_cm / 100) ** 2)
    print(f"Calculated BMI: {bmi:.2f}")

    children = 0
    if age >= 13:
        children = _prompt_int("Number of children", minimum=0)

    customer_data = {
        "age": age,
        "bmi": bmi,
        "charges": _prompt_float("Charges", minimum=0),
        "children": children,
        # Region input is intentionally skipped; use the dataset's most common region code.
        "region": default_region,
        "sex": _prompt_choice("Sex", sex_options),
        "smoker": _prompt_choice("Smoker", smoker_options),
    }

    return pd.DataFrame([customer_data])


def run_prediction_console(trained_model, default_region: int, accuracy: float, matrix) -> None:
    prediction_ran = False
    while True:
        should_predict = input("Do you want to predict for a new customer? (yes/no): ").strip().lower()
        if should_predict not in {"yes", "no"}:
            print("Please enter yes or no.")
            continue
        if should_predict == "no":
            break

        customer_features = _collect_customer_input(default_region=default_region)
        prediction_ran = True
        prediction = int(trained_model.predict(customer_features)[0])
        probability = float(trained_model.predict_proba(customer_features)[0][1])
        no_claim_probability = 1 - probability

        label = "Claim" if prediction == 1 else "No Claim"
        print("\nPrediction Result")
        print("-----------------")
        print(f"Predicted Class: {label}")
        print(f"Claim Probability: {probability * 100:.1f}%")
        print("\nModel Accuracy and Confusion Matrix")
        print("-----------------------------------")
        print(f"Accuracy: {accuracy * 100:.1f}%")
        print("Confusion Matrix:")
        print(matrix)
        print(
            f"Interpretation: {probability * 100:.1f}% chance of Claim and "
            f"{no_claim_probability * 100:.1f}% chance of No Claim."
        )
        print(f"Model Decision: likely {label}.")
        print("Important: This is only a model prediction, not an official insurance approval.")
        print("Final claim decisions are made by the insurance company after document review.")
        print()

    if not prediction_ran:
        print("No prediction was run in this session.")


def main() -> None:
    """Run the full training and evaluation workflow."""
    data = load_and_clean_data()
    print(get_data_overview(data))
    print()
    trained_model, features_test, target_test = train_model(data)
    default_region = int(data["region"].mode().iat[0])
    test_predictions = trained_model.predict(features_test)
    accuracy = accuracy_score(target_test, test_predictions)
    matrix = confusion_matrix(target_test, test_predictions)
    evaluate_model(trained_model, features_test, target_test)
    print()
    run_prediction_console(
        trained_model,
        default_region=default_region,
        accuracy=accuracy,
        matrix=matrix,
    )


if __name__ == "__main__":
    main()