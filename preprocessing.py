# standarization
import pandas as pd

TARGET_COLUMN = "insuranceclaim"
KNOWN_CATEGORICAL_COLUMNS = ("sex", "smoker", "region")


def load_and_clean_data() -> pd.DataFrame:
    """Load the dataset and apply basic cleaning with consistent column structure."""
    data = pd.read_csv("insurance2.csv")
    data.columns = [column.strip().lower().replace(" ", "_") for column in data.columns]
    data = data.dropna().copy()

    if TARGET_COLUMN not in data.columns:
        raise ValueError("Expected target column 'insuranceclaim' was not found.")

    feature_columns = sorted(column for column in data.columns if column != TARGET_COLUMN)
    ordered_columns = [*feature_columns, TARGET_COLUMN]
    data = data[ordered_columns]

    categorical_columns = [
        column
        for column in feature_columns
        if column in KNOWN_CATEGORICAL_COLUMNS or data[column].dtype == "object"
    ]
    for column in categorical_columns:
        data[column] = data[column].astype("category")

    return data


def get_feature_groups(data: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Split feature columns into numeric and categorical groups for modeling."""
    feature_columns = [column for column in data.columns if column != TARGET_COLUMN]

    categorical_columns = [
        column
        for column in feature_columns
        if column in KNOWN_CATEGORICAL_COLUMNS or data[column].dtype.name in {"object", "category"}
    ]
    numeric_columns = [column for column in feature_columns if column not in categorical_columns]

    return numeric_columns, categorical_columns


def get_data_overview(data: pd.DataFrame, preview_rows: int = 5) -> str:
    """Build a plain-language dataset summary for console output."""
    display_data = data.copy()

    display_mappings = {
        "sex": {0: "Female", 1: "Male"},
        "smoker": {0: "No", 1: "Yes"},
        "region": {
            0: "Northeast",
            1: "Northwest",
            2: "Southeast",
            3: "Southwest",
        },
        "insuranceclaim": {0: "No Claim", 1: "Claim"},
    }

    for column, mapping in display_mappings.items():
        if column in display_data.columns:
            display_data[column] = display_data[column].map(mapping).fillna(display_data[column])

    preview = display_data.head(preview_rows).copy().reset_index(drop=True)
    if "charges" in preview.columns:
        preview["charges"] = preview["charges"].apply(lambda value: f"${value:,.2f}")

    column_meanings = {
        "age": "Age of customer",
        "bmi": "Body mass index",
        "charges": "Medical charges",
        "children": "Number of children",
        "region": "Customer region",
        "sex": "Customer sex",
        "smoker": "Smoking status",
        "insuranceclaim": "Claim status",
    }

    target_lines = []
    if "insuranceclaim" in display_data.columns:
        target_counts = display_data["insuranceclaim"].value_counts()
        total = len(display_data)
        target_lines.append("Claim Breakdown:")
        for label, count in target_counts.items():
            percentage = (count / total) * 100
            target_lines.append(f"  {label}: {count} ({percentage:.1f}%)")

    lines = [
        "DATASET SUMMARY (PLAIN LANGUAGE)",
        "===============================",
        f"Total Records: {len(display_data)}",
        f"Total Columns: {display_data.shape[1]}",
    ]

    lines.extend(target_lines)

    lines.append("")
    lines.append("What each column means:")
    for column in display_data.columns:
        meaning = column_meanings.get(column, column)
        lines.append(f"- {column}: {meaning}")

    lines.append("")
    lines.append(f"Sample Records ({preview_rows} examples):")
    for row_index, row in preview.iterrows():
        lines.append(f"Example {row_index + 1}:")
        for column in preview.columns:
            lines.append(f"  {column}: {row[column]}")
        lines.append("")

    if lines[-1] == "":
        lines.pop()

    return "\n".join(lines)