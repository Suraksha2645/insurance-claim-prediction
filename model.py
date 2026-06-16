from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from preprocessing import get_feature_groups


def train_model(df):
    """Train a logistic regression model and return model plus test split."""
    X = df.drop("insuranceclaim", axis=1)
    y = df["insuranceclaim"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    numeric_columns, categorical_columns = get_feature_groups(df)

    transformers = []

    if categorical_columns:
        transformers.append(
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                categorical_columns,
            )
        )

    if numeric_columns:
        transformers.append(
            (
                "numeric",
                StandardScaler(),
                numeric_columns,
            )
        )

    preprocessor = ColumnTransformer(
        transformers=transformers,
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(max_iter=1000)),
        ]
    )

    model.fit(X_train, y_train)

    return model, X_test, y_test