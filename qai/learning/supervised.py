from .base import LearningTechnique


class Supervised(LearningTechnique):
    name = "supervised"
    on_step_signature = ("input", "expected_output", "actual_output")

    def requires(self) -> str:
        return "a labeled dataset (X, y)"


class Unsupervised(LearningTechnique):
    name = "unsupervised"
    on_step_signature = ("input", "discovered_pattern")

    def requires(self) -> str:
        return "raw, unlabeled data (X only)"


class MultiOutputClassifier:
    """Multi-target classification wrapper."""
    def __init__(self, estimator: Any):
        self.estimator = estimator
        self.estimators_ = []

    def train(self, X: Any, Y: Any):
        X_arr = np.array(X)
        Y_arr = np.array(Y)
        self.estimators_ = []
        for col in range(Y_arr.shape[1]):
            from qai.core import build
            est = build(type=getattr(self.estimator, "type", "classifier"))
            est.train(X_arr, Y_arr[:, col])
            self.estimators_.append(est)
        return self

    def predict(self, X: Any) -> np.ndarray:
        X_arr = np.array(X)
        preds = [est.predict(X_arr) for est in self.estimators_]
        return np.column_stack(preds)


class MultiOutputRegressor:
    """Multi-target regression wrapper."""
    def __init__(self, estimator: Any):
        self.estimator = estimator
        self.estimators_ = []

    def train(self, X: Any, Y: Any):
        X_arr = np.array(X)
        Y_arr = np.array(Y)
        self.estimators_ = []
        for col in range(Y_arr.shape[1]):
            from qai.core import build
            est = build(type=getattr(self.estimator, "type", "regression"))
            est.train(X_arr, Y_arr[:, col])
            self.estimators_.append(est)
        return self

    def predict(self, X: Any) -> np.ndarray:
        X_arr = np.array(X)
        preds = [est.predict(X_arr) for est in self.estimators_]
        return np.column_stack(preds)


class ClassifierChain:
    """Chain of binary classifiers learning target dependencies."""
    def __init__(self, estimator: Any):
        self.estimator = estimator
        self.estimators_ = []

    def train(self, X: Any, Y: Any):
        X_arr = np.array(X)
        Y_arr = np.array(Y)
        self.estimators_ = []
        X_aug = X_arr.copy()

        for col in range(Y_arr.shape[1]):
            from qai.core import build
            est = build(type=getattr(self.estimator, "type", "classifier"))
            est.train(X_aug, Y_arr[:, col])
            self.estimators_.append(est)
            X_aug = np.column_stack([X_aug, Y_arr[:, col]])
        return self

    def predict(self, X: Any) -> np.ndarray:
        X_arr = np.array(X)
        X_aug = X_arr.copy()
        preds = []
        for est in self.estimators_:
            p = est.predict(X_aug)
            preds.append(p)
            X_aug = np.column_stack([X_aug, p])
        return np.column_stack(preds)


class StackingClassifier:
    """Stacking ensemble for classification."""
    def __init__(self, estimators: List[Any], final_estimator: Any):
        self.estimators = estimators
        self.final_estimator = final_estimator

    def train(self, X: Any, y: Any):
        X_arr = np.array(X)
        y_arr = np.array(y)
        meta_features = []
        for est in self.estimators:
            est.train(X_arr, y_arr)
            p = est.predict(X_arr)
            meta_features.append(p)
        meta_X = np.column_stack(meta_features)
        self.final_estimator.train(meta_X, y_arr)
        return self

    def predict(self, X: Any) -> np.ndarray:
        X_arr = np.array(X)
        meta_features = [est.predict(X_arr) for est in self.estimators]
        meta_X = np.column_stack(meta_features)
        return self.final_estimator.predict(meta_X)


class StackingRegressor:
    """Stacking ensemble for regression."""
    def __init__(self, estimators: List[Any], final_estimator: Any):
        self.estimators = estimators
        self.final_estimator = final_estimator

    def train(self, X: Any, y: Any):
        X_arr = np.array(X)
        y_arr = np.array(y)
        meta_features = []
        for est in self.estimators:
            est.train(X_arr, y_arr)
            p = est.predict(X_arr)
            meta_features.append(p)
        meta_X = np.column_stack(meta_features)
        self.final_estimator.train(meta_X, y_arr)
        return self

    def predict(self, X: Any) -> np.ndarray:
        X_arr = np.array(X)
        meta_features = [est.predict(X_arr) for est in self.estimators]
        meta_X = np.column_stack(meta_features)
        return self.final_estimator.predict(meta_X)
