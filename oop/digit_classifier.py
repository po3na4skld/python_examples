import numpy as np

from oop.digit_classification import (
    CNNDigitClassifier,
    RandomForestDigitClassifier,
    RandomDigitClassifier
)


class DigitClassifier:
    model_map = {
        "cnn": CNNDigitClassifier,
        "rf": RandomForestDigitClassifier,
        "rand": RandomDigitClassifier
    }

    def __init__(self, algorithm_name: str):
        assert algorithm_name in self.model_map, f"Algorithm name: {algorithm_name} not in {self.model_map.keys()}"
        self.algorithm_name = algorithm_name
        self.clf = self.model_map[algorithm_name]()

    def predict(self, inputs: np.ndarray) -> int:
        return self.clf.predict(inputs)


if __name__ == '__main__':

    # Test only allowed algorithms
    try:
        digit_classifier = DigitClassifier('non_exist')
    except AssertionError as e:
        pass
    else:
        raise AssertionError("Only allowed algorithms test fail")

    inputs = np.random.random((28, 28, 1))

    # Test CNN
    digit_classifier = DigitClassifier('cnn')
    pred = digit_classifier.predict(inputs)
    assert pred in range(0, 10)

    # Test RFC
    digit_classifier = DigitClassifier('rf')
    pred = digit_classifier.predict(inputs)
    assert pred in range(0, 10)

    # Test Random
    digit_classifier = DigitClassifier('rand')
    pred = digit_classifier.predict(inputs)
    assert pred in range(0, 10)


