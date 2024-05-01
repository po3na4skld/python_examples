from abc import ABC

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Input
from sklearn.ensemble import RandomForestClassifier

from oop.random_model import RandomModel


class DigitClassificationInterface:
    input_shape = (28, 28, 1)

    def __init__(self):
        self._model = self._setup_model()

    def _validate_input(self, inputs: np.ndarray):
        assert inputs.shape == self.input_shape, f"inputs shape must be: {self.input_shape}"

    def _setup_model(self):
        raise NotImplementedError

    def _preprocess_inputs(self, inputs: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def _predict(self, inputs: np.ndarray) -> int:
        raise NotImplementedError

    def predict(self, inputs: np.ndarray) -> int:
        self._validate_input(inputs)
        preprocessed_input = self._preprocess_inputs(inputs)
        return self._predict(preprocessed_input)

    def train(self):
        raise NotImplementedError


class CNNDigitClassifier(DigitClassificationInterface, ABC):

    def _setup_model(self) -> Sequential:
        model = Sequential()
        model.add(Input(self.input_shape))
        model.add(Conv2D(32, (3, 3), activation='relu'))
        model.add(MaxPool2D((2, 2)))
        model.add(Flatten())
        model.add(Dense(100, activation='relu'))
        model.add(Dense(10, activation='softmax'))
        return model

    def _preprocess_inputs(self, inputs: np.ndarray) -> np.ndarray:
        return np.expand_dims(np.asarray(inputs) / 255., 0)

    def predict(self, inputs: np.ndarray) -> int:
        preprocessed_inputs = self._preprocess_inputs(inputs)
        probabilities = self._model.predict(preprocessed_inputs, verbose=0)[0]
        return np.argmax(probabilities)


class RandomForestDigitClassifier(DigitClassificationInterface, ABC):
    def __dummy_fit(self) -> RandomForestClassifier:
        # As we cannot inference on unfit RFC, I implemented dummy fit for one sample
        model = RandomForestClassifier()
        dummy_x = self._preprocess_inputs(np.random.random(self.input_shape))
        model.fit(np.asarray([dummy_x]).reshape(1, -1), [0])
        return model

    def _setup_model(self) -> RandomForestClassifier:
        model = self.__dummy_fit()
        return model

    def _preprocess_inputs(self, inputs: np.ndarray) -> np.ndarray:
        return inputs.flatten().reshape(1, -1)

    def _predict(self, inputs: np.ndarray) -> int:
        return self._model.predict(inputs)[0]


class RandomDigitClassifier(DigitClassificationInterface, ABC):
    def __init__(self, crop_size: int = 10):
        super(RandomDigitClassifier, self).__init__()
        self.crop_size = crop_size

    def _setup_model(self) -> RandomModel:
        return RandomModel()

    def _preprocess_inputs(self, inputs: np.ndarray) -> np.ndarray:
        start_x = self.input_shape[0] // 2 - self.crop_size // 2
        end_x = start_x + self.crop_size

        start_y = self.input_shape[1] // 2 - self.crop_size // 2
        end_y = start_y + self.crop_size

        return inputs[start_y:end_y, start_x:end_x, :]

    def _predict(self, inputs: np.ndarray) -> int:
        return self._model.predict(inputs)


if __name__ == '__main__':
    # Test Random center crop and predict
    random_digit_clf = RandomDigitClassifier()
    x_data = np.random.random(random_digit_clf.input_shape)
    preprocessed_x = random_digit_clf._preprocess_inputs(x_data)
    prediction = random_digit_clf.predict(x_data)
    assert preprocessed_x.shape == (10, 10, 1)
    assert prediction in range(0, 10)

    # Test CNN predict
    cnn_digit_clf = CNNDigitClassifier()
    prediction = cnn_digit_clf.predict(x_data)
    assert prediction in range(0, 10)

    # Test RFC predict
    rfc_digit_clf = RandomForestDigitClassifier()
    preprocessed_x = rfc_digit_clf._preprocess_inputs(x_data)
    prediction = rfc_digit_clf.predict(x_data)
    assert preprocessed_x.shape == (1, 784)
    assert prediction in range(0, 10)

    # Test invalid input
    invalid_input = np.random.random((29, 28, 1))
    digit_clf = CNNDigitClassifier()
    try:
        digit_clf._validate_input(invalid_input)
    except AssertionError:
        pass
    else:
        raise AssertionError("Invalid input test fail")