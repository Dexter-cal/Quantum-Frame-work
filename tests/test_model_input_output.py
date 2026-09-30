import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import qai


print("=" * 60)
print("TEST: model.input() / model.output() expose real prediction state")
print("=" * 60)

model = qai.build(type="regression")
model.train(np.array([[0.0], [1.0], [2.0]]), np.array([1.0, 3.0, 5.0]))

assert model.input() is None
assert model.output() is None
print("Fresh model correctly reports no prediction state")

sample = np.array([4.0])
prediction = model.predict(sample)

recorded_input = model.input()
recorded_output = model.output()
assert np.array_equal(recorded_input, sample)
assert recorded_output == prediction
print("Accessors return the exact latest prediction input and output")

recorded_input[0] = 999.0
assert model.input()[0] == 4.0
print("Input accessor returns a defensive copy of mutable array state")

batch = np.array([[3.0], [5.0]])
batch_prediction = model.predict(batch)
recorded_batch_output = model.output()
assert np.array_equal(model.input(), batch)
assert np.array_equal(recorded_batch_output, batch_prediction)

recorded_batch_output[0] = -999.0
assert model.output()[0] != -999.0
print("Output accessor returns a defensive copy of mutable array state")

print("ALL model.input() / model.output() TESTS PASSED")
