import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical

# --- Load and prepare data  --------------------------------------------------
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train.reshape(-1, 784).astype('float32') / 255
X_test  = X_test.reshape(-1, 784).astype('float32') / 255
y_train = to_categorical(y_train, 10)
y_test  = to_categorical(y_test, 10)

# --- Build a small model  -----------------------------------------------------
model = Sequential([
    Dense(64, activation='relu',  input_shape=(784,)),
    Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(X_train, y_train,
          batch_size=128, epochs=10,
          validation_data=(X_test, y_test),
          verbose=1)

score = model.evaluate(X_test, y_test, verbose=0)
print("Test loss:", score[0])
print("Test accuracy:", score[1])

predictions = model.predict(X_test[:19])
predicted_classes = np.argmax(predictions, axis=1)

print("Predictions for the first 19 inputs:")
for i, pred in enumerate(predicted_classes):
    print(f"Input {i}: Predicted digit {pred}")

# --- Exporter A: Weights and Biases as combined array  ----------------------
def export_weights_to_header(model, header_file='mnist_ffn_weights.h'):
    """
    For each trainable layer, export both bias and weight as a combined C array:
    Format: float layerX[] = { bias values..., weight values... };
    """
    with open(header_file, 'w') as f:
        f.write('// Auto-generated C header from Keras model parameters\n\n')
        for idx, layer in enumerate(model.layers):
            weights = layer.get_weights()      # list: [weights, biases] for Dense layer
            if not weights:
                continue  # Skip if no trainable parameters (e.g., Dropout layers)
            # Note: In Keras Dense, get_weights() returns [kernel, bias]
            # We want the order bias then weights.
            kernel, bias = None, None
            if len(weights) == 2:
                kernel, bias = weights[0], weights[1]
            else:
                # If only one set is returned, print it directly.
                bias = weights[0]
            
            combined = []
            # Add bias values first
            if bias is not None:
                combined.extend(bias.flatten())
            # Now add flattened kernel (weights) values if available
            if kernel is not None:
                combined.extend(kernel.flatten())
            
            # Write the combined array
            f.write(f'// Layer {idx}: Bias then Weights; original shapes: ')
            if bias is not None:
                f.write(f'bias {bias.shape} ')
            if kernel is not None:
                f.write(f'weights {kernel.shape}')
            f.write('\n')
            array_name = f'layer{idx}'
            f.write(f'float {array_name}[] = {{\n    ')
            f.write(', '.join(f'{v:.8e}' for v in combined))
            f.write('\n};\n\n')
    print(f'Weights written to {header_file}')

# --- Exporter B: Test inputs as a single array  -----------------------------
def export_inputs_to_header(X, header_file='mnist_test_inputs.h', n_samples=20):
    """
    Print the first n_samples test inputs (flattened) into a single C array.
    Array format: float input[] = { input0, input1, ..., input{n_samples*784 -1} };
    """
    total_samples = X.shape[0]
    if n_samples > total_samples:
        raise ValueError("n_samples exceeds the number of test samples available.")
    
    # Select the first n_samples from X
    selected = X[:n_samples]  # shape (n_samples, 784)
    # Flatten in row-major order (i.e., the first 784 will be from the first input, etc.)
    combined_inputs = selected.flatten()
    
    with open(header_file, 'w') as f:
        f.write('// Auto-generated C header from test inputs (flattened)\n')
        f.write(f'// Contains {n_samples} test inputs; each MNIST image is 784 values long.\n\n')
        f.write('float input[] = {\n    ')
        f.write(', '.join(f'{v:.8e}' for v in combined_inputs))
        f.write('\n};\n')
    print(f'{n_samples} inputs written to {header_file}')

# --- Call exporters -----------------------------------------------------------
export_weights_to_header(model, 'mnist_ffn_weights.h')
export_inputs_to_header(X_test, 'mnist_test_inputs.h', n_samples=19)