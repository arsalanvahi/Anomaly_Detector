
import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# -------------------------
# Parameters / Hyperparams
# -------------------------
WINDOW_SIZE = 32        # bytes per window
WINDOW_STEP = 8         # sliding step
IMAGE_SIDE = 32         # we will convert each 1D window to a IMAGE_SIDE x IMAGE_SIDE "image"
CHANNELS = 1            # grayscale
RANDOM_SEED = 42
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 1e-3
DROPOUT_PROB = 0.5

np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)


# -------------------------
# 1) Window sliding
# -------------------------
def window_sliding(x_list, window_size=WINDOW_SIZE, window_step=WINDOW_STEP):
    """
    Convert a list of variable-length integer byte sequences into a numpy array of fixed-length windows.

    Args:
        x_list (list[list[int]]): list of variable-length sequences (each int in 0..255)
        window_size (int): desired window length in bytes
        window_step (int): sliding step

    Returns:
        np.ndarray: shape (num_windows, window_size), dtype=np.uint8
    """
    windows = []
    for line in x_list:
        n = len(line)
        if n == 0:
            continue
        # If the sequence is shorter than window_size, pad on the left with zeros
        if n <= window_size:
            padded = ([0] * (window_size - n)) + list(line)
            windows.append(padded)
            continue

        # sliding windows across the sequence
        for j in range(0, n - window_size + 1, window_step):
            windows.append(line[j:j + window_size])

        # handle the tail: if last window doesn't align, append final tail window
        if (n - window_size) % window_step != 0:
            windows.append(line[-window_size:])
    return np.array(windows, dtype=np.uint8)


# -------------------------
# 2) Convert 1D windows -> 2D image-like tensors for CNN
# -------------------------
def windows_to_images(windows, image_side=IMAGE_SIDE):
    """
    Convert each 1D window (length WINDOW_SIZE) into a square image of shape (image_side, image_side).
    Simple method: tile/repeat the 1D vector to fill the square. Alternative strategies may be preferred.

    Args:
        windows (np.ndarray): shape (N, WINDOW_SIZE)
        image_side (int): side length of square image

    Returns:
        np.ndarray: shape (N, image_side, image_side, CHANNELS), dtype=float32 normalized in [0,1]
    """
    N = windows.shape[0]
    # Normalize bytes to [0,1]
    windows_f = windows.astype(np.float32) / 255.0  # shape (N, WINDOW_SIZE)

    # Repeat each window to create image_side rows. This is a simple representation:
    # each row is the same sequence (you can experiment with more advanced mappings).
    images = np.tile(windows_f[:, np.newaxis, :], (1, image_side, 1))  # shape (N, image_side, WINDOW_SIZE)

    # If WINDOW_SIZE != image_side, we need to either crop or pad columns to image_side
    if windows.shape[1] < image_side:
        # pad columns on the right with zeros
        pad_width = image_side - windows.shape[1]
        images = np.pad(images, ((0, 0), (0, 0), (0, pad_width)), mode='constant', constant_values=0.0)
    elif windows.shape[1] > image_side:
        images = images[:, :, :image_side]

    # final shape (N, image_side, image_side)
    images = images.reshape(N, image_side, image_side, 1)  # add channel dim
    return images.astype(np.float32)


# -------------------------
# 3) Model creation (CNN)
# -------------------------
def create_network(input_shape=(IMAGE_SIDE, IMAGE_SIDE, CHANNELS),
                   layer_dims=(32, 64, 128),
                   filter_sizes=((3, 3), (3, 3)),
                   dropout_prob=DROPOUT_PROB):
    """
    Build and compile a CNN model for binary classification (normal vs attack).

    Args:
        input_shape (tuple): shape of input images
        layer_dims (tuple): number of filters / units for conv/dense layers
        filter_sizes (tuple): conv kernel sizes
        dropout_prob (float): dropout rate

    Returns:
        compiled tf.keras.Model
    """
    inputs = tf.keras.layers.Input(shape=input_shape, name='InputLayer')

    x = tf.keras.layers.Conv2D(filters=layer_dims[0], kernel_size=filter_sizes[0],
                               padding='same', activation='relu', name='ConvBlock1')(inputs)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), padding='same', name='PoolBlock1')(x)

    x = tf.keras.layers.Conv2D(filters=layer_dims[1], kernel_size=filter_sizes[1],
                               padding='same', activation='relu', name='ConvBlock2')(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), padding='same', name='PoolBlock2')(x)

    x = tf.keras.layers.Flatten(name='FlattenLayer')(x)
    x = tf.keras.layers.Dense(layer_dims[2], activation='relu', name='DenseLayer')(x)
    x = tf.keras.layers.Dropout(rate=dropout_prob, name='DropoutLayer')(x)

    outputs = tf.keras.layers.Dense(2, activation='softmax', name='OutputLayer')(x)

    model = tf.keras.models.Model(inputs=inputs, outputs=outputs, name='NetworkModel')
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model


# -------------------------
# 4) Synthetic dataset builder (for demo/testing)
# -------------------------
def build_synthetic_dataset(n_normal=2000, n_attack=2000, seq_len_range=(20, 150)):
    """
    Build a small synthetic dataset of packet byte sequences.
    - Normal packets: random bytes
    - Attack packets: random bytes but with a repeated malicious pattern embedded to make them distinguishable

    Returns:
        windows (np.ndarray): shape (total_windows, WINDOW_SIZE)
        labels (np.ndarray): shape (total_windows,) 0 normal, 1 attack
    """
    normal_seqs = []
    for _ in range(n_normal):
        L = np.random.randint(seq_len_range[0], seq_len_range[1])
        seq = np.random.randint(0, 256, size=L).tolist()
        normal_seqs.append(seq)

    attack_seqs = []
    # define a simple pattern that will tend to appear in attack packets
    malicious_pattern = [222, 173, 190, 239, 0, 13, 37, 99]  # arbitrary byte pattern
    for _ in range(n_attack):
        L = np.random.randint(seq_len_range[0], seq_len_range[1])
        seq = np.random.randint(0, 256, size=L).tolist()
        # inject the malicious pattern at a random position one or more times
        pos = np.random.randint(0, max(1, L - len(malicious_pattern)))
        seq[pos:pos+len(malicious_pattern)] = malicious_pattern
        # sometimes inject multiple times
        if np.random.rand() < 0.3:
            pos2 = np.random.randint(0, max(1, L - len(malicious_pattern)))
            seq[pos2:pos2+len(malicious_pattern)] = malicious_pattern
        attack_seqs.append(seq)

    # create windows and labels
    normal_windows = window_sliding(normal_seqs, window_size=WINDOW_SIZE, window_step=WINDOW_STEP)
    attack_windows = window_sliding(attack_seqs, window_size=WINDOW_SIZE, window_step=WINDOW_STEP)

    X = np.concatenate([normal_windows, attack_windows], axis=0)
    y = np.concatenate([np.zeros(len(normal_windows), dtype=np.int32),
                        np.ones(len(attack_windows), dtype=np.int32)], axis=0)

    # shuffle
    perm = np.random.permutation(len(X))
    X = X[perm]
    y = y[perm]
    return X, y


# -------------------------
# 5) Main runnable pipeline
# -------------------------
def main():
    print("Building synthetic dataset...")
    X_windows, y = build_synthetic_dataset(n_normal=1200, n_attack=1200)

    print(f"Total windows: {len(X_windows)}, labels distribution: {np.bincount(y)}")

    print("Converting windows to images...")
    X_images = windows_to_images(X_windows, image_side=IMAGE_SIDE)  # shape (N, IMAGE_SIDE, IMAGE_SIDE, 1)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X_images, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y)

    print("Creating model...")
    model = create_network(input_shape=(IMAGE_SIDE, IMAGE_SIDE, CHANNELS))
    model.summary()

    # Callbacks
    checkpoint_path = "dns_model_checkpoint.h5"
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(checkpoint_path, monitor='val_accuracy', save_best_only=True, verbose=1),
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True, verbose=1)
    ]

    print("Starting training...")
    history = model.fit(
        X_train, y_train,
        validation_split=0.15,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=2
    )

    print("Evaluating on test set...")
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test accuracy: {test_acc:.4f}  -  Test loss: {test_loss:.4f}")

    # Predictions + classification report
    y_pred_prob = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_prob, axis=1)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, digits=4))

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # Save final model
    out_model_path = "dns_model.h5"
    model.save(out_model_path)
    print(f"Saved trained model to: {out_model_path}")


if __name__ == "__main__":
    main()
