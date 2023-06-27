import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
import matplotlib.pyplot as plt

def SinModel():
    ipt = keras.Input(shape=(1,), dtype="float32")
    tmp = keras.layers.Dense(40, activation="relu")(ipt)
    tmp = keras.layers.Dense(40, activation="relu")(tmp)
    out = keras.layers.Dense(1)(tmp)
    return keras.Model(inputs=ipt, outputs=out)

def SinGenerator(amp=None, phase=None):
    if amp is None:
        amp = tf.random.uniform(shape=[], minval=0.1, maxval=5.0)
    if phase is None:
        phase = tf.random.uniform(shape=[], minval=0.0, maxval=np.pi)
    def _gen(x):
        return tf.math.sin(x - phase) * amp
    return _gen

def genX(sample, minval=-5.0, maxval=5.0):
    return tf.expand_dims(tf.random.uniform(shape=[sample], minval=minval, maxval=maxval),1)

def maml_train(model, total_iterations=2201, meta_train_steps=10, meta_test_steps=100):
    log_step = total_iterations // 10 if total_iterations > 10 else 1

    optim_test = keras.optimizers.SGD(learning_rate=0.005)
    optim_train = keras.optimizers.SGD(learning_rate=0.005)
    losses = []
    total_l = 0.0

    # Store the predictions for normal training and meta-learning
    normal_predictions = []
    meta_predictions = []

    for step in range(total_iterations):
        sinGen = SinGenerator()

        model_copy = tf.keras.models.clone_model(model)
        model_copy.set_weights(model.get_weights())

        test_x = genX(meta_test_steps)
        test_gt = genX(meta_train_steps)
        train_x = genX(meta_train_steps)
        train_gt = sinGen(train_x)

        with tf.GradientTape(watch_accessed_variables=False) as train_tape:
            train_tape.watch(model_copy.trainable_variables)
            train_pred = model_copy(train_x)
            train_loss = tf.reduce_mean(keras.losses.mse(train_gt, train_pred))

        gradients = train_tape.gradient(train_loss, model_copy.trainable_variables)
        optim_test.apply_gradients(zip(gradients, model.trainable_variables))

        total_l += train_loss

        losses.append(total_l / (step + 1))

        if step % log_step == 0:
            print(f'step: {step}. Loss: {total_l / (step + 1)}')

        # Store the predictions for each step
        normal_predictions.append(model.predict(test_x))
        meta_predictions.append(model_copy.predict(test_x))

    plt.plot(losses)
    plt.show()

    # Plot the predictions from normal training and meta-learning
    plt.plot(test_x, normal_predictions[-1], label="Normal Training")
    plt.plot(test_x, meta_predictions[-1], label="Meta-Learning")
    plt.legend()
    plt.show()

model = SinModel()
maml_train(model)