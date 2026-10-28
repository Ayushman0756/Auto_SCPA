import tensorflow as tf

# Check available GPUs
gpus = tf.config.list_physical_devices('GPU')

print("Available GPUs:", gpus)

