import tensorflow as tf

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_integer("batch_size", 100, "batch size for training")
tf.flags.DEFINE_integer("epoches", 5, "number of epoches")
tf.flags.DEFINE_integer("disp", 50, "how many iterations to display")
tf.flags.DEFINE_float("weight_decay", 0.001, "weight decay")
tf.flags.DEFINE_float("learning_rate", 0.005, "learning rate")
tf.flags.DEFINE_string("data_path", "./data/", "data path storing npy files")
tf.flags.DEFINE_string("log_path", "./log/", "log path storing checkpoints")
tf.flags.DEFINE_string("mode", "test", "train or test")

tfconfig = tf.ConfigProto(allow_soft_placement=True)
tfconfig.gpu_options.allow_growth = True
