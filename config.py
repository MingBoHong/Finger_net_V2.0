# Hyperparameter
growth_k = 24
nb_block = 2 # how many (dense block + Transition Layer) ?
dropout_rate = 0.2
class_num = 2

# Momentum Optimizer will use
nesterov_momentum = 0.9
weight_decay = 1e-4

# Label & batch_size
batch_size = 64


checkpoint_path = 'model/checkpoint'  # 设置模型参数文件所在路径
event_log_path = 'event-log'  # 设置事件文件所在路径，用于周期性存储Summary缓存对象
Trainrecords = 'Train.tfrecords'
Testrecords = 'Test.tfrecords'
epochs = 10

l2 = 0.0001
decay_step = 10000               # 衰减迭代数
learning_rate_decay_factor = 0.1  # 学习率衰减因子
initial_learning_rate = 0.1      # 初始学习率

pic_size=32