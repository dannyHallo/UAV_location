import numpy as np

transmittor_position = np.array([0, 0])
receiver_position_1 = np.array([200, 0])
receiver_position_2 = np.array([100, 100])
receiver_position_3 = np.array([300, 150])

t1=0.01
t2=0.01
train_num_of_lines_to_generate = 10000
test_num_of_lines_to_generate = 1
step_count_per_line = 5
length_per_step = 0.2


epoch = 1000
train_batch_size = 128
# test_batch_size = test_set_num

learning_rate = 0.0001

# oppler and w para
fc = 6e9
c = 3e8
v = 30

# using_model = "rbf"
using_model = "dnn"
# using_model = "Transformer"

# FIXME:
# optimizer = "LBFGS"
optimizer = "Adam"
