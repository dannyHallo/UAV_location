import numpy as np


time_interval=0.02
train_num_of_lines_to_generate_per_region = 7000
test_num_of_lines_to_generate_per_region = 300
# step_count_per_line = 5
# length_per_step = 0.2
train_detecting_region_nums = 100
test_detecting_region_nums = 3

epoch = 500
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
optimizer = "LBFGS"
# optimizer = "Adam"
