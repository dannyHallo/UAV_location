stage_1_train_set_path = "cache/stage_1_train.pt"
stage_1_test_set_path = "cache/stage_1_test.pt"

time_interval = 0.01
train_num_of_lines_to_generate_per_region = 25
test_num_of_lines_to_generate_per_region = 2
train_detecting_region_nums = 800
test_detecting_region_nums = 20
visualize_detecting_region_nums = 5
train_seed = 42
test_seed = 66
visualize_seed = 77

# step_count_per_line = 5
# length_per_step = 0.2

epoch = 50
train_batch_size = 128
# test_batch_size = test_set_num

learning_rate = 0.00008

# doppler and w para
fc = 6e9
c = 3e8
v = 30

optimizer = "Adam"  # ["LBFGS" / "Adam"]
