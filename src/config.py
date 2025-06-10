stage_1_train_set_path = "dataset/stage_1_train.pt"
stage_1_test_set_path = "dataset/stage_1_test.pt"

time_interval = 0.01
train_num_of_lines_to_generate_per_region = 20000
test_num_of_lines_to_generate_per_region = 100
train_detecting_region_nums = 1
test_detecting_region_nums = 1
visualize_detecting_region_nums = 1
region_seed = 42
train_line_seed = 42
test_line_seed = 66
visualize_line_seed = 66

epoch = 300
batch_size = 4096

learning_rate = 2e-4

# doppler and w para
fc = 6e9
c = 3e8
v = 30

optimizer = "Adam"  # ["LBFGS" / "Adam"]
