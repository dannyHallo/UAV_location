stage_1_dataset_train_path = "dataset/stage_1_dataset_train.pt"
stage_1_dataset_test_path = "dataset/stage_1_dataset_test.pt"

train_test_split_ratio = 0.85

time_interval = 0.01

num_detecting_regions = 1
visualize_num_detecting_regions = 1
num_lines_to_generate_per_region = 500

sequence_length = 2

line_seed_train = 2024
line_seed_test = 2026
visualize_line_seed = 0

region_seed = 42  # the only region we are using

train_test_split_seed = 77

#  TRAINING HYPER-PARAMETERS
epoch = 500
batch_size = 4096
learning_rate = 2e-4
optimizer = "Adam"  # ["LBFGS" | "Adam"]

#  CONSTANTS FOR DOPPLER / W
fc = 6e9
c = 3e8
v = 30
