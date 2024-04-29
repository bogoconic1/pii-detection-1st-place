import yaml
import subprocess
import os
import random
from datetime import datetime

# Load the configuration file
with open("configs/bilstm_config.yaml", "r") as file:
    config = yaml.safe_load(file)

# Set environment variables
os.environ["CUDA_HOME"] = config["environment"]["cuda_home"]
os.environ["NCCL_DEBUG"] = config["environment"]["nccl_debug"]
os.environ["CUDA_VISIBLE_DEVICES"] = config["environment"]["cuda_visible_devices"]

# Load training configurations
training_config = config["training"]

# Generate a random port using Python's random module
port = random.randint(25000, 30000)

# Loop through nums
for VALIDATION_FOLD in config["validation_folds"]:
    # Generate a random seed for each loop iteration
    seed = random.randint(1, 32767)
    print(f"Seed: {seed}")  # To mimic the echo of the seed in the original script

    MODEL_NAME = f"bilstm1-mpware-yuv-fp16-fullfit-seed{seed}-fold{VALIDATION_FOLD}"
    OUTPUT_DIR = f"models/{MODEL_NAME}-maxlen{training_config['max_length']}-lr{training_config['learning_rate']}"
    current_date = datetime.now().strftime("%y%m%d_%H%M")

    # Construct the training command as a list
    command = [
        "accelerate",
        "launch",
        "--main_process_port",
        str(port),
        "--multi_gpu",
        "--num_processes",
        "8",
        "deberta-BiLSTM.py",
        "--output_dir",
        OUTPUT_DIR,
        "--model_path",
        training_config["model_path"],
        "--validation_fold",
        str(VALIDATION_FOLD),
        "--max_length",
        str(training_config["max_length"]),
        "--learning_rate",
        str(training_config["learning_rate"]),
        "--per_device_train_batch_size",
        str(training_config["per_device_train_batch_size"]),
        "--per_device_eval_batch_size",
        str(training_config["per_device_eval_batch_size"]),
        "--num_train_epochs",
        str(training_config["num_train_epochs"]),
        "--save_steps",
        str(training_config["save_steps"]),
        "--o_weight",
        str(training_config["o_weight"]),
        "--seed",
        str(seed),
        "--adv_mode",
        training_config["adv_stop_mode"],
        "--adv_start",
        str(training_config["adv_start"]),
        "--loss",
        training_config["loss"],
        "--smoke_test",
        str(training_config["smoke_test"]),
        "--fullfit",
        str(training_config["fullfit"]),
    ]

    # Execute the command and redirect stdout and stderr
    with open(
        f"logs/bilstm1-fold{VALIDATION_FOLD}-fp16-{current_date}.log", "w"
    ) as log_file:
        process = subprocess.Popen(
            command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
        )
        for line in process.stdout:
            print(line, end="")
            log_file.write(line)

    process.stdout.close()
    process.wait()
