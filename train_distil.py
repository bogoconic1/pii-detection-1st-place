import yaml
import subprocess
from datetime import datetime
import os

# Load the configuration file
with open("configs/distil_config.yaml", "r") as file:
    config = yaml.safe_load(file)

# Set environment variables
os.environ["CUDA_HOME"] = config["environment"]["cuda_home"]
os.environ["NCCL_DEBUG"] = config["environment"]["nccl_debug"]
os.environ["CUDA_VISIBLE_DEVICES"] = config["environment"]["cuda_visible_devices"]

# Load training configurations
training_config = config["training"]

# Get a random port
port = subprocess.run(
    ["shuf", "-i25000-30000", "-n1"], capture_output=True, text=True
).stdout.strip()

# Loop through each validation fold
for VALIDATION_FOLD in config["validation_folds"]:
    MODEL_NAME = f"distil2-mpware-fp16-fold{VALIDATION_FOLD}"
    # Teacher model path may be adjusted accordingly
    TEACHER_MODEL_PATH = training_config["teacher_model_path"]
    OUTPUT_DIR = f"models/{MODEL_NAME}-maxlen{training_config['max_length']}-lr{training_config['learning_rate']}"
    current_date = datetime.now().strftime("%y%m%d_%H%M")

    # Construct the training command as a list
    command = [
        "accelerate",
        "launch",
        "--main_process_port",
        port,
        "--multi_gpu",
        "--num_processes",
        "8",
        "distillation.py",
        "--teacher",
        TEACHER_MODEL_PATH,
        "--validation_fold",
        str(VALIDATION_FOLD),
        "--output_dir",
        OUTPUT_DIR,
        "--model_path",
        training_config["model_path"],
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
        str(training_config["seed"]),
        "--smoke_test",
        str(training_config["smoke_test"]),
        "--fullfit",
        str(training_config["fullfit"]),
    ]

    # Execute the command and redirect stdout and stderr
    with open(f"logs/distil2-fp16-{current_date}.log", "w") as log_file:
        process = subprocess.Popen(
            command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
        )
        for line in process.stdout:
            print(line, end="")
            log_file.write(line)

    process.stdout.close()
    process.wait()
