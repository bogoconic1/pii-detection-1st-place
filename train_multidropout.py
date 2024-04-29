import yaml
import subprocess
from datetime import datetime

# Load the configuration file
with open('configs/multidropouts_config.yaml', 'r') as file:
    config = yaml.safe_load(file)

# Load training configurations
training_config = config['training']

# Loop through each validation fold
for VALIDATION_FOLD in config['validation_folds']:
    OUTPUT_DIR = f"models/multidropouts-{VALIDATION_FOLD}-lr{training_config['learning_rate']}"
    MODEL_NAME = f"custom-model-{training_config['max_length']}-fold-{VALIDATION_FOLD}"
    current_date = datetime.now().strftime("%y%m%d_%H%M")

    # Construct the training command
    command = [
        "accelerate", "launch", "--num_processes", "8", "deberta-multi-dropouts.py",
        "--output_dir", OUTPUT_DIR,
        "--validation_fold", str(VALIDATION_FOLD),
        "--model_path", training_config['model_path'],
        "--max_length", str(training_config['max_length']),
        "--learning_rate", str(training_config['learning_rate']),
        "--per_device_train_batch_size", str(training_config['per_device_train_batch_size']),
        "--per_device_eval_batch_size", str(training_config['per_device_eval_batch_size']),
        "--num_train_epochs", str(training_config['num_train_epochs']),
        "--save_steps", str(training_config['save_steps']),
        "--o_weight", str(training_config['o_weight']),
        "--model_name", MODEL_NAME,
        "--hash", training_config['hash_name'],
        "--peft", str(training_config['peft']).lower(),
        "--seed", str(training_config['seed']),
        "--adv_mode", training_config['adv_stop_mode'],
        "--adv_start", str(training_config['adv_start']),
        "--loss", training_config['loss']
    ]

    # Print command (can replace print with subprocess.run(command) to execute)
    print(" ".join(command))
