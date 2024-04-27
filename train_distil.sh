export CUDA_HOME='/usr/local/cuda'
export NCCL_DEBUG=INFO
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

port=$(shuf -i25000-30000 -n1)

FULLFIT=0
TRAINING_MODEL_PATH="microsoft/deberta-v3-large"
TRAINING_MAX_LENGTH=2048
HASH_NAME=maxlen$TRAINING_MAX_LENGTH
LR=2e-5
SAVE_STEPS=0.1
SEED=42

for VALIDATION_FOLD in 2 1 3; do
MODEL_NAME=distil2-mpware-fp16-fold$VALIDATION_FOLD
TEACHER_MODEL_PATH=../models/pii-detection-models/basic-model-2-fold-$VALIDATION_FOLD/basic-model-2-fold-$VALIDATION_FOLD
OUTPUT_DIR=$MODEL_NAME-$HASH_NAME-lr$LR
current_date=$(date "+%y%m%d_%H%M")
accelerate launch --main_process_port $port --multi_gpu --num_processes 8 distillation.py \
    --teacher $TEACHER_MODEL_PATH \
    --validation_fold $VALIDATION_FOLD \
    --output_dir $OUTPUT_DIR \
    --model_path $TRAINING_MODEL_PATH \
    --max_length $TRAINING_MAX_LENGTH \
    --learning_rate $LR \
    --per_device_train_batch_size 10 \
    --per_device_eval_batch_size 10 \
    --num_train_epochs 10 \
    --save_steps $SAVE_STEPS \
    --o_weight 0.05 \
    --seed $SEED \
    --smoke_test 0 \
    --fullfit $FULLFIT \
    2>&1 | tee logs/distil2-fp16-$current_date.log
done

# VALIDATION_FOLD=0
# for NUM in 0 1; do
#     SEED=$(awk 'BEGIN{srand(); print int(rand()*(32767-1+1) + 1)}')
#     echo "Seed: $SEED"  # This line will show you the value of SEED
