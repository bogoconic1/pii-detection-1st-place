export CUDA_HOME='/usr/local/cuda'
export NCCL_DEBUG=INFO
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

port=$(shuf -i25000-30000 -n1)

FULLFIT=1
TRAINING_MODEL_PATH="microsoft/deberta-v3-large"
TRAINING_MAX_LENGTH=2048
ADV_STOP_MODE="epoch"
ADV_START=100
LOSS="ce"
HASH_NAME=maxlen$TRAINING_MAX_LENGTH
LR=7e-5
SAVE_STEPS=0.5

for NUM in 0 1; do
    SEED=$(awk 'BEGIN{srand(); print int(rand()*(32767-1+1) + 1)}')
    echo "Seed: $SEED"  # This line will show you the value of SEED

    MODEL_NAME=bilstm1-mpware-yuv-fp16-fullfit-seed$SEED
    OUTPUT_DIR=$MODEL_NAME-$HASH_NAME-lr$LR
    current_date=$(date "+%y%m%d_%H%M")
    accelerate launch --main_process_port $port --multi_gpu --num_processes 8 deberta-BiLSTM.py \
        --output_dir $OUTPUT_DIR \
        --model_path $TRAINING_MODEL_PATH \
        --max_length $TRAINING_MAX_LENGTH \
        --learning_rate $LR \
        --per_device_train_batch_size 12 \
        --per_device_eval_batch_size 12 \
        --num_train_epochs 4 \
        --save_steps $SAVE_STEPS \
        --o_weight 0.05 \
        --seed $SEED \
        --adv_mode $ADV_STOP_MODE \
        --adv_start $ADV_START \
        --loss $LOSS \
        --smoke_test 0 \
        --fullfit $FULLFIT \
        2>&1 | tee logs/bilstm1-fullfit-fp16-$current_date.log
done

# for VALIDATION_FOLD in 0 1 2 3; do
        # --validation_fold $VALIDATION_FOLD \
# VALIDATION_FOLD=1