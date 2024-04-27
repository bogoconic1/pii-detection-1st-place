port=$(shuf -i25000-30000 -n1)
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NCCL_DEBUG=INFO

TRAINING_MODEL_PATH="microsoft/deberta-v3-large"
TRAINING_MAX_LENGTH=1800
VERSION=21
HASH_NAME="maxlen_$TRAINING_MAX_LENGTH"
PEFT=False
SEED=42
ADV_STOP_MODE="epoch"
ADV_START=100
LOSS="ce"
LR=8e-5

for VALIDATION_FOLD in 0 1 2 3; do
    OUTPUT_DIR="../working/bogo-$VALIDATION_FOLD-lr$LR"
    MODEL_NAME="custom-model-$VERSION-fold-$VALIDATION_FOLD"
    current_date=$(date "+%y%m%d_%H%M")

    accelerate launch --num_processes 8 deberta-multi-dropouts.py \
        --output_dir $OUTPUT_DIR \
        --validation_fold $VALIDATION_FOLD \
        --model_path $TRAINING_MODEL_PATH \
        --max_length $TRAINING_MAX_LENGTH \
        --learning_rate $LR \
        --per_device_train_batch_size 14 \
        --per_device_eval_batch_size 14 \
        --num_train_epochs 10 \
        --save_steps 32 \
        --o_weight 0.05 \
        --model_name $MODEL_NAME \
        --hash $HASH_NAME \
        --peft $PEFT \
        --seed $SEED \
        --adv_mode $ADV_STOP_MODE \
        --adv_start $ADV_START \
        --loss $LOSS \
        2>&1 | tee logs/$MODEL_NAME-$HASH_NAME-$current_date.log
done
