#!/bin/bash

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}

if [ ! -n "$2" ] ;then
    MODELPATH='./checkpoints/llava-vicuna-7b-v1.5-LanguageBind-v1-finetune-music-next'
else
    MODELPATH=$2
fi

RESULT_DIR="./results/vicuna-v1.5-7b-next/Text/Gsm8k_Next_Music"

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m videollava.eval.anyllava.model_text_qa \
        --model-path $MODELPATH \
        --question-file ./Dataset/Text/gsm8k/gsm8k_test.jsonl \
        --answers-file $RESULT_DIR/${CHUNKS}_${IDX}.jsonl \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \
        --conv-mode v1 &
done

wait

output_file=$RESULT_DIR/vicuna_v1.5_7b_music_model_text_modality_gsm8k_dataset_next.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat $RESULT_DIR/${CHUNKS}_${IDX}.jsonl >> "$output_file"
done

python -m videollava.eval.anyllava.text.eval_gsm8k \
    --result-file $output_file \
    --output-dir $RESULT_DIR \
