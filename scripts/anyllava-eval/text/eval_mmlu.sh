#!/bin/bash

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}

if [ ! -n "$2" ] ;then
    MODELPATH='./checkpoints/vicuna-13b-v1.5'
else
    MODELPATH=$2
fi

RESULT_DIR="./results/vicuna-v1.5-13b-mirror/Text/MMLU_Mirror_Text"

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m videollava.eval.anyllava.model_text_qa \
        --model-path $MODELPATH \
        --question-file ./Dataset/Text/mmlu/mmlu_test.jsonl \
        --answers-file $RESULT_DIR/${CHUNKS}_${IDX}.jsonl \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \
        --conv-mode v1 &
done

wait

output_file=$RESULT_DIR/lvicuna_v1.5_13b_text_model_text_modality_mmlu_dataset_mirror.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat $RESULT_DIR/${CHUNKS}_${IDX}.jsonl >> "$output_file"
done

python -m videollava.eval.anyllava.text.eval_mmlu \
    --result-file $output_file \
    --output-dir $RESULT_DIR \
