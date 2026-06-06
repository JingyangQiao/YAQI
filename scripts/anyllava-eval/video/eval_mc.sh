#!/bin/bash

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}

if [ ! -n "$2" ] ;then
    MODELPATH='./checkpoints/llava-llama-2-7b-chat-LanguageBind-llava_llama_2-finetune-video-mirror'
else
    MODELPATH=$2
fi

RESULT_DIR="./results/llama-2-7b-mirror/Video/MC_Mirror_Video_"

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m videollava.eval.anyllava.model_video_qa \
        --model-path $MODELPATH \
        --question-file ./Dataset/Video/mc_test.jsonl \
        --video-folder ./Dataset/Video \
        --answers-file $RESULT_DIR/${CHUNKS}_${IDX}.jsonl \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \
        --conv-mode llava_llama_2 &
done

wait

output_file=$RESULT_DIR/llama_2_7b_video_model_video_modality_mc_dataset_mirror.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat $RESULT_DIR/${CHUNKS}_${IDX}.jsonl >> "$output_file"
done

python -m videollava.eval.anyllava.video.eval_mc \
    --result-file $output_file \
    --output-dir $RESULT_DIR \
