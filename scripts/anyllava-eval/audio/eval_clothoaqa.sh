#!/bin/bash

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}

if [ ! -n "$2" ] ;then
    MODELPATH='./checkpoints/llava-vicuna-13b-v1.5-LanguageBind-v1-finetune-video-mirror'
else
    MODELPATH=$2
fi

RESULT_DIR="./results/vicuna-v1.5-13b-mirror/Audio/Clothoaqa_Mirror_Video"

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m videollava.eval.anyllava.model_audio_qa \
        --model-path $MODELPATH \
        --question-file ./Dataset/Audio/clothoaqa/annotations/clothoaqa_test.jsonl \
        --audio-folder ./Dataset/Audio \
        --answers-file $RESULT_DIR/${CHUNKS}_${IDX}.jsonl \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \
        --conv-mode v1 &
done

wait

output_file=$RESULT_DIR/vicuna_v1.5_13b_video_model_audio_modality_clothoaqa_dataset_mirror.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat $RESULT_DIR/${CHUNKS}_${IDX}.jsonl >> "$output_file"
done

python -m videollava.eval.anyllava.audio.eval_clothoaqa \
    --result-file $output_file \
    --output-dir $RESULT_DIR \
