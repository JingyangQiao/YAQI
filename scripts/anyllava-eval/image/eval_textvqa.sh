#!/bin/bash

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}

if [ ! -n "$2" ] ;then
    MODELPATH='./checkpoints/llama-2-34b-mirror/llava-llama-2-34b-chat-LanguageBind-llava_llama_2-finetune-audio-mirror'
else
    MODELPATH=$2
fi

RESULT_DIR="./results/llama-2-34b-mirror/Image/TextVQA_Mirror_Audio"

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m videollava.eval.anyllava.model_image_qa \
        --model-path $MODELPATH \
        --question-file ./Dataset/Image/textvqa/textvqa_test.jsonl \
        --image-folder ./Dataset/Image \
        --answers-file $RESULT_DIR/${CHUNKS}_${IDX}.jsonl \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \
        --conv-mode llava_llama_2 &
done

wait

output_file=$RESULT_DIR/llama_2_34b_audio_model_image_modality_textvqa_dataset_mirror.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat $RESULT_DIR/${CHUNKS}_${IDX}.jsonl >> "$output_file"
done

python -m videollava.eval.anyllava.image.eval_textvqa \
    --result-file $output_file \
    --output-dir $RESULT_DIR \
