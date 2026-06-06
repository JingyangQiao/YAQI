LLM_VERSION="./checkpoints/llava-vicuna-7b-v1.5-LanguageBind-v1-finetune-text-mirror"
LLM_VERSION_CLEAN="vicuna-7b-v1.5"
IMAGE_MODEL_VERSION="LanguageBind/LanguageBind_Image"
IMAGE_MODEL_VERSION_CLEAN="LanguageBind"
PROMPT_VERSION=v1

deepspeed videollava/train/train_mem.py \
    --deepspeed ./scripts/zero2_offload.json \
    --model_name_or_path ${LLM_VERSION} \
    --version ${PROMPT_VERSION} \
    --data_path ./Dataset/Image/llava_v1_5_image_624k.json ./Dataset/Replay/Vicuna-v1.5-7b/Text.json \
    --image_folder ./Dataset/Image \
    --image_tower ${IMAGE_MODEL_VERSION} \
    --pretrain_mm_mlp_adapter ./checkpoints/llava-vicuna-7b-v1.5-LanguageBind-plain-pretrain-vision-mirror/mm_projector.bin \
    --mm_tunable_parts="mm_mlp_adapter,mm_language_model" \
    --mm_vision_select_layer -2 \
    --mm_projector_type mlp2x_gelu \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir ./checkpoints/llava-$LLM_VERSION_CLEAN-$IMAGE_MODEL_VERSION_CLEAN-$PROMPT_VERSION-finetune-image-mirror \
    --num_train_epochs 6 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 10000 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 4096 --tokenizer_model_max_length 3072 \
    --gradient_checkpointing True \
    --dataloader_num_workers 8 \
    --lazy_preprocess True \
    --report_to none
