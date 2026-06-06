LLM_VERSION="./checkpoints/llava-llama-2-7b-chat-LanguageBind-llava_llama_2-finetune-image-mirror"
LLM_VERSION_CLEAN="llama-2-7b-chat"
IMAGE_MODEL_VERSION="LanguageBind/LanguageBind_Image"
IMAGE_MODEL_VERSION_CLEAN="LanguageBind"
AUDIO_MODEL_VERSION="LanguageBind/LanguageBind_Audio_FT"
AUDIO_MODEL_VERSION_CLEAN="LanguageBind"
PROMPT_VERSION=plain

deepspeed videollava/train/train_mem.py \
    --deepspeed ./scripts/zero3.json \
    --model_name_or_path ${LLM_VERSION} \
    --version ${PROMPT_VERSION} \
    --data_path ./Dataset/LAION-Audio-1M/train.json \
    --audio_folder ./Dataset/LAION-Audio-1M \
    --audio_tower ${AUDIO_MODEL_VERSION} \
    --image_tower ${IMAGE_MODEL_VERSION} \
    --mm_tunable_parts="mm_mlp_adapter" \
    --mm_vision_select_layer -2 \
    --mm_projector_type mlp2x_gelu \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --bf16 True \
    --output_dir ./checkpoints/llava-$LLM_VERSION_CLEAN-$AUDIO_MODEL_VERSION_CLEAN-$PROMPT_VERSION-pretrain-audio-mirror \
    --num_train_epochs 1 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 1000 \
    --save_total_limit 1 \
    --learning_rate 2e-3 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 4096 \
    --gradient_checkpointing True \
    --dataloader_num_workers 8 \
    --lazy_preprocess True \
    --report_to none
