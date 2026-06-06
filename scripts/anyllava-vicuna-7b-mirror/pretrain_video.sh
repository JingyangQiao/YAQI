LLM_VERSION="./checkpoints/llava-vicuna-7b-v1.5-LanguageBind-v1-finetune-audio-mirror"
LLM_VERSION_CLEAN="vicuna-7b-v1.5"
IMAGE_MODEL_VERSION="LanguageBind/LanguageBind_Image"
IMAGE_MODEL_VERSION_CLEAN="LanguageBind"
AUDIO_MODEL_VERSION="LanguageBind/LanguageBind_Audio_FT"
AUDIO_MODEL_VERSION_CLEAN="LanguageBind"
VIDEO_MODEL_VERSION="LanguageBind/LanguageBind_Video_merge"
VIDEO_MODEL_VERSION_CLEAN="LanguageBind"
PROMPT_VERSION=plain

deepspeed videollava/train/train_mem.py \
    --deepspeed ./scripts/zero2.json \
    --model_name_or_path ${LLM_VERSION} \
    --version ${PROMPT_VERSION} \
    --data_path ./Dataset/OpenVid/annotations/chat.json \
    --video_folder ./Dataset/OpenVid/OpenVidHD \
    --video_tower ${VIDEO_MODEL_VERSION} \
    --audio_tower ${AUDIO_MODEL_VERSION} \
    --image_tower ${IMAGE_MODEL_VERSION} \
    --mm_tunable_parts="mm_mlp_adapter" \
    --mm_vision_select_layer -2 \
    --mm_projector_type mlp2x_gelu \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --bf16 True \
    --output_dir ./checkpoints/llava-$LLM_VERSION_CLEAN-$VIDEO_MODEL_VERSION_CLEAN-$PROMPT_VERSION-pretrain-video-mirror \
    --num_train_epochs 1 \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 1300 \
    --save_total_limit 1 \
    --learning_rate 1e-3 \
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
