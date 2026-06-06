LLM_VERSION="./checkpoints/llava-vicuna-7b-v1.5-LanguageBind-plain-pretrain-audio-mirror/checkpoint-xxxx"
LLM_VERSION_CLEAN="vicuna-7b-v1.5"
IMAGE_MODEL_VERSION="LanguageBind/LanguageBind_Image"
IMAGE_MODEL_VERSION_CLEAN="LanguageBind"
AUDIO_MODEL_VERSION="LanguageBind/LanguageBind_Audio_FT"
AUDIO_MODEL_VERSION_CLEAN="LanguageBind"
PROMPT_VERSION=v1

deepspeed videollava/train/train_mem.py \
    --deepspeed ./scripts/zero2_offload.json \
    --model_name_or_path ${LLM_VERSION} \
    --version ${PROMPT_VERSION} \
    --data_path ./Dataset/Audio/audio_instruct_80k.json ./Dataset/Replay/Vicuna-v1.5-7b/Image.json ./Dataset/Replay/Vicuna-v1.5-7b/Text.json \
    --audio_folder ./Dataset/Audio \
    --audio_tower ${AUDIO_MODEL_VERSION} \
    --image_folder ./Dataset/Image \
    --image_tower ${IMAGE_MODEL_VERSION} \
    --mm_tunable_parts="mm_mlp_adapter,mm_language_model" \
    --mm_vision_select_layer -2 \
    --mm_projector_type mlp2x_gelu \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir ./checkpoints/llava-$LLM_VERSION_CLEAN-$AUDIO_MODEL_VERSION_CLEAN-$PROMPT_VERSION-finetune-audio-mirror \
    --num_train_epochs 12 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 2000 \
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
