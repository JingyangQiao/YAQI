import argparse
import asyncio
import glob
import json
import math
import multiprocessing
import os
import time
from concurrent.futures import ProcessPoolExecutor
from typing import Dict, List, Optional, Tuple

import torch
from videollava.constants import (
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IM_START_TOKEN,
    IMAGE_TOKEN_INDEX,
)
from videollava.conversation import conv_templates
from videollava.mm_utils import get_model_name_from_path, tokenizer_image_token
from videollava.model.builder import load_pretrained_model
from videollava.utils import disable_torch_init


DATASET_ROOTS = {
    "Image": "./Dataset/Image",
    "Audio": "./Dataset/Audio",
    "Music": "./Dataset/Music",
    "Video": "./Dataset/Video",
    "Text":  "./Dataset/Text",
}

_WORKER_STATE = {}


def case_id_for(entry: Dict, index: int) -> str:
    return str(entry.get("case_id") or entry.get("id") or index)


def dataset_name_for(json_file: str) -> str:
    return os.path.splitext(os.path.basename(json_file))[0]


def resolve_loss_dir(args) -> str:
    return args.loss_jsonl_path or os.path.join(args.sum_path, "loss_jsonl")


def resolve_dataset_root(args) -> str:
    return args.dataset_root or DATASET_ROOTS[args.modality]


def get_conv_mode(model_name: str) -> str:
    if "llama-2" in model_name.lower():
        return "llava_llama_2"
    if "v1" in model_name.lower():
        return "llava_v1"
    if "mpt" in model_name.lower():
        return "mpt"
    return "llava_v1"


def select_entries(data: List[Dict], sample_ratio: float) -> List[Tuple[int, Dict]]:
    count = max(1, int(len(data) * sample_ratio)) if data else 0
    return list(enumerate(data[:count]))


def build_prompt_and_label(entry: Dict, args, model, image_processor) -> Tuple[str, str, List[torch.Tensor]]:
    conv = conv_templates[args.conv_mode].copy()

    if args.modality == "Image":
        media_path = os.path.join(resolve_dataset_root(args), entry["image"])
        inp = entry["conversations"][0]["value"].replace("<image>", "").replace("\n", "")
        label_text = entry["conversations"][1]["value"]
    elif args.modality == "Audio":
        media_path = os.path.join(resolve_dataset_root(args), entry["audio"])
        inp = entry["conversations"][0]["value"].replace("<audio>", "").replace("\n", "")
        label_text = entry["conversations"][1]["value"]
    elif args.modality == "Music":
        media_path = os.path.join(resolve_dataset_root(args), entry["audio"])
        inp = entry["conversations"][0]["value"].replace("<audio>", "").replace("\n", "")
        label_text = entry["conversations"][1]["value"]
    elif args.modality == "Video":
        media_path = os.path.join(resolve_dataset_root(args), entry["video"])
        inp = entry["conversations"][0]["value"].replace("<video>", "").replace("\n", "")
        label_text = entry["conversations"][1]["value"]
    else:
        inp = entry["conversations"][0]["value"].replace("\n", "")
        label_text = entry["conversations"][1]["value"]

    tensors = []
    special_tokens = []
    if args.modality == "Image":
        media_tensor = image_processor.preprocess(media_path, return_tensors="pt")["pixel_values"][0]
        special_tokens.append(DEFAULT_IMAGE_TOKEN)
    elif args.modality in ["Audio", "Music"]:
        media_tensor = image_processor(media_path, return_tensors="pt")["pixel_values"][0]
        special_tokens.append(DEFAULT_IMAGE_TOKEN)
    elif args.modality == "Video":
        media_tensor = image_processor(media_path, return_tensors="pt")["pixel_values"][0]
        special_tokens.extend([DEFAULT_IMAGE_TOKEN] * model.get_video_tower().config.num_frames)
    else:
        pass

    if args.modality != "Text":
        media_tensor = media_tensor.to(model.device, dtype=torch.float16)
        tensors.append(media_tensor)

    if getattr(model.config, "mm_use_im_start_end", False):
        prefix = "".join(DEFAULT_IM_START_TOKEN + token + DEFAULT_IM_END_TOKEN for token in special_tokens)
    else:
        prefix = "".join(special_tokens)

    user_message = inp if args.modality == "Text" else prefix + "\n" + inp
    conv.append_message(conv.roles[0], user_message)
    conv.append_message(conv.roles[1], None)
    return conv.get_prompt(), label_text, tensors


def compute_loss_torch(entry: Dict, args) -> Optional[float]:
    tokenizer = _WORKER_STATE["tokenizer"]
    model = _WORKER_STATE["model"]
    image_processor = _WORKER_STATE["image_processor"]

    prompt, label_text, tensors = build_prompt_and_label(entry, args, model, image_processor)
    full_input = prompt + label_text

    input_ids = tokenizer_image_token(full_input, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
    input_ids = input_ids.unsqueeze(0).to(model.device)

    prompt_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
    prompt_len = prompt_ids.shape[0] if len(prompt_ids.shape) == 1 else prompt_ids.shape[1]

    labels = input_ids.clone()
    labels[:, :prompt_len] = -100

    model_kwargs = {
        "input_ids": input_ids,
        "labels": labels,
        "use_cache": False,
        "output_attentions": False,
        "output_hidden_states": False,
        "return_dict": True,
    }
    if tensors:
        model_kwargs["images"] = tensors

    with torch.inference_mode():
        outputs = model(**model_kwargs)

    loss = outputs.loss
    if torch.isnan(loss).any():
        return None
    return float(loss.item())


def init_torch_worker(args_dict: Dict, device: str) -> None:
    args = argparse.Namespace(**args_dict)
    disable_torch_init()
    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, processor, _ = load_pretrained_model(
        args.model_path,
        None,
        model_name,
        False,
        False,
        device=device,
        cache_dir=None,
    )
    if args.modality == "Image":
        image_processor = processor["image"]
    elif args.modality in ["Audio", "Music"]:
        image_processor = processor["audio"]
    elif args.modality in ["Video"]:
        image_processor = processor["video"]
    else:
        image_processor = None

    _WORKER_STATE.update(
        {
            "tokenizer": tokenizer,
            "model": model,
            "image_processor": image_processor,
        }
    )


def torch_worker(task: Tuple[str, int, Dict, Dict]) -> Tuple[str, str, Optional[float], Optional[str]]:
    dataset_name, index, entry, args_dict = task
    args = argparse.Namespace(**args_dict)
    case_id = case_id_for(entry, index)
    try:
        return dataset_name, case_id, compute_loss_torch(entry, args), None
    except Exception as exc:
        return dataset_name, case_id, None, str(exc)


def chunk_round_robin(items: List, parts: int) -> List[List]:
    chunks = [[] for _ in range(parts)]
    for idx, item in enumerate(items):
        chunks[idx % parts].append(item)
    return chunks


def parse_gpu_ids(args) -> List[str]:
    if args.gpu_ids:
        return [item.strip() for item in args.gpu_ids.split(",") if item.strip()]
    if args.device.startswith("cuda") and torch.cuda.is_available():
        return [str(idx) for idx in range(torch.cuda.device_count())]
    return [args.device]


def normalize_device(args, gpu_id: str) -> str:
    if gpu_id.startswith("cuda") or gpu_id == "cpu":
        return gpu_id
    if args.device.startswith("cuda"):
        return f"cuda:{gpu_id}"
    return args.device


def write_loss_jsonl(path: str, losses: Dict[str, float]) -> None:
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        for case_id, loss in losses.items():
            f.write(json.dumps({case_id: loss}, ensure_ascii=False) + "\n")
    os.replace(tmp_path, path)


def run_torch_backend(args, datasets: Dict[str, List[Tuple[int, Dict]]]) -> Dict[str, Dict[str, float]]:
    all_tasks = []
    args_dict = vars(args).copy()
    for dataset_name, entries in datasets.items():
        for index, entry in entries:
            all_tasks.append((dataset_name, index, entry, args_dict))

    if not all_tasks:
        return {name: {} for name in datasets}

    gpu_ids = parse_gpu_ids(args)
    worker_count = args.num_workers or len(gpu_ids)
    worker_count = max(1, min(worker_count, len(all_tasks)))
    device_ids = [gpu_ids[idx % len(gpu_ids)] for idx in range(worker_count)]
    task_chunks = chunk_round_robin(all_tasks, worker_count)

    results = {name: {} for name in datasets}
    mp_context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=worker_count, mp_context=mp_context) as executor:
        futures = []
        for worker_index, tasks in enumerate(task_chunks):
            if not tasks:
                continue
            device = normalize_device(args, device_ids[worker_index])
            futures.append(
                executor.submit(run_torch_worker_chunk, args_dict, device, tasks)
            )
        for future in futures:
            for dataset_name, case_id, loss, error in future.result():
                if error:
                    print(f"[{dataset_name}/{case_id}] loss failed: {error}")
                    continue
                if loss is not None and not math.isnan(loss):
                    results[dataset_name][case_id] = loss
                    print(f"[{dataset_name}/{case_id}] CE Loss: {loss}")
    return results


def run_torch_worker_chunk(args_dict: Dict, device: str, tasks: List[Tuple[str, int, Dict, Dict]]):
    init_torch_worker(args_dict, device)
    return [torch_worker(task) for task in tasks]


async def run_vllm_backend(args, datasets: Dict[str, List[Tuple[int, Dict]]]) -> Dict[str, Dict[str, float]]:
    from vllm import SamplingParams
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.engine.async_llm_engine import AsyncLLMEngine

    if args.modality != "Text" and not args.allow_vllm_multimodal:
        raise ValueError(
            "vLLM backend is enabled, but this script cannot safely reproduce Video-LLaVA "
            "multimodal loss without vLLM multimodal support for the loaded model. "
            "Use --backend torch, or pass --allow-vllm-multimodal after verifying your vLLM model support."
        )

    model_name = get_model_name_from_path(args.model_path)
    args.conv_mode = args.conv_mode or get_conv_mode(model_name)
    tokenizer, _, _, _ = load_pretrained_model(
        args.model_path,
        None,
        model_name,
        False,
        False,
        device="cpu",
        cache_dir=None,
    )

    engine_args = AsyncEngineArgs(
        model=args.model_path,
        tensor_parallel_size=args.tensor_parallel_size,
        dtype=args.dtype,
        trust_remote_code=True,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )
    engine = AsyncLLMEngine.from_engine_args(engine_args)
    sampling_params = SamplingParams(max_tokens=1, temperature=0.0, prompt_logprobs=1)

    async def one_request(dataset_name: str, index: int, entry: Dict):
        conv = conv_templates[args.conv_mode].copy()
        question = entry["conversations"][0]["value"]
        label_text = entry["conversations"][1]["value"]
        conv.append_message(conv.roles[0], question)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        full_input = prompt + label_text
        full_ids = tokenizer_image_token(full_input, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
        prompt_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
        prompt_len = prompt_ids.shape[0] if len(prompt_ids.shape) == 1 else prompt_ids.shape[1]
        request_id = f"{dataset_name}-{index}"

        final_output = None
        async for output in engine.generate(full_input, sampling_params, request_id=request_id):
            final_output = output

        if final_output is None:
            return dataset_name, case_id_for(entry, index), None

        token_ids = full_ids.tolist()
        logprobs = final_output.prompt_logprobs or []
        selected_logprobs = []
        for pos in range(prompt_len, min(len(token_ids), len(logprobs))):
            token_logprobs = logprobs[pos]
            if not token_logprobs:
                continue
            token_id = int(token_ids[pos])
            token_info = token_logprobs.get(token_id)
            if token_info is not None:
                selected_logprobs.append(float(token_info.logprob))
        if not selected_logprobs:
            return dataset_name, case_id_for(entry, index), None
        return dataset_name, case_id_for(entry, index), -sum(selected_logprobs) / len(selected_logprobs)

    semaphore = asyncio.Semaphore(args.async_concurrency)

    async def guarded(dataset_name: str, index: int, entry: Dict):
        async with semaphore:
            try:
                return await one_request(dataset_name, index, entry)
            except Exception as exc:
                print(f"[{dataset_name}/{case_id_for(entry, index)}] vLLM loss failed: {exc}")
                return dataset_name, case_id_for(entry, index), None

    requests = [
        guarded(dataset_name, index, entry)
        for dataset_name, entries in datasets.items()
        for index, entry in entries
    ]
    outputs = await asyncio.gather(*requests)
    results = {name: {} for name in datasets}
    for dataset_name, case_id, loss in outputs:
        if loss is not None and not math.isnan(loss):
            results[dataset_name][case_id] = float(loss)
            print(f"[{dataset_name}/{case_id}] CE Loss: {loss}")
    return results


def load_datasets(args) -> Dict[str, List[Tuple[int, Dict]]]:
    datasets = {}
    for json_file in glob.glob(os.path.join(args.file_path, "*.json")):
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        datasets[dataset_name_for(json_file)] = select_entries(data, args.sample_ratio)
    return datasets


def save_outputs(args, dataset_losses: Dict[str, Dict[str, float]], dataset_times: Dict[str, float]) -> None:
    os.makedirs(args.sum_path, exist_ok=True)
    loss_dir = resolve_loss_dir(args)
    os.makedirs(loss_dir, exist_ok=True)

    for dataset_name, losses in dataset_losses.items():
        loss_path = os.path.join(loss_dir, f"{dataset_name}_loss.jsonl")
        write_loss_jsonl(loss_path, losses)
        values = list(losses.values())
        mean_loss = sum(values) / len(values) if values else float("nan")
        save_dict = {
            "dataset": dataset_name,
            "mean": mean_loss,
            "loss_jsonl": loss_path,
            "num_losses": len(values),
            "time": dataset_times.get(dataset_name, 0.0),
        }
        with open(os.path.join(args.sum_path, f"{dataset_name}_mean.json"), "w", encoding="utf-8") as f:
            json.dump(save_dict, f, ensure_ascii=False, indent=2)
        print(f"finishing the dataset: {dataset_name}")


def main(args):
    model_name = get_model_name_from_path(args.model_path)
    args.conv_mode = args.conv_mode or get_conv_mode(model_name)
    datasets = load_datasets(args)
    starts = {name: time.time() for name in datasets}

    if args.backend == "vllm":
        dataset_losses = asyncio.run(run_vllm_backend(args, datasets))
    else:
        dataset_losses = run_torch_backend(args, datasets)

    dataset_times = {name: time.time() - start for name, start in starts.items()}
    save_outputs(args, dataset_losses, dataset_times)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--modality", type=str, default="Audio")
    parser.add_argument("--file-path", type=str, required=True)
    parser.add_argument("--sum-path", type=str, required=True)
    parser.add_argument("--loss-jsonl-path", type=str, default=None)
    parser.add_argument("--conv-mode", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dataset-root", type=str, default=None)
    parser.add_argument("--sample-ratio", type=float, default=1.0)
    parser.add_argument("--backend", choices=["torch", "vllm"], default="torch")
    parser.add_argument("--gpu-ids", type=str, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--async-concurrency", type=int, default=64)
    parser.add_argument("--dtype", type=str, default="float16")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--allow-vllm-multimodal", action="store_true")
    args = parser.parse_args()
    main(args)
