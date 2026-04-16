#!/usr/bin/env python3
import argparse
import json
import logging
import math
import os
import sys
from io import BytesIO
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import torch
import torch.distributed as dist
from einops import rearrange
from PIL import Image
from tqdm import tqdm

from pipeline.inference_pipeline import MatrixGame3Pipeline
from utils.cam_utils import (
    _interpolate_camera_poses_handedness,
    compute_relative_poses,
    get_intrinsics,
    select_memory_idx_fov,
)
from utils.misc import set_seed
from utils.transform import get_video_transform
from utils.utils import (
    build_plucker_from_c2ws,
    build_plucker_from_pose,
    compute_all_poses_from_actions,
    get_extrinsics,
)
from utils.visualize import process_video
from wan.configs import MAX_AREA_CONFIGS, WAN_CONFIGS
from wan.distributed.util import init_distributed_group
from wan.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_DATASET = "/home/builder/workspace/FastVideo/eval/validation_sf.json"
FIRST_CHUNK_FRAMES = 57
CHUNK_FRAMES = 56
CHUNK_OVERLAP = 16
CHUNK_STRIDE = CHUNK_FRAMES - CHUNK_OVERLAP

NOOP_KEYBOARD = [0, 0, 0, 0, 0, 0]
NOOP_MOUSE = [0.0, 0.0]

MOVEMENT_TOKEN_MAP = {
    "-": NOOP_KEYBOARD,
    "W": [1, 0, 0, 0, 0, 0],
    "S": [0, 1, 0, 0, 0, 0],
    "A": [0, 0, 1, 0, 0, 0],
    "D": [0, 0, 0, 1, 0, 0],
    "WA": [1, 0, 1, 0, 0, 0],
    "WD": [1, 0, 0, 1, 0, 0],
    "SD": [0, 1, 0, 1, 0, 0],
}

CAMERA_TOKEN_MAP = {
    "-": NOOP_MOUSE,
    "←": [0.0, -0.1],
    "→": [0.0, 0.1],
    "↑": [0.1, 0.0],
    "↓": [-0.1, 0.0],
    "↑←": [0.1, -0.1],
    "↑→": [0.1, 0.1],
    "↓←": [-0.1, -0.1],
    "↓→": [-0.1, 0.1],
}


def _validate_args(args):
    if args.ulysses_size <= 1 and (args.t5_fsdp or args.dit_fsdp):
        logging.info(
            "Single GPU run detected. Disabling FSDP flags for dataset inference."
        )
        args.t5_fsdp = False
        args.dit_fsdp = False

    if args.size not in MAX_AREA_CONFIGS:
        supported = ", ".join(sorted(MAX_AREA_CONFIGS))
        raise ValueError(f"Unsupported size {args.size!r}. Supported sizes: {supported}")

    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

    if args.num_frames <= 0:
        raise ValueError("--num-frames must be positive.")

    if args.num_inference_steps <= 0:
        raise ValueError("--num-inference-steps must be positive.")

    ckpt_dir = Path(args.ckpt_dir)
    if not ckpt_dir.is_absolute():
        ckpt_dir = REPO_ROOT / ckpt_dir
    args.ckpt_dir = str(ckpt_dir)

    if args.sample_ids:
        args.sample_ids = set(args.sample_ids)


def _parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run Matrix-Game-3.0 on validation_sf.json by converting the dataset's "
            "movement/camera actions into Matrix-Game conditioning tensors. "
            "This script ignores per-sample generation settings in the JSON and "
            "uses CLI flags instead."
        )
    )
    parser.add_argument("--dataset", type=str, default=DEFAULT_DATASET)
    parser.add_argument(
        "--image-root",
        type=str,
        default=None,
        help=(
            "Optional local root used as a fallback when first_frame is not directly "
            "readable, for example a mirror of the S3 key namespace."
        ),
    )
    parser.add_argument(
        "--s3-anon",
        action="store_true",
        default=False,
        help="Use anonymous S3 access when reading s3:// first_frame images.",
    )
    parser.add_argument(
        "--s3-endpoint-url",
        type=str,
        default=None,
        help="Optional custom S3 endpoint URL passed to s3fs.",
    )
    parser.add_argument(
        "--size",
        type=str,
        default="704*1280",
        choices=sorted(MAX_AREA_CONFIGS),
        help="Matrix-Game-3.0 only exposes 704*1280 in this repo.",
    )
    parser.add_argument(
        "--ckpt-dir",
        type=str,
        default="Matrix-Game-3.0",
        help="Checkpoint directory that contains the T5, VAE, and DiT weights.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./output_validation_sf",
        help="Root directory for per-sample outputs.",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument(
        "--sample-ids",
        nargs="*",
        default=None,
        help="Optional list of dataset ids to run.",
    )

    parser.add_argument("--ulysses_size", type=int, default=1)
    parser.add_argument("--t5_fsdp", action="store_true", default=False)
    parser.add_argument("--t5_cpu", action="store_true", default=False)
    parser.add_argument("--dit_fsdp", action="store_true", default=False)
    parser.add_argument("--convert_model_dtype", action="store_true", default=False)

    parser.add_argument("--sample_shift", type=float, default=None)
    parser.add_argument("--sample_guide_scale", type=float, default=None)
    parser.add_argument(
        "--num_frames",
        type=int,
        default=481,
        help="Global frame count for all samples; ignores JSON generation.num_frames.",
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=25,
        help="Global inference step count; ignores JSON generation.num_inference_steps.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Global seed used for all samples; ignores JSON generation.seed.",
    )

    parser.add_argument("--lightvae_pruning_rate", type=float, default=None)
    parser.add_argument("--compile_vae", action="store_true", default=False)
    parser.add_argument(
        "--vae_type",
        type=str,
        default="mg_lightvae_v2",
        choices=["wan", "mg_lightvae", "mg_lightvae_v2"],
    )
    parser.add_argument("--use_int8", action="store_true", default=False)
    parser.add_argument("--verify_quant", action="store_true", default=False)
    parser.add_argument(
        "--fa_version",
        type=str,
        default=None,
        choices=["0", "2", "3"],
        help="Flash Attention version. Use 0 to force SDPA fallback.",
    )
    parser.add_argument("--use_base_model", action="store_true", default=False)
    args = parser.parse_args()
    _validate_args(args)
    return args


def _init_logging(rank):
    level = logging.INFO if rank == 0 else logging.ERROR
    logging.basicConfig(
        level=level,
        format="[%(asctime)s] %(levelname)s: %(message)s",
        handlers=[logging.StreamHandler(stream=sys.stdout)],
    )


def _supported_total_frames(requested_frames):
    if requested_frames <= FIRST_CHUNK_FRAMES:
        return FIRST_CHUNK_FRAMES
    extra = requested_frames - FIRST_CHUNK_FRAMES
    return FIRST_CHUNK_FRAMES + int(math.ceil(extra / CHUNK_STRIDE)) * CHUNK_STRIDE


def _num_iterations_for_total_frames(total_frames):
    return 1 + max(0, (total_frames - FIRST_CHUNK_FRAMES) // CHUNK_STRIDE)


def _resample_tokens(tokens, target_len, pad_token):
    if target_len <= 0:
        return []
    if not tokens:
        return [pad_token] * target_len
    if len(tokens) == target_len:
        return list(tokens)
    src = np.asarray(tokens, dtype=object)
    indices = np.floor(np.arange(target_len) * len(src) / target_len).astype(int)
    indices = np.clip(indices, 0, len(src) - 1)
    return src[indices].tolist()


def _candidate_local_paths(raw_path, image_root):
    candidates = []
    path = Path(raw_path)
    if path.exists():
        candidates.append(path)
    parsed = urlparse(raw_path)
    if image_root is not None:
        root = Path(image_root)
        if parsed.scheme == "s3":
            key_path = Path(parsed.path.lstrip("/"))
            candidates.append(root / key_path)
            candidates.append(root / key_path.name)
        else:
            cleaned = raw_path.lstrip("/")
            candidates.append(root / cleaned)
            candidates.append(root / Path(cleaned).name)

    unique_candidates = []
    seen = set()
    for candidate in candidates:
        candidate_str = str(candidate)
        if candidate_str not in seen:
            unique_candidates.append(candidate)
            seen.add(candidate_str)
    return unique_candidates


def _open_local_image(raw_path, image_root):
    for candidate in _candidate_local_paths(raw_path, image_root):
        if candidate.exists():
            return Image.open(candidate).convert("RGB"), str(candidate)
    return None, None


def _open_s3_image(raw_path, args):
    parsed = urlparse(raw_path)
    if parsed.scheme != "s3":
        return None, None

    try:
        import s3fs
    except ImportError as exc:
        raise RuntimeError(
            "s3fs is required to read s3:// first_frame images. "
            "Run `uv sync` in this repo to install it."
        ) from exc

    client_kwargs = {}
    if args.s3_endpoint_url:
        client_kwargs["endpoint_url"] = args.s3_endpoint_url

    fs = s3fs.S3FileSystem(anon=args.s3_anon, client_kwargs=client_kwargs)
    bucket = parsed.netloc
    key = parsed.path.lstrip("/")
    s3_path = f"{bucket}/{key}"

    with fs.open(s3_path, "rb") as f:
        image_bytes = f.read()
    return Image.open(BytesIO(image_bytes)).convert("RGB"), raw_path


def _load_first_frame(raw_path, image_root, args):
    parsed = urlparse(raw_path)
    if parsed.scheme == "s3":
        try:
            return _open_s3_image(raw_path, args)
        except Exception as s3_error:
            logging.warning(
                "Direct S3 read failed for %s (%s). Trying local fallback paths.",
                raw_path,
                s3_error,
            )

    pil_image, resolved_path = _open_local_image(raw_path, image_root)
    if pil_image is not None:
        return pil_image, resolved_path

    searched = ", ".join(str(candidate) for candidate in _candidate_local_paths(raw_path, image_root))
    if parsed.scheme == "s3":
        searched = f"direct s3:// read, then local fallbacks: {searched or 'none'}"
    else:
        searched = searched or "no candidates"
    raise FileNotFoundError(
        f"Could not resolve first_frame={raw_path!r}. Searched: {searched}"
    )


def _load_samples(args):
    with open(args.dataset, "r", encoding="utf-8") as f:
        samples = json.load(f)

    if args.sample_ids:
        samples = [sample for sample in samples if sample["id"] in args.sample_ids]

    if args.start_index:
        samples = samples[args.start_index :]

    if args.limit is not None:
        samples = samples[: args.limit]

    return samples


def _build_conditions_for_sample(sample, total_frames, device, dtype):
    movement_tokens = _resample_tokens(
        sample.get("actions", {}).get("Movement", []),
        total_frames - 1,
        "-",
    )
    camera_tokens = _resample_tokens(
        sample.get("actions", {}).get("Camera", []),
        total_frames - 1,
        "-",
    )

    keyboard_rows = [NOOP_KEYBOARD]
    mouse_rows = [NOOP_MOUSE]

    for token in movement_tokens:
        if token not in MOVEMENT_TOKEN_MAP:
            raise ValueError(f"Unsupported movement token: {token!r}")
        keyboard_rows.append(MOVEMENT_TOKEN_MAP[token])

    for token in camera_tokens:
        if token not in CAMERA_TOKEN_MAP:
            raise ValueError(f"Unsupported camera token: {token!r}")
        mouse_rows.append(CAMERA_TOKEN_MAP[token])

    keyboard_condition = torch.tensor(keyboard_rows, dtype=torch.float32)
    mouse_condition = torch.tensor(mouse_rows, dtype=torch.float32)

    first_pose = np.zeros(5, dtype=np.float32)
    all_poses = compute_all_poses_from_actions(
        keyboard_condition,
        mouse_condition,
        first_pose=first_pose,
    )
    positions = all_poses[:, :3].tolist()
    rotations = np.concatenate(
        [
            np.zeros((all_poses.shape[0], 1), dtype=np.float32),
            all_poses[:, 3:5],
        ],
        axis=1,
    ).tolist()
    extrinsics_all = get_extrinsics(rotations, positions)

    return (
        keyboard_condition.unsqueeze(0).to(device=device, dtype=dtype),
        mouse_condition.unsqueeze(0).to(device=device, dtype=dtype),
        # Keep camera poses in fp32 because downstream plucker helpers
        # convert them through NumPy, which does not support bfloat16 tensors.
        extrinsics_all.to(device=device, dtype=torch.float32),
    )


def _preprocess_image(pil_image, height, width, device, dtype):
    input_image = torch.from_numpy(np.array(pil_image)).unsqueeze(0).permute(0, 3, 1, 2)
    transform = get_video_transform(height, width, lambda x: 2.0 * x - 1.0)
    input_image = transform(input_image)
    return input_image.transpose(0, 1).unsqueeze(0).to(device=device, dtype=dtype)


class ValidationSFPipeline(MatrixGame3Pipeline):
    def generate_from_conditions(
        self,
        text,
        pil_image,
        keyboard_condition_all,
        mouse_condition_all,
        extrinsics_all,
        total_frames,
        requested_frames,
        max_area,
        shift,
        num_inference_steps,
        guide_scale,
        seed,
        save_name,
        args,
        source_sample,
        resolved_image_path,
    ):
        self._log_flash_attention_config(args)

        mouse_icon = REPO_ROOT / "assets/images/mouse.png"
        vae_cache = [None for _ in range(32)]
        weight_dtype = torch.bfloat16
        height = int(args.size.split("*")[0])
        width = int(args.size.split("*")[1])
        num_iterations = _num_iterations_for_total_frames(total_frames)
        generator = torch.Generator(device=self.device).manual_seed(seed)

        current_image = _preprocess_image(
            pil_image=pil_image,
            height=height,
            width=width,
            device=self.device,
            dtype=weight_dtype,
        )
        cond = self.text_encoder([text], device=self.device)
        neg_cond = self.text_encoder([self.config.sample_neg_prompt], device=self.device)

        h_orig = current_image.shape[-2]
        w_orig = current_image.shape[-1]
        aspect_ratio = h_orig / w_orig
        lat_h = round(
            np.sqrt(max_area * aspect_ratio)
            // self.vae_stride[1]
            // self.patch_size[1]
            * self.patch_size[1]
        )
        lat_w = round(
            np.sqrt(max_area / aspect_ratio)
            // self.vae_stride[2]
            // self.patch_size[2]
            * self.patch_size[2]
        )
        target_h = lat_h * self.vae_stride[1]
        target_w = lat_w * self.vae_stride[2]
        base_k = get_intrinsics(target_h, target_w)

        if self.rank == 0:
            img_cond = (
                self.vae.encode([current_image[0]])[0]
                .unsqueeze(0)
                .to(device=self.device, dtype=weight_dtype)
                .contiguous()
            )
        else:
            img_cond = torch.zeros(
                (1, 48, 1, lat_h, lat_w),
                device=self.device,
                dtype=weight_dtype,
            ).contiguous()

        if dist.is_initialized():
            dist.broadcast(img_cond, src=0)

        max_lat_f = (FIRST_CHUNK_FRAMES - 1) // self.vae_stride[0] + 1
        max_mem_f = 5
        max_total_f = max_lat_f + max_mem_f
        max_seq_len = (
            max_total_f
            * lat_h
            * lat_w
            // (self.patch_size[1] * self.patch_size[2])
        )
        if self.sp_size > 1:
            max_seq_len = int(math.ceil(max_seq_len / self.sp_size)) * self.sp_size

        with torch.no_grad():
            all_latents_list = []
            all_videos_list = []

            for clip_idx in range(num_iterations):
                first_clip = clip_idx == 0
                if self.rank == 0:
                    logging.info(
                        "Sample %s iteration %d/%d",
                        save_name,
                        clip_idx + 1,
                        num_iterations,
                    )

                def align_frame_to_block(frame_idx):
                    return (frame_idx - 1) // 4 * 4 + 1 if frame_idx > 0 else 1

                def get_latent_idx(frame_idx):
                    return (frame_idx - 1) // 4 + 1

                current_end_frame_idx = (
                    FIRST_CHUNK_FRAMES
                    if first_clip
                    else FIRST_CHUNK_FRAMES + clip_idx * CHUNK_STRIDE
                )
                current_start_frame_idx = 0 if first_clip else current_end_frame_idx - CHUNK_FRAMES

                c2ws_chunk = extrinsics_all[current_start_frame_idx:current_end_frame_idx]
                src_indices = np.linspace(
                    current_start_frame_idx,
                    current_end_frame_idx - 1,
                    FIRST_CHUNK_FRAMES if first_clip else CHUNK_FRAMES,
                )
                tgt_len = (
                    (FIRST_CHUNK_FRAMES - 1) // 4 + 1
                    if first_clip
                    else CHUNK_FRAMES // 4
                )
                tgt_indices = np.linspace(
                    0 if first_clip else current_start_frame_idx + 3,
                    current_end_frame_idx - 1,
                    tgt_len,
                )

                plucker = build_plucker_from_c2ws(
                    c2ws_chunk,
                    src_indices=src_indices,
                    tgt_indices=tgt_indices,
                    framewise=True,
                    base_K=base_k,
                    target_h=target_h,
                    target_w=target_w,
                    lat_h=lat_h,
                    lat_w=lat_w,
                )
                plucker_no_mem = plucker

                if first_clip:
                    x_memory = None
                    memory_mouse_condition = None
                    memory_keyboard_condition = None
                    latent_idx = None
                    timestep_memory = None
                else:
                    if self.rank == 0:
                        selected_index_base = [
                            current_end_frame_idx - offset for offset in range(1, 34, 8)
                        ]
                        selected_index = select_memory_idx_fov(
                            extrinsics_all,
                            current_start_frame_idx,
                            selected_index_base,
                            use_gpu=True,
                        )
                        selected_index[-1] = 4
                    else:
                        selected_index = [0] * 5
                        selected_index_base = [
                            current_end_frame_idx - offset for offset in range(1, 34, 8)
                        ]

                    if dist.is_initialized():
                        dist.broadcast_object_list(selected_index, src=0)

                    memory_pluckers = []
                    latent_idx = []
                    for mem_idx, reference_idx in zip(selected_index, selected_index_base):
                        latent_idx.append(get_latent_idx(mem_idx))
                        mem_idx_aligned = align_frame_to_block(mem_idx)
                        mem_block = extrinsics_all[mem_idx_aligned : mem_idx_aligned + 4]
                        mem_src = np.linspace(
                            mem_idx_aligned,
                            mem_idx_aligned + 3,
                            mem_block.shape[0],
                        )
                        mem_tgt = np.array([mem_idx_aligned + 3], dtype=np.float32)
                        mem_pose = _interpolate_camera_poses_handedness(
                            src_indices=mem_src,
                            src_rot_mat=mem_block[:, :3, :3].float().cpu().numpy(),
                            src_trans_vec=mem_block[:, :3, 3].float().cpu().numpy(),
                            tgt_indices=mem_tgt,
                        )
                        reference_pose = extrinsics_all[reference_idx : reference_idx + 1]
                        rel_pair = torch.cat([reference_pose, mem_pose.to(reference_pose.device)], dim=0)
                        rel_pose = compute_relative_poses(rel_pair, framewise=False)[1:2]
                        memory_pluckers.append(
                            build_plucker_from_pose(
                                rel_pose.to(device=self.device),
                                base_K=base_k,
                                target_h=target_h,
                                target_w=target_w,
                                lat_h=lat_h,
                                lat_w=lat_w,
                            )
                        )

                    plucker = torch.cat(memory_pluckers + [plucker], dim=2)
                    src = torch.cat(all_latents_list, dim=2)
                    x_memory = src[:, :, latent_idx]
                    memory_mouse_condition = torch.ones(
                        (1, len(selected_index), 2),
                        device=self.device,
                        dtype=weight_dtype,
                    )
                    memory_keyboard_condition = -torch.ones(
                        (1, len(selected_index), 6),
                        device=self.device,
                        dtype=weight_dtype,
                    )
                    timestep_memory = x_memory.new_zeros(
                        (1, x_memory.shape[2] * x_memory.shape[3] * x_memory.shape[4] // 4)
                    )

                keyboard_condition = keyboard_condition_all[
                    :, current_start_frame_idx:current_end_frame_idx
                ]
                mouse_condition = mouse_condition_all[
                    :, current_start_frame_idx:current_end_frame_idx
                ]
                plucker = plucker.to(device=self.device, dtype=weight_dtype)
                plucker_no_mem = plucker_no_mem.to(device=self.device, dtype=weight_dtype)

                scheduler = FlowUniPCMultistepScheduler()
                scheduler.set_timesteps(
                    num_inference_steps,
                    device=self.device,
                    shift=shift,
                )
                timesteps = scheduler.timesteps

                latent_start_idx = get_latent_idx(current_start_frame_idx)
                latent_end_idx = get_latent_idx(current_end_frame_idx)
                latents = torch.randn(
                    (
                        1,
                        48,
                        latent_end_idx - latent_start_idx,
                        img_cond.shape[-2],
                        img_cond.shape[-1],
                    ),
                    generator=generator,
                    device=self.device,
                    dtype=weight_dtype,
                )
                latents = torch.cat([img_cond, latents[:, :, img_cond.shape[2] :]], dim=2)

                conditions_full = {
                    "mouse_cond": mouse_condition,
                    "keyboard_cond": keyboard_condition,
                    "context": cond,
                    "plucker_emb": plucker,
                    "x_memory": x_memory,
                    "timestep_memory": timestep_memory,
                    "keyboard_cond_memory": memory_keyboard_condition,
                    "mouse_cond_memory": memory_mouse_condition,
                    "memory_latent_idx": latent_idx,
                    "predict_latent_idx": (latent_start_idx, latent_end_idx),
                    "fa_version": self.fa_version,
                }
                conditions_null = {
                    "mouse_cond": torch.ones_like(mouse_condition).to(
                        device=self.device, dtype=weight_dtype
                    ),
                    "keyboard_cond": -torch.ones_like(keyboard_condition).to(
                        device=self.device, dtype=weight_dtype
                    ),
                    "context": neg_cond,
                    "plucker_emb": plucker_no_mem,
                    "x_memory": None,
                    "timestep_memory": None,
                    "keyboard_cond_memory": None,
                    "mouse_cond_memory": None,
                    "memory_latent_idx": None,
                    "predict_latent_idx": (latent_start_idx, latent_end_idx),
                }

                for step_idx, timestep_value in enumerate(
                    tqdm(timesteps, disable=(self.rank != 0))
                ):
                    timestep = latents.new_full(
                        (latents.shape[2], latents.shape[3] * latents.shape[4] // 4),
                        timestep_value,
                    )
                    timestep[: img_cond.shape[2]].zero_()
                    timestep = timestep.flatten().unsqueeze(0)

                    model_kwargs = {
                        "x": latents,
                        "t": timestep,
                        "seq_len": max_seq_len,
                        **conditions_full,
                    }
                    model_kwargs_null = {
                        "x": latents,
                        "t": timestep,
                        "seq_len": max_seq_len,
                        **conditions_null,
                    }

                    if args.use_base_model:
                        noise_pred_full = self.model(**model_kwargs)
                        noise_pred_null = self.model(**model_kwargs_null)
                        noise_pred = noise_pred_null + guide_scale * (
                            noise_pred_full - noise_pred_null
                        )
                    else:
                        noise_pred = self.model(**model_kwargs)

                    if args.use_int8 and args.verify_quant and step_idx == 0 and self.rank == 0:
                        logging.info(
                            "Verification stats for %s: mean=%.6f std=%.6f",
                            save_name,
                            noise_pred.mean().item(),
                            noise_pred.std().item(),
                        )

                    latents = scheduler.step(
                        noise_pred,
                        timestep_value,
                        latents,
                        return_dict=False,
                    )[0]
                    latents = torch.cat([img_cond, latents[:, :, img_cond.shape[2] :]], dim=2)

                img_cond = latents[:, :, -4:]
                denoised_pred = latents if first_clip else latents[:, :, -10:]

                if self.rank == 0:
                    do_compile = bool(args.compile_vae and clip_idx >= 1)
                    vae_profiler = {}
                    video, vae_cache = self.vae.stream_decode(
                        denoised_pred.to(dtype=self.vae.dtype),
                        vae_cache,
                        first_chunk=first_clip,
                        segment_size=int(os.environ.get("WAN_VAE_SEGMENT_SIZE", "4")),
                        profiler=vae_profiler,
                        compile_decoder=do_compile,
                    )
                    all_videos_list.append(video.cpu())

                all_latents_list.append(denoised_pred)

            if self.rank == 0 and all_videos_list:
                concatenated_video = np.ascontiguousarray(
                    (
                        (
                            rearrange(
                                torch.concat(all_videos_list, dim=2)[0],
                                "C T H W -> T H W C",
                            ).float()
                            + 1
                        )
                        * 127.5
                    )
                    .clip(0, 255)
                    .numpy()
                    .astype(np.uint8)
                )
                trimmed_video = concatenated_video[:requested_frames]
                keyboard_np = (
                    keyboard_condition_all.squeeze(0)[:requested_frames].float().cpu().numpy()
                )
                mouse_np = (
                    mouse_condition_all.squeeze(0)[:requested_frames].float().cpu().numpy()
                )
                output_path = Path(self.output_dir) / f"{save_name}.mp4"
                process_video(
                    trimmed_video,
                    str(output_path),
                    (keyboard_np, mouse_np),
                    str(mouse_icon),
                    mouse_scale=0.2,
                    default_frame_res=(height, width),
                )

                metadata = {
                    "id": save_name,
                    "resolved_image": str(resolved_image_path),
                    "requested_frames": requested_frames,
                    "generated_frames_before_trim": int(concatenated_video.shape[0]),
                    "size": args.size,
                    "num_iterations": num_iterations,
                    "num_inference_steps": num_inference_steps,
                    "seed": seed,
                    "sample_shift": shift,
                    "sample_guide_scale": guide_scale,
                    "use_base_model": bool(args.use_base_model),
                    "source_sample": source_sample,
                }
                metadata_path = Path(self.output_dir) / f"{save_name}.json"
                metadata_path.write_text(
                    json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
                    encoding="utf-8",
                )


def main():
    args = _parse_args()

    rank = int(os.getenv("RANK", 0))
    world_size = int(os.getenv("WORLD_SIZE", 1))
    local_rank = int(os.getenv("LOCAL_RANK", 0))
    device_id = local_rank

    _init_logging(rank)
    set_seed(args.seed if args.seed is not None else 42)

    if world_size > 1:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            rank=rank,
            world_size=world_size,
        )
    else:
        if args.t5_fsdp or args.dit_fsdp:
            raise ValueError("FSDP flags require torchrun with more than one process.")
        if args.ulysses_size > 1:
            raise ValueError("ulysses_size > 1 requires torchrun with more than one process.")

    if args.ulysses_size > 1:
        if args.ulysses_size != world_size:
            raise ValueError("ulysses_size must match WORLD_SIZE for this script.")
        init_distributed_group()

    cfg = WAN_CONFIGS["matrix_game3"]
    if args.sample_shift is None:
        args.sample_shift = cfg.sample_shift
    if args.sample_guide_scale is None:
        args.sample_guide_scale = cfg.sample_guide_scale

    samples = _load_samples(args)
    if not samples:
        raise ValueError("No samples selected from the dataset.")

    output_root = Path(args.output_dir)
    if not output_root.is_absolute():
        output_root = REPO_ROOT / output_root
    args.output_dir = str(output_root)
    if rank == 0:
        output_root.mkdir(parents=True, exist_ok=True)
        logging.info("Selected %d samples", len(samples))
        logging.info("Checkpoint dir: %s", args.ckpt_dir)
        logging.info("Output root: %s", output_root)
        logging.info(
            "Using CLI overrides for all samples: size=%s num_frames=%d "
            "num_inference_steps=%d seed=%d",
            args.size,
            args.num_frames,
            args.num_inference_steps,
            args.seed,
        )

    pipeline = ValidationSFPipeline(
        config=cfg,
        checkpoint_dir=args.ckpt_dir,
        device_id=device_id,
        rank=rank,
        t5_fsdp=args.t5_fsdp,
        dit_fsdp=args.dit_fsdp,
        use_sp=(args.ulysses_size > 1),
        t5_cpu=args.t5_cpu,
        convert_model_dtype=args.convert_model_dtype,
        args=args,
        fa_version=args.fa_version,
        use_base_model=args.use_base_model,
    )

    for sample_idx, sample in enumerate(samples, start=1):
        sample_id = sample["id"]
        requested_frames = int(args.num_frames)
        total_frames = _supported_total_frames(requested_frames)
        sample_seed = int(args.seed)
        sample_steps = int(args.num_inference_steps)

        if rank == 0:
            logging.info(
                "[%d/%d] Running %s | cli_frames=%d rounded_frames=%d | cli_size=%s",
                sample_idx,
                len(samples),
                sample_id,
                requested_frames,
                total_frames,
                args.size,
            )

        pil_image, resolved_image_path = _load_first_frame(
            sample["first_frame"],
            args.image_root,
            args,
        )
        keyboard_condition_all, mouse_condition_all, extrinsics_all = _build_conditions_for_sample(
            sample=sample,
            total_frames=total_frames,
            device=pipeline.device,
            dtype=torch.bfloat16,
        )

        pipeline.output_dir = str(output_root)
        args.output_dir = str(output_root)

        pipeline.generate_from_conditions(
            text=sample["prompt"],
            pil_image=pil_image,
            keyboard_condition_all=keyboard_condition_all,
            mouse_condition_all=mouse_condition_all,
            extrinsics_all=extrinsics_all,
            total_frames=total_frames,
            requested_frames=requested_frames,
            max_area=MAX_AREA_CONFIGS[args.size],
            shift=args.sample_shift,
            num_inference_steps=sample_steps,
            guide_scale=args.sample_guide_scale,
            seed=sample_seed,
            save_name=sample_id,
            args=args,
            source_sample=sample,
            resolved_image_path=resolved_image_path,
        )

        if dist.is_initialized():
            dist.barrier()

    if rank == 0:
        logging.info("Finished dataset inference.")

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
