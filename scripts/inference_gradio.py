import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Tuple
from zipfile import ZipFile

import gradio as gr
import numpy as np
import torch
from accelerate import Accelerator
from einops import repeat
from gradio import Blocks
from safetensors.torch import load_file
from sanitize_filename import sanitize

from osu_fusion.data.const import ERA_LABELS
from osu_fusion.data.decode import Metadata, decode_sequence

# from osu_fusion.data.descriptors import DESCRIPTOR_ANCESTORS, DESCRIPTOR_NAME_TO_IDX, DESCRIPTOR_TAGS, NUM_DESCRIPTORS
from osu_fusion.data.prepare_data import load_audio
from osu_fusion.models.diffusion_dit import (
    DiTConfig_L,
    DiTConfig_M,
    DiTConfig_S,
    OsuFusionDiT,
)

MODEL_CONFIGS = {
    "s": DiTConfig_S,
    "m": DiTConfig_M,
    "l": DiTConfig_L,
}

ERA_CHOICES = {label: float(idx) for idx, label in enumerate(ERA_LABELS)}

VERSION_TEMPLATE = "{version_name} - batch {batch_number}_{batch_size}"

global_model = None
global_accelerator = None
# global_num_mappers = 0
# global_mapper_index = {}  # {user_id_str: index}
global_temp_dir = tempfile.TemporaryDirectory()


def create_model_from_checkpoint(model_path: str, model_size: str) -> OsuFusionDiT:
    config = MODEL_CONFIGS[model_size]
    # config.num_mappers = num_mappers
    model = OsuFusionDiT(**asdict(config))

    if model_path.endswith(".pt"):
        checkpoint = torch.load(model_path, weights_only=False)
        model.dit.load_state_dict(checkpoint["dit_state_dict"])
    else:
        state_dict = load_file(model_path)
        model.dit.load_state_dict(state_dict)

    return model.eval()


def load_model(model_path: str, model_size: str, mixed_precision: str) -> str:
    global global_model, global_accelerator
    # global global_num_mappers, global_mapper_index

    # global_num_mappers = 0
    # global_mapper_index = {}
    # if mapper_index_path and Path(mapper_index_path).exists():
    #     with open(mapper_index_path, "r") as f:
    #         global_mapper_index = json.load(f)
    #     global_num_mappers = max(int(v) for v in global_mapper_index.values()) + 1
    #     print(f"Loaded mapper index: {global_num_mappers} mappers")

    global_accelerator = Accelerator(mixed_precision=mixed_precision)
    global_model = create_model_from_checkpoint(model_path, model_size)
    global_model = global_accelerator.prepare(global_model)

    model_dtype = {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }.get(global_accelerator.mixed_precision, torch.float32)
    global_model = global_model.to(dtype=model_dtype)

    return "Model loaded!"


# def expand_with_ancestors(selected_tags: list) -> list:
#     expanded = set(selected_tags)
#     for tag in selected_tags:
#         tag = tag.strip()
#         if tag in DESCRIPTOR_NAME_TO_IDX:
#             idx = DESCRIPTOR_NAME_TO_IDX[tag]
#             for ancestor_idx in DESCRIPTOR_ANCESTORS.get(idx, [idx]):
#                 expanded.add(DESCRIPTOR_TAGS[ancestor_idx])
#     return [t for t in DESCRIPTOR_TAGS if t in expanded]


# def build_descriptor_vector(selected_tags: list) -> Optional[torch.Tensor]:
#     if not selected_tags:
#         return None
#     vec = torch.zeros(NUM_DESCRIPTORS, dtype=torch.float32)
#     has_any = False
#     for tag in selected_tags:
#         tag = tag.strip()
#         if tag in DESCRIPTOR_NAME_TO_IDX:
#             vec[DESCRIPTOR_NAME_TO_IDX[tag]] = 1.0
#             has_any = True
#     return vec if has_any else None


# def build_mapper_vector(mapper_id_str: str) -> Optional[torch.Tensor]:
#     if not mapper_id_str.strip() or global_num_mappers == 0:
#         return None
#     vec = torch.zeros(global_num_mappers + 1, dtype=torch.float32)
#     has_any = False
#     for uid_str in mapper_id_str.split(","):
#         uid_str = uid_str.strip()
#         if uid_str in global_mapper_index:
#             vec[int(global_mapper_index[uid_str])] = 1.0
#             has_any = True
#         else:
#             vec[global_num_mappers] = 1.0  # unknown
#             has_any = True
#     return vec if has_any else None


def generate_beatmap(
    music_path: str,
    cs: float,
    ar: float,
    od: float,
    hp: float,
    sr: float,
    slider_multiplier: float,
    slider_tick_rate: float,
    era: str,
    # selected_descriptors: list,
    # mapper_ids: str,
    music_artists: str,
    music_title: str,
    version_name: str,
    batch_size: int,
    cfg: float,
    steps: int,
) -> Tuple[dict, str]:
    global global_model, global_accelerator

    if global_model is None or global_accelerator is None:
        return None, "Error: Model not loaded."

    global_model.sampling_timesteps = steps

    device = global_accelerator.device
    dtype = {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }.get(global_accelerator.mixed_precision, torch.float32)

    # Load audio
    a = load_audio(music_path)
    era_val = ERA_CHOICES.get(era, 3.0)
    context = np.array([cs, ar, od, hp, sr, slider_multiplier, slider_tick_rate, era_val], dtype=np.float32)

    a_tensor = torch.from_numpy(a).unsqueeze(0).to(device, dtype)
    c_tensor = torch.from_numpy(context).unsqueeze(0).to(device, dtype)

    # desc_vec = build_descriptor_vector(selected_descriptors)
    # desc_tensor = desc_vec.unsqueeze(0).to(device, dtype) if desc_vec is not None else None

    # mapper_vec = build_mapper_vector(mapper_ids)
    # mapper_tensor = mapper_vec.unsqueeze(0).to(device, dtype) if mapper_vec is not None else None

    # Batch
    a_tensor = repeat(a_tensor, "1 n d -> b n d", b=batch_size)
    c_tensor = repeat(c_tensor, "1 c -> b c", b=batch_size)
    # if desc_tensor is not None:
    #     desc_tensor = repeat(desc_tensor, "1 d -> b d", b=batch_size)
    # if mapper_tensor is not None:
    #     mapper_tensor = repeat(mapper_tensor, "1 d -> b d", b=batch_size)

    # Generate
    with torch.inference_mode(), global_accelerator.autocast():
        generated = global_model.sample(
            a_tensor,
            c_tensor,
            # descriptors=desc_tensor,
            # mappers=mapper_tensor,
            cond_scale=cfg,
        )

    metadata = Metadata(
        Path(music_path).name,
        music_title,
        music_artists,
        version_name,
        cs,
        ar,
        od,
        hp,
        slider_multiplier,
        slider_tick_rate,
    )

    mapset_path = f"{metadata.artist} - {metadata.title} (OsuFusion) [{metadata.version}].osz"
    mapset_path = sanitize(mapset_path)
    mapset_path = str(Path(global_temp_dir.name) / mapset_path)

    with ZipFile(mapset_path, "w") as mapset_archive:
        mapset_archive.write(music_path, metadata.audio_filename)

        sequences = generated.cpu().detach().float().numpy()
        for i, seq in enumerate(sequences):
            metadata.version = VERSION_TEMPLATE.format(
                version_name=version_name,
                batch_number=i + 1,
                batch_size=batch_size,
            )
            beatmap = decode_sequence(metadata, seq)
            mapset_archive.writestr(
                f"{metadata.artist} - {metadata.title} (OsuFusion) [{metadata.version}].osu",
                beatmap,
            )

    return gr.update(value=mapset_path, visible=True), f"Beatmap generated successfully: {mapset_path}"


def gradio_interface() -> Blocks:
    with gr.Blocks(title="OsuFusion Generator") as app:
        gr.Markdown("# 🎵 OsuFusion Beatmap Generator")

        with gr.Row():
            model_path = gr.Textbox(label="Model Path")
            model_size = gr.Dropdown(["s", "m", "l"], value="s", label="Model Size")
            mixed_precision = gr.Dropdown(["no", "fp16", "bf16"], value="bf16", label="Mixed Precision")
            # mapper_index_path = gr.Textbox(label="Mapper Index JSON (optional)", value="")

        load_button = gr.Button("Load Model")
        load_output = gr.Textbox(label="Load Status")

        load_button.click(
            load_model,
            inputs=[model_path, model_size, mixed_precision],
            outputs=load_output,
        )

        with gr.Row():
            music_path = gr.File(label="Music File")

        with gr.Row():
            cs = gr.Slider(0, 10, value=4.0, label="CS")
            ar = gr.Slider(0, 10, value=9.5, label="AR")
            od = gr.Slider(0, 10, value=9.5, label="OD")
            hp = gr.Slider(0, 10, value=4.0, label="HP")
            sr = gr.Slider(0, 10, value=6.0, label="SR")

        with gr.Row():
            slider_multiplier = gr.Slider(0.4, 3.6, value=1.4, label="Slider Multiplier")
            slider_tick_rate = gr.Slider(0.5, 8, value=1.0, step=0.5, label="Slider Tick Rate")
            era = gr.Dropdown(
                choices=ERA_LABELS,
                value="current",
                label="Era",
                info="classic (≤2012), transitional (2013-2016), modern (2017-2020), current (2021+)",
            )

        # with gr.Row():
        #     selected_descriptors = gr.CheckboxGroup(
        #         choices=DESCRIPTOR_TAGS,
        #         label="Style Descriptors",
        #         info="Select mapping style tags (ancestors are auto-selected)",
        #     )

        # selected_descriptors.change(
        #     fn=expand_with_ancestors,
        #     inputs=[selected_descriptors],
        #     outputs=[selected_descriptors],
        # )
        # with gr.Row():
        #     mapper_ids = gr.Textbox(
        #         label="Mapper User IDs (comma-separated)",
        #         value="",
        #         placeholder="e.g., 4452992,896613",
        #         info="Enter osu! user IDs for mapper style conditioning",
        #     )

        with gr.Row():
            music_artists = gr.Textbox(label="Artist", value="Unknown Artists")
            music_title = gr.Textbox(label="Title", value="Unknown Title")
            version_name = gr.Textbox(label="Version Name", value="Unknown Version")

        with gr.Row():
            batch_size = gr.Slider(1, 10, value=1, step=1, label="Batch Size")
            steps = gr.Slider(1, 100, value=35, step=1, label="Steps")
            cfg = gr.Slider(0, 10, value=2.0, label="CFG")

        generate_button = gr.Button("🎮 Generate Beatmap", variant="primary")
        output_file = gr.File(label="Generated Beatmap", interactive=False)
        output_text = gr.Textbox(label="Status")

        generate_button.click(
            generate_beatmap,
            inputs=[
                music_path,
                cs,
                ar,
                od,
                hp,
                sr,
                slider_multiplier,
                slider_tick_rate,
                era,
                # selected_descriptors,
                # mapper_ids,
                music_artists,
                music_title,
                version_name,
                batch_size,
                cfg,
                steps,
            ],
            outputs=[output_file, output_text],
        )

    return app


if __name__ == "__main__":
    app = gradio_interface()
    app.launch(share=True)
