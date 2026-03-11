import tempfile
from dataclasses import asdict
from pathlib import Path
from zipfile import ZipFile
from typing import Tuple

import gradio as gr
import numpy as np
import torch
from accelerate import Accelerator
from einops import repeat
from gradio import Blocks
from safetensors.torch import load_file
from sanitize_filename import sanitize

from osu_fusion.data.decode import Metadata, decode_sequence
from osu_fusion.data.encode import SequenceEncoding
from osu_fusion.data.prepare_data import load_audio
from osu_fusion.models.diffusion_dit import (
    DiTConfig_S,
    DiTConfig_M,
    DiTConfig_L,
    OsuFusionDiT,
)

MODEL_CONFIGS = {
    "s": DiTConfig_S,
    "m": DiTConfig_M,
    "l": DiTConfig_L,
}

VERSION_TEMPLATE = "{version_name} - batch {batch_number}_{batch_size}"

global_model = None
global_accelerator = None
global_temp_dir = tempfile.TemporaryDirectory()


def strip_padding(seq: np.ndarray) -> np.ndarray:
    """Remove TYPE_PAD tokens from the end of a decoded sequence (N, D)."""
    pad_idx = SequenceEncoding.TYPE_PAD
    type_logits = seq[:, 6:18]
    event_types = np.argmax(type_logits, axis=1) + 6

    # Find last non-pad event
    non_pad = np.where(event_types != pad_idx)[0]
    if len(non_pad) == 0:
        return seq[:0]
    return seq[: non_pad[-1] + 1]


def create_model_from_checkpoint(model_path: str, model_size: str) -> OsuFusionDiT:
    config = MODEL_CONFIGS[model_size]
    model = OsuFusionDiT(**asdict(config))

    if model_path.endswith(".pt"):
        checkpoint = torch.load(model_path, weights_only=False)
        model.dit.load_state_dict(checkpoint["dit_state_dict"])
        if "length_predictor_state_dict" in checkpoint:
            model.length_predictor.load_state_dict(checkpoint["length_predictor_state_dict"])
    else:
        state_dict = load_file(model_path)
        model.dit.load_state_dict(state_dict)

    return model.eval()


def load_model(model_path: str, model_size: str, mixed_precision: str) -> str:
    global global_model, global_accelerator
    global_accelerator = Accelerator(mixed_precision=mixed_precision)
    global_model = create_model_from_checkpoint(model_path, model_size)
    global_model = global_accelerator.prepare(global_model)

    model_dtype = {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }.get(global_accelerator.mixed_precision, torch.float32)
    global_model = global_model.to(dtype=model_dtype)

    return "Model loaded successfully!"


def generate_beatmap(
    music_path: str,
    cs: float,
    ar: float,
    od: float,
    hp: float,
    sr: float,
    slider_multiplier: float,
    slider_tick_rate: float,
    music_artists: str,
    music_title: str,
    bpm: float,
    bpm_enable: bool,
    allow_beat_snap: bool,
    version_name: str,
    batch_size: int,
    cfg: float,
    steps: int,
) -> Tuple[dict, str]:
    global global_model, global_accelerator

    if global_model is None or global_accelerator is None:
        return None, "Error: Model not loaded. Please load the model first."

    global_model.sampling_timesteps = steps

    device = global_accelerator.device
    dtype = {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }.get(global_accelerator.mixed_precision, torch.float32)

    # Load audio and prepare inputs
    a = load_audio(music_path)
    context = np.array([cs, ar, od, hp, sr, slider_multiplier, slider_tick_rate], dtype=np.float32)

    a_tensor = torch.from_numpy(a).unsqueeze(0).to(device, dtype)
    c_tensor = torch.from_numpy(context).unsqueeze(0).to(device, dtype)

    a_tensor = repeat(a_tensor, "1 n d -> b n d", b=batch_size)
    c_tensor = repeat(c_tensor, "1 c -> b c", b=batch_size)

    # Generate — length is auto-predicted
    with torch.inference_mode(), global_accelerator.autocast():
        generated = global_model.sample(a_tensor, c_tensor, cond_scale=cfg)

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
            # Strip padding tokens from the output
            seq = strip_padding(seq)

            metadata.version = VERSION_TEMPLATE.format(
                version_name=version_name,
                batch_number=i + 1,
                batch_size=batch_size,
            )
            beatmap = decode_sequence(
                metadata,
                seq,
                bpm=bpm if bpm_enable else None,
                allow_beat_snap=allow_beat_snap,
            )
            mapset_archive.writestr(
                f"{metadata.artist} - {metadata.title} (OsuFusion) [{metadata.version}].osu",
                beatmap,
            )

    return gr.update(value=mapset_path, visible=True), f"Beatmap generated successfully: {mapset_path}"


def update_bpm_interactivity(bpm_enable: bool) -> dict:
    return gr.Slider(interactive=bpm_enable)


def gradio_interface() -> Blocks:
    with gr.Blocks() as app:
        gr.Markdown("# OsuFusion Beatmap Generator")

        with gr.Row():
            model_path = gr.Textbox(label="Model Path")
            model_size = gr.Dropdown(["s", "m", "l"], value="s", label="Model Size")
            mixed_precision = gr.Dropdown(["no", "fp16", "bf16"], value="bf16", label="Mixed Precision")

        load_button = gr.Button("Load Model")
        load_output = gr.Textbox(label="Load Status")

        load_button.click(
            load_model,
            inputs=[model_path, model_size, mixed_precision],
            outputs=load_output,
        )

        with gr.Row():
            music_path = gr.File(label="Music Path")

        with gr.Row():
            cs = gr.Slider(0, 10, value=4.0, label="CS")
            ar = gr.Slider(0, 10, value=9.5, label="AR")
            od = gr.Slider(0, 10, value=9.5, label="OD")
            hp = gr.Slider(0, 10, value=4.0, label="HP")
            sr = gr.Slider(0, 10, value=6.0, label="SR")

        with gr.Row():
            slider_multiplier = gr.Slider(0.4, 3.6, value=1.4, label="Slider Multiplier")
            slider_tick_rate = gr.Slider(0.5, 8, value=1.0, step=0.5, label="Slider Tick Rate")

        with gr.Row():
            music_artists = gr.Textbox(label="Music Artists", value="Unknown Artists")
            music_title = gr.Textbox(label="Music Title", value="Unknown Title")
            version_name = gr.Textbox(label="Version Name", value="Unknown Version")
            with gr.Column():
                bpm_enable = gr.Checkbox(value=False, label="Enable BPM")
                allow_beat_snap = gr.Checkbox(value=False, label="Allow Beat Snap")
                bpm = gr.Slider(1, 300, value=1, step=1, label="BPM", interactive=False)

        with gr.Row():
            batch_size = gr.Slider(1, 10, value=1, step=1, label="Batch Size")
            steps = gr.Slider(1, 100, value=35, step=1, label="Steps")
            cfg = gr.Slider(0, 10, value=2.0, label="CFG")

        generate_button = gr.Button("Generate Beatmap")
        output_file = gr.File(label="Generated Beatmap", interactive=False)
        output_text = gr.Textbox(label="Generation Status")

        bpm_enable.change(update_bpm_interactivity, inputs=[bpm_enable], outputs=[bpm])

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
                music_artists,
                music_title,
                bpm,
                bpm_enable,
                allow_beat_snap,
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
