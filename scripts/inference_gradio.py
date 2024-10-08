import tempfile
from pathlib import Path
from typing import Dict, Tuple, Union
from zipfile import ZipFile

import gradio as gr
import librosa
import numpy as np
import torch
from accelerate import Accelerator
from einops import repeat
from gradio import Blocks
from safetensors.torch import load_file
from sanitize_filename import sanitize

from osu_fusion.data.const import BEATMAP_DIM
from osu_fusion.data.decode import Metadata, decode_beatmap
from osu_fusion.data.prepare_data import HOP_LENGTH, SR, load_audio, normalize_context
from osu_fusion.models.diffusion import OsuFusion as DiffusionOsuFusion
from osu_fusion.models.rectified_flow import OsuFusion as RectifiedFlowOsuFusion

Model = Union[DiffusionOsuFusion, RectifiedFlowOsuFusion]

VERSION_TEMPLATE = "{version_name} - batch {batch_number}_{batch_size}"

# Global variables to store the model and accelerator
global_model = None
global_accelerator = None
global_temp_dir = tempfile.TemporaryDirectory()


def create_model_from_checkpoint(model_path: str, model_type: str) -> Model:
    if model_path.endswith(".pt"):
        checkpoint = torch.load(model_path)
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = load_file(model_path)

    model_class = DiffusionOsuFusion if model_type == "diffusion" else RectifiedFlowOsuFusion
    model = model_class(128)
    model.load_state_dict(state_dict)
    return model.eval()


def create_input(
    audio_path: str,
    cs: float,
    ar: float,
    od: float,
    hp: float,
    sr: float,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    audio = load_audio(audio_path)
    context = normalize_context(np.array([cs, ar, od, hp, sr], dtype=np.float32))

    audio = torch.from_numpy(audio).to(device=device, dtype=dtype)
    context = torch.from_numpy(context).to(device=device, dtype=dtype)

    audio = repeat(audio, "d n -> b d n", b=batch_size)
    context = repeat(context, "c -> b c", b=batch_size)

    n = audio.shape[-1]
    x = torch.randn((batch_size, BEATMAP_DIM, n), device=device, dtype=dtype)

    return x, audio, context


def load_model(model_path: str, model_type: str, mixed_precision: str) -> str:
    global global_model, global_accelerator
    global_accelerator = Accelerator(mixed_precision=mixed_precision)
    global_model = create_model_from_checkpoint(model_path, model_type)
    global_model = global_accelerator.prepare(global_model)

    model_dtype = torch.float32
    if global_accelerator.mixed_precision == "fp16":
        model_dtype = torch.float16
    elif global_accelerator.mixed_precision == "bf16":
        model_dtype = torch.bfloat16
    global_model = global_model.to(dtype=model_dtype)

    return "Model loaded successfully!"


def generate_beatmap(
    music_path: str,
    cs: float,
    ar: float,
    od: float,
    hp: float,
    sr: float,
    music_artists: str,
    music_title: str,
    bpm: float,
    bpm_enable: bool,
    allow_beat_snap: bool,
    version_name: str,
    batch_size: int,
    cfg: float,
    steps: int,
) -> Tuple[Dict, str]:
    global global_model, global_accelerator
    global_model.sampling_timesteps = steps

    if global_model is None or global_accelerator is None:
        return None, "Error: Model not loaded. Please load the model first."

    device = global_accelerator.device
    dtype = torch.float32
    if global_accelerator.mixed_precision == "fp16":
        dtype = torch.float16
    elif global_accelerator.mixed_precision == "bf16":
        dtype = torch.bfloat16
    x, audio, context = create_input(
        music_path,
        cs,
        ar,
        od,
        hp,
        sr,
        batch_size,
        device,
        dtype,
    )

    audio_lat = global_model.unet.encode_audio(audio)
    context_prep = global_model.unet.prepare_condition(context)
    context_uncond_prep = global_model.unet.prepare_condition(context, cond_drop_prob=1.0)
    generated = global_model.sample(
        x.shape[-1],
        audio_lat,
        context_prep,
        c_uncond_prep=context_uncond_prep,
        x=x,
        cond_scale=cfg,
    )

    frame_times = frame_times = (
        librosa.frames_to_time(np.arange(generated.shape[-1]), sr=SR, hop_length=HOP_LENGTH) * 1000
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
    )

    mapset_path = f"{metadata.artist} - {metadata.title} (OsuFusion) [{metadata.version}].osz"
    mapset_path = sanitize(mapset_path)
    mapset_path = str(Path(global_temp_dir.name) / mapset_path)

    with ZipFile(mapset_path, "w") as mapset_archive:
        mapset_archive.write(music_path, metadata.audio_filename)

        signals = generated.cpu().detach().numpy()
        for i, signal in enumerate(signals):
            metadata.version = VERSION_TEMPLATE.format(
                version_name=version_name,
                batch_number=i + 1,
                batch_size=batch_size,
            )
            beatmap = decode_beatmap(metadata, signal, frame_times, bpm if bpm_enable else None, allow_beat_snap)
            mapset_archive.writestr(
                f"{metadata.artist} - {metadata.title} (OsuFusion) [{metadata.version}].osu",
                beatmap,
            )

    return gr.update(value=mapset_path, visible=True), f"Beatmap generated successfully: {mapset_path}"


def update_bpm_interactivity(bpm_enable: bool) -> dict:
    return gr.Slider(interactive=bpm_enable)


# Define the Gradio interface
def gradio_interface() -> Blocks:
    with gr.Blocks() as app:
        gr.Markdown("# OsuFusion Beatmap Generator")

        with gr.Row():
            model_path = gr.Textbox(label="Model Path")
            model_type = gr.Dropdown(["diffusion", "rectified-flow"], value="diffusion", label="Model Type")
            mixed_precision = gr.Dropdown(["no", "fp16", "bf16"], value="no", label="Mixed Precision")

        load_button = gr.Button("Load Model")
        load_output = gr.Textbox(label="Load Status")

        load_button.click(
            load_model,
            inputs=[model_path, model_type, mixed_precision],
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


# Launch the Gradio app
if __name__ == "__main__":
    app = gradio_interface()
    app.launch(share=True)
