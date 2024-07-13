import tempfile
from pathlib import Path
from typing import Dict, Tuple
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

from osu_fusion.library.osu.data.decode import Metadata, decode_beatmap
from osu_fusion.library.osu.data.encode import TOTAL_DIM

# Import necessary functions and classes from your original script
from osu_fusion.models.diffusion import OsuFusion
from osu_fusion.scripts.dataset_creator import HOP_LENGTH, N_FFT, SR, load_audio, normalize_context

# Global variables to store the model and accelerator
global_model = None
global_accelerator = None
global_temp_dir = tempfile.TemporaryDirectory()


def create_model_from_checkpoint(model_path: str) -> OsuFusion:
    if model_path.endswith(".pt"):
        checkpoint = torch.load(model_path)
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = load_file(model_path)

    model = OsuFusion(128, attn_infini=False)
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
) -> torch.Tensor:
    audio = load_audio(audio_path)
    context = normalize_context(np.array([cs, ar, od, hp, sr], dtype=np.float32))

    audio = torch.from_numpy(audio).to(device)
    context = torch.from_numpy(context).to(device)

    audio = repeat(audio, "d n -> b d n", b=batch_size)
    context = repeat(context, "c -> b c", b=batch_size)

    n = audio.shape[-1]
    x = torch.randn((batch_size, TOTAL_DIM, n), device=device)

    return x, audio, context


def load_model(model_path: str, mixed_precision: str) -> str:
    global global_model, global_accelerator
    global_accelerator = Accelerator(mixed_precision=mixed_precision)
    global_model = create_model_from_checkpoint(model_path)
    global_model = global_accelerator.prepare(global_model)
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
    x, audio, context = create_input(
        music_path,
        cs,
        ar,
        od,
        hp,
        sr,
        batch_size,
        device,
    )

    with global_accelerator.autocast():
        generated = global_model.sample(audio, context, x, cond_scale=cfg)

    frame_times = (
        librosa.frames_to_time(np.arange(generated.shape[-1]), sr=SR, hop_length=HOP_LENGTH, n_fft=N_FFT) * 1000
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
            metadata.version = f"{metadata.version} - batch {i + 1}_{batch_size}"
            beatmap = decode_beatmap(metadata, signal, frame_times)
            mapset_archive.writestr(
                f"{metadata.artist} - {metadata.title} (OsuFusion) [{metadata.version}].osu",
                beatmap,
            )

    return gr.update(value=mapset_path, visible=True), f"Beatmap generated successfully: {mapset_path}"


# Define the Gradio interface
def gradio_interface() -> Blocks:
    with gr.Blocks() as app:
        gr.Markdown("# OsuFusion Beatmap Generator")

        with gr.Row():
            model_path = gr.Textbox(label="Model Path")
            mixed_precision = gr.Dropdown(["no", "fp16", "bf16"], value="no", label="Mixed Precision")

        load_button = gr.Button("Load Model")
        load_output = gr.Textbox(label="Load Status")

        load_button.click(
            load_model,
            inputs=[model_path, mixed_precision],
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

        with gr.Row():
            batch_size = gr.Slider(1, 10, value=1, step=1, label="Batch Size")
            steps = gr.Slider(1, 100, value=35, step=1, label="Steps")
            cfg = gr.Slider(0, 10, value=2.0, label="CFG")

        generate_button = gr.Button("Generate Beatmap")
        output_file = gr.File(label="Generated Beatmap", interactive=False)
        output_text = gr.Textbox(label="Generation Status")

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
    app.launch()