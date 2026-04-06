from dataclasses import asdict
from pathlib import Path

import gradio as gr
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from gradio import Blocks
from rosu_pp_py import Beatmap as RosuBeatmap
from rosu_pp_py import Difficulty as RosuDifficulty
from safetensors.torch import load_file
from slider.beatmap import Beatmap

from osu_fusion.data.descriptors import DESCRIPTOR_TAGS, NUM_DESCRIPTORS
from osu_fusion.data.encode import encode_sequence
from osu_fusion.data.prepare_data import load_audio
from osu_fusion.models.classifier import (
    ClassifierConfig_L,
    ClassifierConfig_M,
    ClassifierConfig_S,
    OsuFusionClassifier,
)

matplotlib.use("Agg")

MODEL_CONFIGS = {
    "s": ClassifierConfig_S,
    "m": ClassifierConfig_M,
    "l": ClassifierConfig_L,
}

global_model = None
global_device = None


def create_model_from_checkpoint(model_path: str, model_size: str) -> OsuFusionClassifier:
    config = MODEL_CONFIGS[model_size]
    model = OsuFusionClassifier(**asdict(config))

    if model_path.endswith(".pt"):
        checkpoint = torch.load(model_path, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        state_dict = load_file(model_path)
        model.load_state_dict(state_dict)

    return model.eval()


def load_model(model_path: str, model_size: str) -> str:
    global global_model, global_device
    global_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    global_model = create_model_from_checkpoint(model_path, model_size)
    global_model = global_model.to(global_device)
    return f"Model loaded on {global_device}!"


def encode_osu_file(osu_path: str) -> tuple:
    beatmap = Beatmap.from_path(osu_path)

    audio_file = Path(osu_path).parent / beatmap.audio_filename
    if audio_file.exists():
        spec = load_audio(audio_file)
        total_frames = spec.shape[0]
    else:
        msg = f"Audio file not found: {audio_file}"
        raise FileNotFoundError(msg)

    x = encode_sequence(beatmap, total_frames=total_frames)
    a = spec  # (T, AUDIO_DIM)
    with open(osu_path, "r", encoding="utf-8") as f:
        rosu_beatmap = RosuBeatmap(content=f.read())
    sr = RosuDifficulty().calculate(rosu_beatmap).stars

    c = np.array(
        [
            rosu_beatmap.cs,
            rosu_beatmap.ar,
            rosu_beatmap.od,
            rosu_beatmap.hp,
            sr,
            rosu_beatmap.slider_multiplier,
            rosu_beatmap.slider_tick_rate,
            -1.0,  # era unknown from .osu file alone
        ],
        dtype=np.float32,
    )

    return x, a, c, beatmap, sr


def make_category_radar(all_tags: list) -> plt.Figure:
    categories = {}
    for tag, score in all_tags:
        cat = tag.split("/")[0]
        if tag == cat:
            categories[cat] = score

    labels = list(categories.keys())
    values = list(categories.values())

    if not labels:
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        return fig

    n = len(labels)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    values_closed = [*values, values[0]]
    angles_closed = [*angles, angles[0]]

    fig, ax = plt.subplots(figsize=(5, 5), subplot_kw={"polar": True})
    ax.fill(angles_closed, values_closed, alpha=0.25, color="#4CAF50")
    ax.plot(angles_closed, values_closed, color="#4CAF50", linewidth=2)
    ax.set_xticks(angles)
    ax.set_xticklabels([label.capitalize() for label in labels], fontsize=9)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["25%", "50%", "75%", "100%"], fontsize=7, color="gray")
    ax.set_title("Category Overview", fontsize=12, fontweight="bold", pad=20)
    plt.tight_layout()
    return fig


def make_top_tags_bar(all_tags: list, threshold: float, top_n: int = 20) -> plt.Figure:
    leaf_tags = [(tag, score) for tag, score in all_tags if "/" in tag]
    leaf_tags.sort(key=lambda x: -x[1])
    top = leaf_tags[:top_n]

    if not top:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, "No tags detected", ha="center", va="center")
        return fig

    labels = [t.split("/")[-1].replace("_", " ") for t, _ in reversed(top)]
    scores = [s for _, s in reversed(top)]
    colors = ["#4CAF50" if s >= threshold else "#9E9E9E" for s in scores]

    fig, ax = plt.subplots(figsize=(8, max(4, len(top) * 0.35)))
    bars = ax.barh(labels, scores, color=colors, edgecolor="white", linewidth=0.5)
    ax.axvline(x=threshold, color="#F44336", linestyle="--", linewidth=1, label=f"Threshold ({threshold:.0%})")
    ax.set_xlim(0, 1)
    ax.set_xlabel("Confidence")
    ax.set_title("Top Tag Predictions", fontsize=12, fontweight="bold")
    ax.legend(loc="lower right")

    for bar, score in zip(bars, scores, strict=True):
        ax.text(min(score + 0.02, 0.95), bar.get_y() + bar.get_height() / 2, f"{score:.0%}", va="center", fontsize=8)

    plt.tight_layout()
    return fig


def make_confidence_histogram(all_tags: list) -> plt.Figure:
    scores = [s for _, s in all_tags]

    fig, ax = plt.subplots(figsize=(6, 3))
    ax.hist(scores, bins=20, range=(0, 1), color="#2196F3", edgecolor="white", alpha=0.8)
    ax.set_xlabel("Confidence Score")
    ax.set_ylabel("Number of Tags")
    ax.set_title("Confidence Distribution", fontsize=12, fontweight="bold")
    ax.axvline(x=0.5, color="#F44336", linestyle="--", linewidth=1, alpha=0.5)
    plt.tight_layout()
    return fig


def format_tag_tree(tags_with_scores: list) -> str:
    categories = {}
    for tag, score in tags_with_scores:
        parts = tag.split("/")
        cat = parts[0]
        if cat not in categories:
            categories[cat] = []
        categories[cat].append((tag, score))

    lines = []
    for cat in sorted(categories.keys()):
        items = sorted(categories[cat], key=lambda x: -x[1])
        cat_score = next((s for t, s in items if t == cat), 0.0)
        lines.append(f"\n### {cat.upper()} ({cat_score:.0%})")
        for tag, score in items:
            if tag == cat:
                continue
            depth = tag.count("/")
            indent = "  " * depth
            bar = "█" * int(score * 20) + "░" * (20 - int(score * 20))
            tag_short = tag.split("/")[-1].replace("_", " ")
            lines.append(f"{indent}**{tag_short}** {bar} {score:.1%}")

    return "\n".join(lines)


def analyze_beatmap(osu_file: str, threshold: float) -> tuple:
    global global_model, global_device

    if global_model is None:
        return "Error: Model not loaded.", None, None, None

    if not osu_file or not Path(osu_file).exists():
        return "Error: File not found. Paste the full path to a .osu file.", None, None, None

    try:
        x, a, c, beatmap, sr = encode_osu_file(osu_file)
    except Exception as e:
        return f"Error encoding beatmap: {e}", None, None, None

    x_tensor = torch.from_numpy(x).unsqueeze(0).to(global_device, torch.float32)
    a_tensor = torch.from_numpy(a).unsqueeze(0).to(global_device, torch.float32)
    c_tensor = torch.from_numpy(c).unsqueeze(0).to(global_device, torch.float32)

    with torch.inference_mode():
        logits = global_model(x_tensor, a_tensor, c_tensor)
        probs = torch.sigmoid(logits).cpu().numpy()[0]

    all_tags = [(DESCRIPTOR_TAGS[i], float(probs[i])) for i in range(NUM_DESCRIPTORS)]
    predicted = [(tag, score) for tag, score in all_tags if score >= threshold]
    predicted.sort(key=lambda x: -x[1])

    hit_objects = beatmap.hit_objects()
    n_circles = sum(1 for ho in hit_objects if type(ho).__name__ == "Circle")
    n_sliders = sum(1 for ho in hit_objects if type(ho).__name__ == "Slider")
    n_spinners = sum(1 for ho in hit_objects if type(ho).__name__ == "Spinner")

    info = f"""## Beatmap Analysis

### Map Info
- **Title:** {beatmap.title}
- **Artist:** {beatmap.artist}
- **Creator:** {beatmap.creator}
- **Version:** {beatmap.version}
- **Star Rating:** {sr:.2f}★
- **CS:** {c[0]:.1f} | **AR:** {c[1]:.1f} | **OD:** {c[2]:.1f} | **HP:** {c[3]:.1f}
- **Objects:** {len(hit_objects)} ({n_circles} circles, {n_sliders} sliders, {n_spinners} spinners)
- **Length:** {x.shape[0] * 16 / 1000:.1f}s ({x.shape[0]} frames)

### Predicted Tags ({len(predicted)} detected at {threshold:.0%} threshold)

"""

    if predicted:
        top_tags = ", ".join([f"**{t.split('/')[-1].replace('_', ' ')}** ({s:.0%})" for t, s in predicted[:10]])
        info += f"🏷️ {top_tags}\n"

    info += format_tag_tree(all_tags)

    # Generate charts
    radar_fig = make_category_radar(all_tags)
    bar_fig = make_top_tags_bar(all_tags, threshold)
    hist_fig = make_confidence_histogram(all_tags)

    return info, radar_fig, bar_fig, hist_fig


def gradio_interface() -> Blocks:
    with gr.Blocks(title="OsuFusion Classifier") as app:
        gr.Markdown("# 🎵 OsuFusion Beatmap Classifier\nAnalyze beatmap style and characteristics")

        with gr.Row():
            model_path = gr.Textbox(label="Model Path (safetensors or pt)")
            model_size = gr.Dropdown(["s", "m", "l"], value="s", label="Model Size")

        load_button = gr.Button("Load Model")
        load_output = gr.Textbox(label="Status")

        load_button.click(load_model, inputs=[model_path, model_size], outputs=load_output)

        with gr.Row():
            osu_file = gr.Textbox(
                label=".osu file path (paste full path)",
                placeholder=r"C:\Users\...\Songs\...\map.osu",
            )
            threshold = gr.Slider(0.1, 0.9, value=0.5, step=0.05, label="Detection Threshold")

        analyze_button = gr.Button("🔍 Analyze Beatmap", variant="primary")

        with gr.Row():
            radar_plot = gr.Plot(label="Category Radar")
            bar_plot = gr.Plot(label="Top Tags")

        with gr.Row():
            hist_plot = gr.Plot(label="Confidence Distribution")

        output = gr.Markdown(label="Detailed Analysis")

        analyze_button.click(
            analyze_beatmap,
            inputs=[osu_file, threshold],
            outputs=[output, radar_plot, bar_plot, hist_plot],
        )

    return app


if __name__ == "__main__":
    app = gradio_interface()
    app.launch(share=True)
