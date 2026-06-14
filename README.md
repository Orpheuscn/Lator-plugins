# Lator Plugins

A collection of plugins for the [Lator](https://github.com/Orpheuscn) translation/subtitle workspace.

This repository holds the **source** for each plugin. Pre-built, installable bundles
(`.lator-plugin`) are attached to the corresponding [GitHub Releases](../../releases).

## Plugins

| Plugin | Version | What it does | Release |
|---|---|---|---|
| [bilingual-align](src/bilingual-align) | 0.1.0 | Sentence-level bilingual alignment using LaBSE (ONNX). | `bilingual-align-v0.1.0` |
| [fast-whisper-subtitle](src/fast-whisper-subtitle) | 0.1.0 | Media subtitle recognition: FFmpeg + VAD + faster-whisper. | `fast-whisper-subtitle-v0.1.0` |
| [segment-word-aligner](src/segment-word-aligner) | 0.2.0 | Per-segment source/translation word-alignment dictionaries (awesome-align + HanLP). | `segment-word-aligner-v0.2.0` |
| [subtitle-playback](src/subtitle-playback) | 0.1.0 | Subtitle playback via bundled ffplay/ffmpeg. | `subtitle-playback-v0.1.0` |

## Installing a bundle

1. Open the [Releases](../../releases) page and download the `.lator-plugin` (or zip) for the plugin you want.
2. Install it from within Lator.

> Bundles are currently built for **macOS arm64 (darwin-arm64)**.

`fast-whisper-subtitle` ships two bundle variants in its release:

- **Flexible VAD** — keeps the `vadBackend` setting (Silero or pyannote).
- **Bundled pyannote** — pyannote VAD only, with the segmentation model embedded.

## Per-plugin docs

Each plugin's own `README.md` (linked above) documents its models, assets, settings, and build steps.
