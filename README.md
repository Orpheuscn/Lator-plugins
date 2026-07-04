# Lator Plugins

A source repository for Lator workspace plugins.

This repository currently contains three maintained plugins. Each plugin lives under `src/` with its own `plugin.json`, runtime code, assets declaration, and plugin-specific documentation where applicable.

## Plugins

| Plugin | Version | Display name | What it does |
|---|---:|---|---|
| [bilingual-align](src/bilingual-align) | 0.1.0 | Bilingual Align | Aligns translated text back to the source line layout with LaBSE embeddings and N:M alignment. |
| [fast-whisper-subtitle](src/fast-whisper-subtitle) | 0.1.0 | Subtitle Transcription | Transcribes speech from audio or video into editable subtitles using local faster-whisper models. |
| [segment-word-aligner](src/segment-word-aligner) | 0.2.0 | Lexicon QA | Extracts reusable source and translation term pairs for terminology consistency checks. |

## Repository Layout

```text
src/
  bilingual-align/
  fast-whisper-subtitle/
  segment-word-aligner/
```

## Plugin Documentation

- [Bilingual Align README](src/bilingual-align/README.md)
- [Subtitle Transcription README](src/fast-whisper-subtitle/README.md)
- [Lexicon QA README](src/segment-word-aligner/README.md)

## Installation

This repository tracks plugin source code. Installable `.lator-plugin` bundles are produced separately from these sources when a release build is needed.

Python dependencies must be installed into each plugin's virtual environment. When installing manually, use the trusted PyPI hosts required by the local development environment:

```bash
python -m pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org <packages>
```
