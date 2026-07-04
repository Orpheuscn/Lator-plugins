# Fast Whisper Media Subtitle Plugin

This external plugin is a trimmed copy of the standalone FastWhisperSubtitle
backend, adapted to Lator's Python runtime plugin protocol.

It intentionally does not include the original Web UI, CLI commands,
transliteration code, or bundled model weights. The core backend path is kept:
FFmpeg audio extraction, pyannote VAD, optional SpeechBrain language
identification, and faster-whisper recognition. Lator owns the UI, project
persistence, subtitle table, editing, playback, and import flow.

## Capability

`transcribe-media` is declared as a `subtitle-media-import` capability.

Payload:

```json
{
  "mode": "transcribe",
  "filePath": "/absolute/path/to/media.mp4",
  "projectId": "project-id",
  "parameters": {
    "model": "base",
    "language": "auto",
    "task": "transcribe",
    "beamSize": 5,
    "computeType": "default",
    "silenceThreshold": 2.0,
    "speechPad": 300,
    "strict": false,
    "forceRedetect": false,
    "vadModelPath": "/optional/local/pyannote-segmentation-3.0",
    "languageIdModelPath": "/optional/local/language-id-voxlingua107-ecapa",
    "allowModelDownload": false
  }
}
```

Known hallucination phrases from
`fastwhisper_subtitle/data/hallucination_phrases.json` are always filtered.
`strict` only controls experimental low-confidence subtitle marking with `<del>`
using thresholds from `fastwhisper_subtitle/data/quality_profiles.json`.

Events:

```json
{"type":"progress","stage":"transcribe","message":"Recognized 12 subtitle segments","current":12,"total":0}
{"type":"segment","segment":{"startTime":"00:00:01,000","endTime":"00:00:03,500","text":"..."}}
```

Final result:

```json
{
  "media": {"sourcePath": "/absolute/path/to/media.mp4", "kind": "video"},
  "segments": [
    {"startTime": "00:00:01,000", "endTime": "00:00:03,500", "text": "..."}
  ],
  "srt": "1\n00:00:01,000 --> 00:00:03,500\n...",
  "detectedLanguage": "en"
}
```

The plugin also declares `play-subtitle-segment` as a `subtitle-playback`
capability. Lator prepares cache output paths and URLs in `hostMedia`; the plugin
cuts the requested time range with FFmpeg and returns `audioUrl` and, for video
sources, `videoUrl`.

FFmpeg is declared as a `tools` archive asset and is exposed to Python through
`LATOR_PLUGIN_ASSET_FFMPEG`. The runtime falls back to `FFMPEG_BINARY` or `ffmpeg`
on PATH if the asset is not installed.

## Dependencies

The runtime dependency list keeps the ASR pipeline dependencies and removes only
the parts no longer used by the host-integrated backend:

- `torch`
- `soundfile`
- `numpy`
- `speechbrain`
- `faster-whisper`
- `huggingface-hub`
- `silero-vad`

`pyannote.audio` is installed only when the plugin setting `vadBackend` is set
to `pyannote`.

Transliteration dependencies from the original app (`pykakasi`, `aksharamukha`,
`unidecode`) were removed. Flask and Flask-CORS were removed with the standalone
web server.

The plugin declares these model assets:

- `ffmpeg` from `https://evermeet.cx/ffmpeg/ffmpeg-8.1.1.zip`
- `Systran/faster-whisper-base` as the default required ASR model
- optional on-demand ASR models: `Systran/faster-whisper-small`,
  `Systran/faster-whisper-medium`, `Systran/faster-whisper-large-v3`, and
  `dropbox-dash/faster-whisper-large-v3-turbo`
- optional high-accuracy VAD model: `pyannote/segmentation-3.0`
- `speechbrain/lang-id-voxlingua107-ecapa`

Only `base` and the selected VAD backend are installed with the plugin. The
larger faster-whisper models are shown in Lator settings as on-demand model
downloads; after downloading one, select the matching `model` option (`small`,
`medium`, `large-v3`, or `turbo`) in the media recognition dialog for the
current audio or video file.

`pyannote/segmentation-3.0` may require the user to sign in to Hugging Face and
accept the model terms before the host downloader can fetch it. Lator validates
the pasted Hugging Face token before continuing and uses that token for the host
download request.

If a model asset cannot be downloaded by the host, the caller can pass local
model directories with `parameters.vadModelPath` and
`parameters.languageIdModelPath`. The original project's local `models/`
directories work for those values.

## Bundle Packaging

The bundle keeps the standard `vadBackend` setting so users can choose Silero or pyannote. The pyannote model has separate Hugging Face access terms, so users who choose High Accuracy mode must authorize and download that model through the normal Lator asset flow.

The examples assume:

```bash
HOST=/path/to/Lator-Electron
PLUGIN=/path/to/Lator-plugins/src/fast-whisper-subtitle
OUT=/path/to/lator-bundle-build
cd "$HOST"
```

### Build a Bundle

This command keeps the plugin source unchanged and includes every optional Python requirement group in the wheelhouse. Including `pyannote.audio` lets the High Accuracy mode work after the user separately authorizes and downloads the pyannote model.

```bash
npm run plugin:bundle -- "$PLUGIN" \
  --python 3.12 \
  --platform darwin-arm64 \
  --include-optional all \
  --uv .bundle-venv/bin/uv \
  --python-bin .bundle-venv/bin/python \
  --out "$OUT/fast-whisper-subtitle-darwin-arm64.lator-plugin"
```

Expected result:

```text
fast-whisper-subtitle-darwin-arm64.lator-plugin
  plugin.json                     # keeps vadBackend setting
  bundle.json                     # no bundledAssets entry
  requirements.lock
  wheelhouse/darwin-arm64-cp312/  # includes runtime wheels, including optional pyannote.audio
```

### Verify the Bundle

```bash
unzip -p "$OUT/fast-whisper-subtitle-darwin-arm64.lator-plugin" \
  local.fast-whisper-subtitle/bundle.json

unzip -l "$OUT/fast-whisper-subtitle-darwin-arm64.lator-plugin" \
  | rg 'bundled-assets|pyannote-segmentation-3.0'
```

The second command should print nothing. If it shows `bundled-assets` or `pyannote-segmentation-3.0`, the bundle is incorrectly redistributing the pyannote model.

## License

This plugin is distributed under the GNU General Public License v3.0. See [LICENSE](LICENSE).
