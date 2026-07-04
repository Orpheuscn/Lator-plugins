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

This plugin is distributed as two `.lator-plugin` bundle variants:

- **Flexible VAD bundle**: keeps the `vadBackend` setting so users can choose Silero or pyannote. The wheelhouse includes both `silero-vad` and the optional `pyannote.audio` dependency group.
- **Bundled pyannote bundle**: removes Silero from runtime dependencies, removes the `vadBackend` setting, forces `vad_backend="pyannote"`, and embeds the local `pyannote/segmentation-3.0` model as a bundled asset.

Do not maintain two complete source trees. Keep this directory as the only source of truth, then create a temporary staging copy for the pyannote-only variant when packaging.

The examples assume:

```bash
HOST=/path/to/Lator-Electron
PLUGIN=/path/to/Lator-plugins/src/fast-whisper-subtitle
OUT=/path/to/lator-bundle-build
PYANNOTE_MODEL="$HOME/.lator/models/huggingface/pyannote/segmentation-3.0/main"
cd "$HOST"
```

### Build the Flexible VAD Bundle

This bundle keeps the plugin source unchanged and includes every optional requirement group in the wheelhouse:

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
  bundle.json                     # no bundledAssets
  requirements.lock
  wheelhouse/darwin-arm64-cp312/  # includes silero-vad and pyannote.audio
```

### Build the Bundled pyannote Bundle

Create a temporary copy and patch only that copy before calling the host bundle builder:

```bash
STAGING_ROOT=$(mktemp -d /private/tmp/lator-fast-whisper-pyannote-XXXXXX)
STAGING_PLUGIN="$STAGING_ROOT/fast-whisper-subtitle-pyannote"
rsync -a --exclude .git --exclude .venv --exclude node_modules --exclude __pycache__ --exclude .pytest_cache --exclude .mypy_cache --exclude .ruff_cache --exclude dist --exclude release --exclude .DS_Store "$PLUGIN/" "$STAGING_PLUGIN/"

node - "$STAGING_PLUGIN" <<'NODE'
import { promises as fs } from 'node:fs'
import path from 'node:path'

const pluginRoot = process.argv[2]
const manifestPath = path.join(pluginRoot, 'plugin.json')
const manifest = JSON.parse(await fs.readFile(manifestPath, 'utf8'))

manifest.label = {
  'zh-CN': '字幕转写',
  'en-US': 'Subtitle Transcription'
}
manifest.description = {
  'zh-CN': '将音频或视频中的人声转写为可编辑字幕，内置高精度语音分段模型。',
  'en-US': 'Transcribes speech from audio or video into editable subtitles, with a bundled high-accuracy speech segmentation model.'
}
manifest.runtime.requirements = Array.from(new Set([
  ...manifest.runtime.requirements.filter(item => !/^silero-vad\b/i.test(item)),
  'pyannote.audio>=4.0.0'
]))
delete manifest.runtime.optionalRequirements

for (const asset of manifest.assets || []) {
  if (asset.id === 'pyannote-segmentation-3.0') {
    delete asset.optional
    delete asset.when
  }
}

if (manifest.contributes?.settings) {
  manifest.contributes.settings = manifest.contributes.settings.filter(item => item.id !== 'vadBackend')
}
if (manifest.contributes?.helpLinks) {
  manifest.contributes.helpLinks = manifest.contributes.helpLinks.filter(item => item.id !== 'pyannote-segmentation-auth')
}
await fs.writeFile(manifestPath, `${JSON.stringify(manifest, null, 2)}\n`)

const configPath = path.join(pluginRoot, 'fastwhisper_subtitle/config.py')
let configText = await fs.readFile(configPath, 'utf8')
configText = configText.replace('"vad_backend": "silero"', '"vad_backend": "pyannote"')
await fs.writeFile(configPath, configText)

const pipelinePath = path.join(pluginRoot, 'fastwhisper_subtitle/pipeline.py')
let pipelineText = await fs.readFile(pipelinePath, 'utf8')
pipelineText = pipelineText.replace(
`        vad_backend=normalize_choice(\n            read_parameter(parameter_values, "vadBackend", DEFAULT_CONFIG["vad_backend"]),\n            {"silero", "pyannote"},\n            DEFAULT_CONFIG["vad_backend"],\n        ),`,
'        vad_backend="pyannote",'
)
await fs.writeFile(pipelinePath, pipelineText)

const requirementsPath = path.join(pluginRoot, 'requirements.txt')
try {
  let requirementsText = await fs.readFile(requirementsPath, 'utf8')
  requirementsText = requirementsText
    .split(/\r?\n/)
    .filter(line => !/^silero-vad\b/i.test(line.trim()))
    .join('\n')
    .trimEnd()
  if (!/^pyannote\.audio\b/im.test(requirementsText)) {
    requirementsText += '\npyannote.audio>=4.0.0'
  }
  await fs.writeFile(requirementsPath, `${requirementsText}\n`)
} catch (error) {
  if (error?.code !== 'ENOENT') throw error
}
NODE

npm run plugin:bundle -- "$STAGING_PLUGIN" \
  --python 3.12 \
  --platform darwin-arm64 \
  --uv .bundle-venv/bin/uv \
  --python-bin .bundle-venv/bin/python \
  --bundle-asset "pyannote-segmentation-3.0=$PYANNOTE_MODEL" \
  --out "$OUT/fast-whisper-subtitle-pyannote-darwin-arm64.lator-plugin"
```

Expected result:

```text
fast-whisper-subtitle-pyannote-darwin-arm64.lator-plugin
  plugin.json                     # no vadBackend setting, no silero-vad
  bundle.json                     # has bundledAssets entry
  bundled-assets/pyannote-segmentation-3.0/pytorch_model.bin
  requirements.lock
  wheelhouse/darwin-arm64-cp312/  # includes pyannote.audio, excludes silero-vad
```

### Verify the Bundles

```bash
unzip -p "$OUT/fast-whisper-subtitle-darwin-arm64.lator-plugin" \
  local.fast-whisper-subtitle/bundle.json

unzip -p "$OUT/fast-whisper-subtitle-pyannote-darwin-arm64.lator-plugin" \
  local.fast-whisper-subtitle/bundle.json

unzip -l "$OUT/fast-whisper-subtitle-pyannote-darwin-arm64.lator-plugin" \
  | rg 'bundled-assets|pytorch_model.bin'

unzip -p "$OUT/fast-whisper-subtitle-pyannote-darwin-arm64.lator-plugin" \
  local.fast-whisper-subtitle/plugin.json \
  | rg 'silero|vadBackend|optionalRequirements|pyannote.audio|pyannote-segmentation-3.0'
```

The final check should only show `pyannote.audio` and `pyannote-segmentation-3.0`; it should not show `silero`, `vadBackend`, or `optionalRequirements`.
