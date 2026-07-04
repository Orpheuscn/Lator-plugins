# Bilingual Align Plugin

Bilingual Align is a Lator plugin for fitting translated text back onto the source document's line layout. It is designed for parallel text and subtitle import workflows where the translated text is already available, but its line breaks and segment boundaries do not match the source.

The plugin compares source lines and target fragments with multilingual sentence embeddings, builds N:M alignment groups, and returns a row-by-row result that preserves the original source layout.

## What It Does

- Aligns translated text to the source line structure.
- Supports one-to-one, one-to-many, many-to-one, and many-to-many alignment groups.
- Preserves blank source lines in the output.
- Marks missing source content, additional target content, and likely mistranslations.
- Streams progress and row updates so Lator can update the UI while alignment is running.
- Works for both text and subtitle parallel translation import modes.

## Lator Integration

The plugin exposes one streaming capability:

- `bilingual-align.align`: aligns `sourceText` and `targetText` and returns layout-preserving rows.

It contributes two host integrations:

- `Bert Alignment`: a parallel text aligner.
- `Add parallel translation`: a parallel translation importer for `text` and `subtitle` modes.

## Input

The plugin expects source and target text from Lator:

```json
{
  "sourceText": "First source line\nSecond source line",
  "targetText": "Translated content that may have different breaks."
}
```

`sourceText` should contain the layout that must be preserved. `targetText` can be plain translated text or subtitle-derived text with different segmentation.

## Output

The final response contains aligned rows and a summary:

```json
{
  "rows": [
    {
      "line_number": 1,
      "source": "First source line",
      "target": "First translated line",
      "status": "aligned",
      "similarity": 0.91
    }
  ],
  "summary": {
    "source_line_count": 2,
    "target_segment_count": 2,
    "missing_count": 0,
    "addition_count": 0,
    "mistranslation_count": 0
  }
}
```

Common row statuses include:

- `aligned`: a source line has a matched target fragment.
- `missing`: source content appears to have no matching target content.
- `addition`: target content appears to have no matching source line.
- `mistranslation`: the match exists but has low semantic confidence.
- `blank`: the source line is blank and is preserved as blank output.

Some statuses may be combined, such as an aligned row that also carries an added target fragment.

## Alignment Pipeline

The pipeline has four main stages:

1. Normalize source and target text and split them into layout-aware units.
2. Encode source units and target fragments with the LaBSE ONNX embedding model.
3. Run N:M dynamic-programming alignment to build contiguous source-target groups.
4. Assign target fragments back to individual source rows and audit low-confidence matches.

For large documents, the aligner processes source and target text in overlapping chunks. This keeps memory usage bounded while preserving enough context around chunk boundaries to avoid obvious split errors.

## Model Asset

The plugin uses `sentence-transformers/LaBSE` through an ONNX model asset declared in `plugin.json`:

- `model.onnx`
- `tokenizer.json`
- `tokenizer_config.json`
- `vocab.txt`

Lator downloads this asset during plugin installation. At runtime, the plugin reads it from `LATOR_PLUGIN_ASSET_LABSE_ONNX` and maps that path to `LABSE_ONNX_DIR` before loading the encoder.

If you need to override the model location for local testing, set:

```bash
LABSE_ONNX_DIR=/path/to/labse-onnx
```

## Runtime Dependencies

Lator installs Python dependencies into the plugin virtual environment. For manual testing, install them in a virtual environment and include the required trusted hosts:

```bash
python -m pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org numpy 'onnxruntime>=1.15.0' 'transformers>=4.30.0'
```

## Notes for Maintainers

The public plugin id is `lator.bilingual-align`. The model key currently exposed by the implementation is `bert`, with `labse` accepted as an alias for the same LaBSE ONNX encoder.

Alignment tuning lives in `bilingual_align/config/bert/alignment.json`. The most important groups are:

- `encoder`: ONNX model path and runtime batching settings.
- `alignment.nm`: N:M dynamic-programming settings.
- `alignment.assignment`: fragment assignment and addition thresholds.
- `alignment.mistranslation_threshold`: low-confidence audit threshold.

## Attribution

Part of the N:M alignment implementation is based on Bertalign's two-step sentence alignment approach: first finding approximate 1:1 anchor points, then using dynamic programming to extract valid 1:many, many:1, and many:many alignments. The original Bertalign repository is available at [bfsujason/bertalign](https://github.com/bfsujason/bertalign).
