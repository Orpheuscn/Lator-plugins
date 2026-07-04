# Lexicon QA Plugin

Lexicon QA adds a ribbon button to Lator that builds a reusable terminology index from the active root projects. It reads each translation segment, extracts likely source and target term pairs, and writes the result to `word-alignments.json` for terminology consistency checks and bilingual lookup.

## What It Does

- Extracts word and phrase pairs from source and translated text.
- Keeps phrase-level terminology by default and can optionally include word-level entries.
- Filters stopwords and noisy single-character Chinese word entries to keep the lexicon reusable.
- Stores one result object per segment, with source text, target text, and extracted entries.
- Registers a host index search provider so Lator can search the generated lexicon.

## User Controls

The plugin contributes these settings:

- `Include word-level terms`: keeps non-stopword word-level term entries. Phrase entries are always retained.
- `Source phrase length limit`: limits how many source words a phrase entry can contain. The default is `6`.

The ribbon tooltip is `Lexicon QA`.

## Output

The plugin writes `word-alignments.json` through Lator's project batch output flow. Each segment includes an `entries` array. A typical entry looks like this:

```json
{
  "src_text": "Book of Disquiet",
  "src_span": [10, 26],
  "tgt_text": "不安之书",
  "tgt_span": [4, 8],
  "score": 0.84,
  "alignment_type": "phrase"
}
```

Entry fields:

- `src_text` and `tgt_text`: the extracted source and target term text.
- `src_span` and `tgt_span`: character offsets in the original segment text.
- `score`: alignment confidence score.
- `alignment_type`: `phrase` or `word`.

## Alignment Pipeline

The alignment model is `aneuraz/awesome-align-with-co`, an mBERT model fine-tuned for word alignment on parallel data. The plugin uses awesome-align-style softmax extraction from hidden states and allows many-to-many wordpiece links, which is important for translations where one word maps to multiple characters or words.

After wordpiece links are extracted, the plugin groups connected links into minimal translation units. Units with compact source and target spans become phrase entries. Other valid units fall back to word entries when word-level output is enabled.

## Filtering Rules

- Standalone stopword entries are removed.
- Phrase entries are not removed just because they contain internal stopwords. For example, `Book of Disquiet` can keep `of` inside the phrase.
- Single-character Chinese entries are not emitted as standalone word entries by default, because they often create noisy and non-reusable lexicon records.

## Tokenization Assets

Chinese tokenization uses HanLP. The required tokenizer assets are declared in `plugin.json` and downloaded by the Lator plugin system during installation:

- `hanlp-coarse-electra-small-zh`: HanLP coarse Chinese tokenizer.
- `chinese-electra-180g-small-discriminator`: the ELECTRA encoder used by the tokenizer.
- `hanlp-char-table`: HanLP character normalization table.

At runtime, the plugin prefers these local asset paths:

- `LATOR_PLUGIN_ASSET_HANLP_COARSE_ELECTRA_SMALL_ZH`
- `LATOR_PLUGIN_ASSET_CHINESE_ELECTRA_180G_SMALL_DISCRIMINATOR`
- `LATOR_PLUGIN_ASSET_HANLP_CHAR_TABLE`

You can override the Chinese tokenizer with:

```bash
LATOR_HANLP_ZH_TOKENIZER=/path/to/hanlp/tokenizer
```

If neither the plugin assets nor the override are available, HanLP may fall back to downloading the tokenizer from `https://file.hankcs.com/hanlp/tok/coarse_electra_small_20220616_012050.zip`. If the HanLP tokenizer cannot be loaded, the plugin fails instead of falling back to character-by-character tokenization.

Japanese tokenization uses SudachiPy with `sudachidict_core`. You can customize it with:

```bash
LATOR_SUDACHI_JA_CONFIG=/path/to/sudachi.json
LATOR_SUDACHI_JA_SPLIT_MODE=A|B|C
```

The default Japanese split mode is `C`.

## Runtime Dependencies

Lator installs plugin dependencies into the plugin virtual environment. For manual testing, install them in a virtual environment and include the required trusted hosts:

```bash
python -m pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org hanlp safetensors stopwordsiso sudachidict_core sudachipy torch 'transformers>=4.40'
```
