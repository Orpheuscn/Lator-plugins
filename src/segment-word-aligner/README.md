# Segment Word Aligner Plugin

这个插件给 Lator 工作区 ribbon 顶部增加一个按钮。点击后，插件会读取宿主传来的当前项目全部 segments，对每个 segment 的原文和译文做词/词组对齐，并把结果写成静态 JSON 词典文件。

模型和方法：

- 模型：`aneuraz/awesome-align-with-co`（在平行语料上用 awesome-align 目标微调过的 mBERT，对齐质量显著优于原版 `bert-base-multilingual-cased`）
- 中文分词：HanLP `COARSE_ELECTRA_SMALL_ZH`
- 日语分词：SudachiPy + `sudachidict_core`
- 方法：awesome-align 的 softmax 抽取：第 8 层 hidden states、similarity matrix、双向 softmax 超过阈值（0.001）即建立 wordpiece 链接，允许多对多
- 词典构建：把词级链接按连通分量聚成最小翻译单元；两侧 span 一致（span 内跳过的词都未对齐到其他单元）且不超长的分量输出为 phrase 条目，其余拆回单词条目。一侧全部是停用词的单元直接丢弃。这样不会再产生互相重叠的滑动窗口词组
- 输出：每个 segment 一个对象，`entries` 中保存词或词组对齐条目

输出条目格式：

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

停用词规则：

- 单独成词的停用词会被排除。
- 词组条目不会因为内部包含停用词而被排除，例如 `Book of Disquiet` 可以保留其中的 `of`。
- 中文单字词条默认不作为单独 word 输出，以免把静态词典污染成大量不可复用的单字映射。

HanLP 分词模型：

- 中文分词所需的三个文件都作为插件资产，在**安装插件时一次性下载**，运行时不再按需联网拉取：
  - `hanlp-coarse-electra-small-zh`：分词模型 `coarse_electra_small_20220616_012050.zip`（file.hankcs.com）。
  - `chinese-electra-180g-small-discriminator`：分词模型依赖的 ELECTRA 编码器 `hfl/chinese-electra-180g-small-discriminator`（HuggingFace，含国内镜像）。
  - `hanlp-char-table`：字符规范化表 `char_table_20210602_202632.json`（file.hankcs.com）。
- 运行时插件优先使用上述本地资产：分词模型从 `LATOR_PLUGIN_ASSET_HANLP_COARSE_ELECTRA_SMALL_ZH` 读取，并把 `transformer` 指向 `LATOR_PLUGIN_ASSET_CHINESE_ELECTRA_180G_SMALL_DISCRIMINATOR`、把字符表 `mapper` 指向 `LATOR_PLUGIN_ASSET_HANLP_CHAR_TABLE`。
- 如果你想用自己本地缓存/安装的分词模型，可以设置 `LATOR_HANLP_ZH_TOKENIZER=/path/to/hanlp/tokenizer`（优先级最高）。
- 若资产与该环境变量都不存在，才会回退到从 `https://file.hankcs.com/hanlp/tok/coarse_electra_small_20220616_012050.zip` 按需下载。
- 如果 HanLP 模型不可加载，插件会报错，不会退回到单字分词。

SudachiPy 日语分词：

- 日语文本会走 SudachiPy，不再退回到逐 CJK 字符切分。
- 默认使用 `sudachidict_core`，词典作为 Python 依赖安装在插件虚拟环境里，卸载插件环境时会一起清理。
- 如果你想使用自定义 Sudachi 配置，可以设置 `LATOR_SUDACHI_JA_CONFIG=/path/to/sudachi.json`。
- 默认使用 SplitMode C；可通过 `LATOR_SUDACHI_JA_SPLIT_MODE=A|B|C` 调整切分粒度。

安装依赖由 Lator 插件系统处理。手动测试时请在虚拟环境里安装：

```bash
python -m pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org hanlp safetensors stopwordsiso sudachidict_core sudachipy torch 'transformers>=4.40'
```
