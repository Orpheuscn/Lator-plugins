import json
import os
import tempfile
from typing import Dict, List, Tuple

from speechbrain.inference.classifiers import EncoderClassifier

from fastwhisper_subtitle.model_paths import resolve_language_id_model_path
from fastwhisper_subtitle.services.audio import read_mono_audio


def load_language_id_model(model_dir: str | None = None):
    model_path = resolve_language_id_model_path(model_dir)
    print("\n正在加载本地语言识别模型 speechbrain/lang-id-voxlingua107-ecapa...")

    try:
        language_id = EncoderClassifier.from_hparams(
            source=str(model_path),
            savedir=str(model_path),
            overrides={"pretrained_path": str(model_path)},
        )
        print(f"✓ 语言识别模型加载成功: {model_path}\n")
        return language_id
    except Exception as e:
        print(f"✗ 加载语言识别模型失败: {e}")
        import traceback
        traceback.print_exc()
        raise


def detect_language_for_segments(
    audio_file: str,
    segments: List[Tuple[float, float]],
    temp_dir: str = None,
    confidence_threshold: float = 0.5,
    model_dir: str | None = None,
) -> List[Dict]:
    """Detect language for each speech segment using SpeechBrain."""
    if not segments:
        return []

    language_id = load_language_id_model(model_dir)
    waveform, sample_rate = read_mono_audio(audio_file)

    segments_with_language = []
    all_language_results = []

    print("=" * 60)
    print("正在检测每个语音片段的语言")
    print("=" * 60)

    for i, (start_ms, end_ms) in enumerate(segments):
        start_sample = int(start_ms * sample_rate / 1000)
        end_sample = int(end_ms * sample_rate / 1000)
        segment_audio = waveform[start_sample:end_sample]

        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
            tmp_path = tmp_file.name

        try:
            import soundfile as sf
            sf.write(tmp_path, segment_audio, sample_rate)

            prediction = language_id.classify_file(tmp_path)
            language_code_str = prediction[3][0]
            lang_prefix = language_code_str.split(':')[0].strip()
            confidence = float(prediction[1].exp()[0])
            low_confidence = confidence < confidence_threshold

            if lang_prefix in ('la', 'cy'):
                low_confidence = True
                print("  ⚠️  检测到拉丁语或威尔士语（可能是误识别），将由 Whisper 自动识别")

            segment_info = {
                'segment': (start_ms, end_ms),
                'language': lang_prefix,
                'language_full': language_code_str,
                'confidence': confidence,
                'low_confidence': low_confidence,
            }

            all_language_results.append({
                'segment_index': i + 1,
                'start_time': start_ms / 1000,
                'end_time': end_ms / 1000,
                'duration': (end_ms - start_ms) / 1000,
                'language_code': lang_prefix,
                'language_name': language_code_str,
                'confidence': float(confidence),
                'low_confidence': low_confidence,
            })
            segments_with_language.append(segment_info)

            status = "⚠️ 低置信度(Whisper自动识别)" if low_confidence else f"✓ 将使用 {lang_prefix}"
            print(
                f"  片段{i+1}: {start_ms/1000:.2f}s-{end_ms/1000:.2f}s | "
                f"语言: {language_code_str} | 置信度: {confidence:.2f} | {status}"
            )
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    print("=" * 60)
    low_conf_count = len([r for r in all_language_results if r['low_confidence']])
    high_conf_count = len(all_language_results) - low_conf_count
    print("语言检测完成!")
    print(f"高置信度片段: {high_conf_count} (将使用检测到的语言)")
    if low_conf_count > 0:
        print(f"低置信度片段: {low_conf_count} (将由 Whisper 自动识别)")
    print("=" * 60 + "\n")

    if temp_dir and all_language_results:
        json_path = os.path.join(temp_dir, 'language_detection_results.json')
        json_data = {
            'total_segments': len(segments),
            'high_confidence_segments': high_conf_count,
            'low_confidence_segments': low_conf_count,
            'confidence_threshold': confidence_threshold,
            'detection_results': all_language_results,
        }
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, ensure_ascii=False, indent=2)
        print(f"语言检测结果已保存到: {json_path}\n")

    return segments_with_language
