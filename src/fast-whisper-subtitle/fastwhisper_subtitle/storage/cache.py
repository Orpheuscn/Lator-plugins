import json
import os
from typing import Dict, List, Optional, Tuple


def check_segment_completed(segment_index: int, output_dir: str) -> bool:
    json_file = os.path.join(output_dir, f"segment_{segment_index:04d}.json")
    return os.path.exists(json_file) and os.path.getsize(json_file) > 0


def load_existing_result(segment_index: int, output_dir: str) -> Dict:
    json_file = os.path.join(output_dir, f"segment_{segment_index:04d}.json")
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            result = json.load(f)
        segment_count = len(result.get('segments', []))
        print(f"    ✓ 使用已有结果: {segment_count} 个字幕片段")
        return result
    except Exception as e:
        print(f"    ⚠️ 读取已有结果失败: {e}")
        return {'segments': []}


def save_segments_cache(temp_dir: str, segments: List[Tuple[float, float]]) -> None:
    cache_file = os.path.join(temp_dir, 'segments_cache.json')
    with open(cache_file, 'w', encoding='utf-8') as f:
        json.dump(segments, f, ensure_ascii=False, indent=2)
    print("片段信息已保存到缓存")


def load_segments_cache(temp_dir: str) -> Optional[List[Tuple[float, float]]]:
    cache_file = os.path.join(temp_dir, 'segments_cache.json')
    if os.path.exists(cache_file):
        try:
            with open(cache_file, 'r', encoding='utf-8') as f:
                segments = json.load(f)
            print(f"✓ 从缓存加载了 {len(segments)} 个片段信息")
            return [tuple(seg) for seg in segments]
        except Exception as e:
            print(f"⚠️ 加载缓存失败: {e}")
    return None
