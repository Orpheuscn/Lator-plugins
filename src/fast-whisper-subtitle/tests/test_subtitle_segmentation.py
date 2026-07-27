import unittest

from fastwhisper_subtitle.services.subtitle_segmentation import (
    SubtitleSegmentationOptions,
    should_request_word_timestamps,
    split_whisper_segment,
)


class SubtitleSegmentationTests(unittest.TestCase):
    def setUp(self):
        self.options = SubtitleSegmentationOptions(
            mode="auto",
            max_duration_seconds=7,
            target_duration_seconds=4.5,
            min_duration_seconds=1,
            max_words=20,
            max_characters=80,
            min_pause_seconds=0.25,
        )

    def test_requests_word_timestamps_only_for_long_auto_chunks(self):
        self.assertFalse(should_request_word_timestamps("auto", 7, 7))
        self.assertTrue(should_request_word_timestamps("auto", 7.01, 7))
        self.assertTrue(should_request_word_timestamps("readable", 1, 7))
        self.assertFalse(should_request_word_timestamps("pause", 30, 7))

    def test_splits_long_english_segment_at_sentence_boundaries(self):
        segment = {
            "start": 0,
            "end": 12,
            "text": "One short sentence. Another short sentence. Final sentence.",
            "words": [
                {"word": " One", "start": 0, "end": 0.5},
                {"word": " short", "start": 0.5, "end": 1},
                {"word": " sentence.", "start": 1, "end": 2.5},
                {"word": " Another", "start": 3, "end": 3.5},
                {"word": " short", "start": 3.5, "end": 4},
                {"word": " sentence.", "start": 4, "end": 5.5},
                {"word": " Final", "start": 8, "end": 8.5},
                {"word": " sentence.", "start": 8.5, "end": 10},
            ],
        }

        cues = split_whisper_segment(segment, self.options)

        self.assertEqual([cue["text"] for cue in cues], [
            "One short sentence. Another short sentence.",
            "Final sentence.",
        ])
        self.assertEqual([(cue["start"], cue["end"]) for cue in cues], [(0, 5.5), (8, 10)])

    def test_uses_a_word_boundary_when_there_is_no_punctuation_or_pause(self):
        segment = {
            "start": 0,
            "end": 10,
            "text": "one two three four five six seven eight nine ten",
            "words": [
                {"word": f" {word}", "start": index, "end": index + 0.8}
                for index, word in enumerate("one two three four five six seven eight nine ten".split())
            ],
        }

        cues = split_whisper_segment(segment, self.options)

        self.assertGreater(len(cues), 1)
        self.assertTrue(all(cue["end"] - cue["start"] <= 7 for cue in cues))
        self.assertEqual(" ".join(cue["text"] for cue in cues), segment["text"])

    def test_splits_chinese_without_inserting_spaces(self):
        segment = {
            "start": 0,
            "end": 9,
            "text": "这是第一句话。这里是第二句话。最后一句话。",
            "words": [
                {"word": "这是", "start": 0, "end": 1},
                {"word": "第一句话。", "start": 1, "end": 2.5},
                {"word": "这里是", "start": 3, "end": 4},
                {"word": "第二句话。", "start": 4, "end": 5.5},
                {"word": "最后一句话。", "start": 7, "end": 8.5},
            ],
        }

        cues = split_whisper_segment(segment, self.options)

        self.assertEqual("".join(cue["text"] for cue in cues), segment["text"])
        self.assertTrue(all(" " not in cue["text"] for cue in cues))

    def test_keeps_short_and_legacy_segments_unchanged(self):
        segment = {"start": 0, "end": 3, "text": "Already readable."}

        self.assertEqual(split_whisper_segment(segment, self.options), [segment])


if __name__ == "__main__":
    unittest.main()
