import unittest

from fastwhisper_subtitle.services.speech_segments import merge_and_pad_speech_segments, merge_speech_segments


class SpeechSegmentsTests(unittest.TestCase):
    def test_merges_using_raw_silence_before_padding(self):
        merged = merge_speech_segments([(1_000, 2_000), (2_600, 3_000)], 0.5)

        self.assertEqual(merged, [(1_000, 2_000), (2_600, 3_000)])

    def test_pads_only_after_merging(self):
        segments = merge_and_pad_speech_segments(
            [(1_000, 2_000), (2_200, 3_000)],
            silence_threshold_sec=0.5,
            speech_pad_ms=300,
            audio_length_ms=5_000,
        )

        self.assertEqual(segments, [(700, 3_300)])


if __name__ == "__main__":
    unittest.main()
