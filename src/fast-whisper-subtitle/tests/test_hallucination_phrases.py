import unittest

from fastwhisper_subtitle.pipeline import is_hallucination_text


class HallucinationPhraseTests(unittest.TestCase):
    def test_filters_added_chinese_and_french_credit_phrases(self):
        for phrase in (
            "中文字幕 李宗盛",
            "字幕志愿者 李宗盛",
            "Sous-titrage Société Radio-Canada",
        ):
            with self.subTest(phrase=phrase):
                self.assertTrue(is_hallucination_text(phrase))


if __name__ == "__main__":
    unittest.main()
