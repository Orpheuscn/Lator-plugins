import unittest

from fastwhisper_subtitle.pipeline import build_host_segments


class PipelineIdentityTests(unittest.TestCase):
    def test_emits_import_source_item_ids_not_project_segment_ids(self):
        segments = build_host_segments(
            {
                "segments": [
                    {"start": 0, "end": 1, "text": "First"},
                    {"start": 1, "end": 2, "text": "Second"},
                ]
            },
            segment_start_ms=0,
            segment_id_offset=4,
            enable_quality_filter=False,
            model="small",
        )

        self.assertEqual(
            [segment["sourceItemId"] for segment in segments],
            ["5", "6"],
        )
        self.assertTrue(all("segmentId" not in segment for segment in segments))


if __name__ == "__main__":
    unittest.main()
