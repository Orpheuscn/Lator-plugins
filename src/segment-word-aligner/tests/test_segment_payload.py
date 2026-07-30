import unittest

from segment_payload import collect_segments, normalize_segment


class SegmentPayloadTests(unittest.TestCase):
    def test_v2_payload_uses_stable_segment_ref_and_projection_fields(self) -> None:
        segment = normalize_segment(
            {
                "requestKey": "segment-0007",
                "segmentRef": {
                    "projectId": "project-stable",
                    "segmentId": "seg_aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa",
                },
                "displayOrdinal": 7,
                "displayLabel": "2.3",
                "sourceText": "alpha",
                "translatedText": "阿尔法",
            },
            0,
        )

        self.assertIsNotNone(segment)
        assert segment is not None
        self.assertEqual(segment.segment_id, "seg_aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa")
        self.assertEqual(segment.segment_index, 6)
        self.assertEqual(segment.display_label, "2.3")

    def test_legacy_payload_remains_readable_during_package_transition(self) -> None:
        segments = collect_segments(
            {
                "segments": [
                    {
                        "segmentId": "seg_bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb",
                        "segmentIndex": 2,
                        "sourceText": "beta",
                        "targetText": "贝塔",
                    }
                ]
            }
        )

        self.assertEqual(len(segments), 1)
        self.assertEqual(segments[0].segment_index, 2)
        self.assertEqual(segments[0].display_label, "3")

    def test_request_key_is_never_persisted_as_segment_identity(self) -> None:
        self.assertIsNone(
            normalize_segment(
                {
                    "requestKey": "segment-0001",
                    "sourceText": "alpha",
                    "targetText": "阿尔法",
                },
                0,
            )
        )


if __name__ == "__main__":
    unittest.main()
