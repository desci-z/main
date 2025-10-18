from __future__ import annotations

import json
from unittest.mock import create_autospec

from paddleocr import PaddleOCR
from talos.services.implementations.paddle_ocr_vl import PaddleOcrVlService


def test_analyze_document() -> None:
    """
    Tests that the analyze_document method returns the expected result.
    """
    # Arrange
    mock_ocr_client = create_autospec(PaddleOCR)
    mock_ocr_client.ocr.return_value = [
        [
            [["1"], ["a", 0.1]],
            [["2"], ["b", 0.2]],
        ]
    ]

    service = PaddleOcrVlService(ocr=mock_ocr_client)
    document_path = "path/to/document.png"

    # Act
    result = service.analyze_document(document_path)

    # Assert
    mock_ocr_client.ocr.assert_called_once_with(document_path, cls=True)
    assert json.loads(result) == [
        [
            [["1"], ["a", 0.1]],
            [["2"], ["b", 0.2]],
        ]
    ]