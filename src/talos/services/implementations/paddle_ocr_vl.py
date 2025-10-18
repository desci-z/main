from __future__ import annotations

import json
from paddleocr import PaddleOCR
from pydantic import ConfigDict

from talos.services.abstract.service import Service


class PaddleOcrVlService(Service):
    """
    A service for analyzing documents with PaddleOCR-VL.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)
    ocr: PaddleOCR | None = None

    def model_post_init(self, __context: any) -> None:
        if self.ocr is None:
            self.ocr = PaddleOCR()

    @property
    def name(self) -> str:
        return "paddle_ocr_vl"

    def analyze_document(self, document_path: str) -> str:
        """
        Analyzes a document using PaddleOCR-VL and returns the results as a JSON string.

        Args:
            document_path: The path or URL to the document to analyze.

        Returns:
            A JSON string containing the analysis results.
        """
        if self.ocr is None:
            raise RuntimeError("OCR client not initialized")
        result = self.ocr.ocr(document_path, cls=True)
        return json.dumps(result)