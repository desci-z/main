from __future__ import annotations

import typer

from talos.services.implementations.paddle_ocr_vl import PaddleOcrVlService

document_app = typer.Typer()


@document_app.command(name="analyze")
def analyze_document(
    document_path: str = typer.Argument(..., help="The path or URL to the document to analyze.")
) -> None:
    """
    Analyzes a document using PaddleOCR-VL.
    """
    service = PaddleOcrVlService()
    result = service.analyze_document(document_path)
    print(result)