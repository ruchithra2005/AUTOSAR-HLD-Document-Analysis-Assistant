"""Unit tests for Phase 3 LLM client, prompt builder, and error handling.

Uses mocked API clients and responses to guarantee tests run completely offline
without requiring real API keys or incurring any API charges.
"""

import os
import unittest
from unittest.mock import MagicMock, patch

from src.llm.client import (
    GeminiClient,
    GroundedResponse,
    LLMConfigurationError,
    LLMNetworkError,
    LLMQuotaExceededError,
    LLMResponseError,
)
from src.llm.prompts import AUTOSARPromptBuilder
from src.parsers.models import (
    EntityType,
    EvidenceStatus,
    ExtractedEntity,
    ExtractionResult,
    SectionHeading,
)


class TestAUTOSARPromptBuilder(unittest.TestCase):
    def setUp(self):
        self.doc_overview = {
            "metadata": {"filename": "test_hld.pdf"},
            "total_pages": 2,
            "pages": [
                {"page_number": 1, "text": "Component: SWC_EngineControl\n1. ARCHITECTURAL OVERVIEW"},
                {"page_number": 2, "text": "Runnable: RE_CalculateTorque Trigger: 10ms"},
            ],
        }
        self.extraction_result = ExtractionResult(
            source_file="test_hld.pdf",
            total_pages=2,
            sections=[
                SectionHeading(
                    section_number="1",
                    title="ARCHITECTURAL OVERVIEW",
                    page_number=1,
                    source_file="test_hld.pdf",
                    supporting_text="1. ARCHITECTURAL OVERVIEW",
                    level=1,
                )
            ],
            entities=[
                ExtractedEntity(
                    name="SWC_EngineControl",
                    entity_type=EntityType.SWC,
                    page_number=1,
                    source_file="test_hld.pdf",
                    supporting_text="Component: SWC_EngineControl",
                    status=EvidenceStatus.DETECTED,
                    attributes={"component_type": "Application SWC"},
                ),
                ExtractedEntity(
                    name="RE_CalculateTorque",
                    entity_type=EntityType.RUNNABLE,
                    page_number=2,
                    source_file="test_hld.pdf",
                    supporting_text="Runnable: RE_CalculateTorque",
                    status=EvidenceStatus.DETECTED,
                    attributes={"trigger_event": "10ms"},
                ),
            ],
        )

    def test_prompt_contains_grounding_elements(self):
        """Prompt must include document metadata, structured entities, citations, and grounding rules."""
        prompt = AUTOSARPromptBuilder.build_grounded_prompt(
            user_query="What runnables exist?",
            doc_overview=self.doc_overview,
            extraction_result=self.extraction_result,
        )

        # Check metadata & sections
        self.assertIn("Source File: test_hld.pdf", prompt)
        self.assertIn("Section 1: \"ARCHITECTURAL OVERVIEW\"", prompt)

        # Check structured architecture model
        self.assertIn("SWC_EngineControl", prompt)
        self.assertIn("RE_CalculateTorque", prompt)
        self.assertIn("Page: 1", prompt)
        self.assertIn("Page: 2", prompt)

        # Check page text boundaries
        self.assertIn("--- [START OF PAGE 1 (test_hld.pdf)] ---", prompt)
        self.assertIn("--- [START OF PAGE 2 (test_hld.pdf)] ---", prompt)

        # Check user query
        self.assertIn("What runnables exist?", prompt)

        # Check strict grounding system instructions
        self.assertIn("Base your answers ONLY on the supplied HLD document", AUTOSARPromptBuilder.SYSTEM_INSTRUCTION)
        self.assertIn("Every factual assertion MUST cite its location", AUTOSARPromptBuilder.SYSTEM_INSTRUCTION)
        self.assertIn("DO NOT fabricate, guess, or assume missing parameters", AUTOSARPromptBuilder.SYSTEM_INSTRUCTION)


class TestGeminiClient(unittest.TestCase):
    def setUp(self):
        self.doc_overview = {
            "metadata": {"filename": "test_hld.pdf"},
            "total_pages": 1,
            "pages": [{"page_number": 1, "text": "Sample text"}],
        }
        self.extraction_result = ExtractionResult(
            source_file="test_hld.pdf",
            total_pages=1,
            sections=[],
            entities=[],
        )

    def test_missing_api_key_raises_configuration_error(self):
        """Client should raise LLMConfigurationError if no API key is provided and env is empty."""
        with patch.dict(os.environ, {}, clear=True):
            # Ensure GEMINI_API_KEY is not in env
            if "GEMINI_API_KEY" in os.environ:
                del os.environ["GEMINI_API_KEY"]
            with self.assertRaises(LLMConfigurationError):
                GeminiClient(api_key=None)

    def test_client_init_with_explicit_key(self):
        """Client should initialize without errors when key is passed explicitly."""
        with patch("google.genai.Client") as mock_genai_class:
            client = GeminiClient(api_key="test_fake_key_12345")
            self.assertEqual(client.model, "gemini-2.5-flash")
            mock_genai_class.assert_called_once_with(api_key="test_fake_key_12345")

    def test_generate_grounded_response_success(self):
        """Successful response generation using a mocked Gemini client."""
        mock_client_instance = MagicMock()
        mock_response = MagicMock()
        mock_response.text = (
            "**Direct Answer**:\nThe HLD specifies SWC_EngineControl.\n\n"
            "**Explicit Document Facts**:\n- SWC_EngineControl [Source: test_hld.pdf, Page: 1]\n"
        )
        mock_client_instance.models.generate_content.return_value = mock_response

        client = GeminiClient(client_instance=mock_client_instance)
        response: GroundedResponse = client.generate_grounded_response(
            user_query="What SWCs are present?",
            doc_overview=self.doc_overview,
            extraction_result=self.extraction_result,
        )

        self.assertTrue(response.is_successful)
        self.assertIn("SWC_EngineControl", response.content)
        self.assertEqual(response.model_name, "gemini-2.5-flash")
        mock_client_instance.models.generate_content.assert_called_once()

    def test_empty_query_returns_safe_failure(self):
        """Empty or whitespace queries return safe failure without making API calls."""
        mock_client_instance = MagicMock()
        client = GeminiClient(client_instance=mock_client_instance)

        response = client.generate_grounded_response(
            user_query="   ",
            doc_overview=self.doc_overview,
            extraction_result=self.extraction_result,
        )
        self.assertFalse(response.is_successful)
        self.assertIn("Empty query", response.error_message)
        mock_client_instance.models.generate_content.assert_not_called()

    def test_quota_exceeded_error_handling(self):
        """Handles 429 quota exhaustion gracefully by raising LLMQuotaExceededError."""
        from google.genai.errors import ClientError

        mock_client_instance = MagicMock()
        mock_client_instance.models.generate_content.side_effect = ClientError(
            429, {"error": {"message": "RESOURCE_EXHAUSTED: Quota exceeded", "code": 429}}
        )

        client = GeminiClient(client_instance=mock_client_instance)
        with self.assertRaises(LLMQuotaExceededError):
            client.generate_grounded_response(
                user_query="What is the ASIL rating?",
                doc_overview=self.doc_overview,
                extraction_result=self.extraction_result,
            )

    def test_network_error_handling(self):
        """Handles socket/connection errors gracefully by raising LLMNetworkError."""
        mock_client_instance = MagicMock()
        mock_client_instance.models.generate_content.side_effect = ConnectionError("Connection reset by peer")

        client = GeminiClient(client_instance=mock_client_instance)
        with self.assertRaises(LLMNetworkError):
            client.generate_grounded_response(
                user_query="What are the ports?",
                doc_overview=self.doc_overview,
                extraction_result=self.extraction_result,
            )

    def test_empty_response_text_handling(self):
        """Raises LLMResponseError if Gemini returns empty text."""
        mock_client_instance = MagicMock()
        mock_response = MagicMock()
        mock_response.text = ""  # Blocked or empty response
        mock_client_instance.models.generate_content.return_value = mock_response

        client = GeminiClient(client_instance=mock_client_instance)
        with self.assertRaises(LLMResponseError):
            client.generate_grounded_response(
                user_query="Tell me about the design",
                doc_overview=self.doc_overview,
                extraction_result=self.extraction_result,
            )


if __name__ == "__main__":
    unittest.main()
