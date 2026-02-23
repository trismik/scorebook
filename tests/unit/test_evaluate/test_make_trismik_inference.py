"""Unit tests for make_trismik_inference() in evaluate_helpers."""

from typing import Any, List
from unittest.mock import MagicMock

import pytest
from trismik.types import TrismikMultipleChoiceTextItem, TrismikOpenEndedTextItem, TrismikTextChoice

from scorebook.evaluate.evaluate_helpers import make_trismik_inference


def mock_inference(items: List[Any], **kwargs: Any) -> List[str]:
    """Mock inference function that returns item count as strings."""
    return [f"response_{i}" for i in range(len(items))]


async def mock_async_inference(items: List[Any], **kwargs: Any) -> List[str]:
    """Mock async inference function."""
    return [f"async_response_{i}" for i in range(len(items))]


class TestSingleMultipleChoiceItem:
    """Tests for single TrismikMultipleChoiceTextItem input."""

    def test_converts_mc_item_to_dict(self):
        mc_item = TrismikMultipleChoiceTextItem(
            id="1",
            question="What is 2+2?",
            choices=[
                TrismikTextChoice(id="A", text="3"),
                TrismikTextChoice(id="B", text="4"),
            ],
        )
        spy = MagicMock(side_effect=mock_inference)
        wrapped = make_trismik_inference(spy)
        wrapped(mc_item)

        # Should have been called with a list containing the dict
        spy.assert_called_once()
        args = spy.call_args[0][0]
        assert isinstance(args, list)
        assert len(args) == 1
        assert args[0] == {
            "id": "1",
            "question": "What is 2+2?",
            "choices": [{"id": "A", "text": "3"}, {"id": "B", "text": "4"}],
        }

    def test_returns_single_value_not_list(self):
        mc_item = TrismikMultipleChoiceTextItem(
            id="1",
            question="Q?",
            choices=[TrismikTextChoice(id="A", text="Yes")],
        )
        wrapped = make_trismik_inference(mock_inference)
        result = wrapped(mc_item)
        assert result == "response_0"

    def test_returns_list_when_return_list_true(self):
        mc_item = TrismikMultipleChoiceTextItem(
            id="1",
            question="Q?",
            choices=[TrismikTextChoice(id="A", text="Yes")],
        )
        wrapped = make_trismik_inference(mock_inference, return_list=True)
        result = wrapped(mc_item)
        assert result == ["response_0"]


class TestSingleOpenEndedItem:
    """Tests for single TrismikOpenEndedTextItem input."""

    def test_converts_oe_item_to_dict(self):
        oe_item = TrismikOpenEndedTextItem(
            id="2",
            question="Explain gravity.",
            reference="force between masses",
            response_text=None,
        )
        spy = MagicMock(side_effect=mock_inference)
        wrapped = make_trismik_inference(spy)
        wrapped(oe_item)

        spy.assert_called_once()
        args = spy.call_args[0][0]
        assert isinstance(args, list)
        assert len(args) == 1
        assert args[0] == {
            "id": "2",
            "question": "Explain gravity.",
            "reference": "force between masses",
            "response_text": None,
        }

    def test_returns_single_value_not_list(self):
        oe_item = TrismikOpenEndedTextItem(id="2", question="Explain gravity.")
        wrapped = make_trismik_inference(mock_inference)
        result = wrapped(oe_item)
        assert result == "response_0"

    def test_returns_list_when_return_list_true(self):
        oe_item = TrismikOpenEndedTextItem(id="2", question="Explain gravity.")
        wrapped = make_trismik_inference(mock_inference, return_list=True)
        result = wrapped(oe_item)
        assert result == ["response_0"]


class TestSingleDictItem:
    """Tests for single dict (Mapping) input."""

    def test_passes_dict_as_list(self):
        dict_item = {"question": "What?", "choices": []}
        spy = MagicMock(side_effect=mock_inference)
        wrapped = make_trismik_inference(spy)
        wrapped(dict_item)

        spy.assert_called_once()
        args = spy.call_args[0][0]
        assert args == [dict_item]

    def test_returns_single_value(self):
        dict_item = {"question": "What?"}
        wrapped = make_trismik_inference(mock_inference)
        result = wrapped(dict_item)
        assert result == "response_0"


class TestIterableOfItems:
    """Tests for iterable of mixed items."""

    def test_converts_mixed_items(self):
        items = [
            TrismikMultipleChoiceTextItem(
                id="1",
                question="Q1",
                choices=[TrismikTextChoice(id="A", text="Yes")],
            ),
            TrismikOpenEndedTextItem(id="2", question="Q2"),
            {"question": "Q3"},
        ]
        spy = MagicMock(side_effect=mock_inference)
        wrapped = make_trismik_inference(spy)
        wrapped(items)

        spy.assert_called_once()
        args = spy.call_args[0][0]
        assert len(args) == 3
        # MC item converted to dict
        assert args[0] == {
            "id": "1",
            "question": "Q1",
            "choices": [{"id": "A", "text": "Yes"}],
        }
        # OE item converted to dict
        assert args[1] == {
            "id": "2",
            "question": "Q2",
            "reference": None,
            "response_text": None,
        }
        # Plain dict passed through
        assert args[2] == {"question": "Q3"}

    def test_returns_full_list(self):
        items = [
            TrismikMultipleChoiceTextItem(
                id="1", question="Q1", choices=[TrismikTextChoice(id="A", text="Yes")]
            ),
            TrismikOpenEndedTextItem(id="2", question="Q2"),
        ]
        wrapped = make_trismik_inference(mock_inference)
        result = wrapped(items)
        assert result == ["response_0", "response_1"]


class TestAsyncInference:
    """Tests for async inference functions."""

    def test_async_with_mc_item(self):
        mc_item = TrismikMultipleChoiceTextItem(
            id="1",
            question="Q?",
            choices=[TrismikTextChoice(id="A", text="Yes")],
        )
        wrapped = make_trismik_inference(mock_async_inference)
        result = wrapped(mc_item)
        assert result == "async_response_0"

    def test_async_with_oe_item(self):
        oe_item = TrismikOpenEndedTextItem(id="2", question="Explain.")
        wrapped = make_trismik_inference(mock_async_inference)
        result = wrapped(oe_item)
        assert result == "async_response_0"

    def test_async_with_iterable(self):
        items = [
            TrismikMultipleChoiceTextItem(
                id="1", question="Q1", choices=[TrismikTextChoice(id="A", text="Yes")]
            ),
            TrismikOpenEndedTextItem(id="2", question="Q2"),
        ]
        wrapped = make_trismik_inference(mock_async_inference)
        result = wrapped(items)
        assert result == ["async_response_0", "async_response_1"]


class TestInvalidInput:
    """Tests for invalid inputs."""

    def test_string_raises_type_error(self):
        wrapped = make_trismik_inference(mock_inference)
        with pytest.raises(TypeError, match="Expected a single item"):
            wrapped("not a valid input")

    def test_int_raises_type_error(self):
        wrapped = make_trismik_inference(mock_inference)
        with pytest.raises(TypeError, match="Expected a single item"):
            wrapped(42)


class TestKwargsPassthrough:
    """Tests that kwargs are forwarded to the inference function."""

    def test_kwargs_forwarded_for_single_item(self):
        mc_item = TrismikMultipleChoiceTextItem(
            id="1", question="Q?", choices=[TrismikTextChoice(id="A", text="Yes")]
        )
        spy = MagicMock(side_effect=mock_inference)
        wrapped = make_trismik_inference(spy)
        wrapped(mc_item, temperature=0.5)

        spy.assert_called_once()
        assert spy.call_args[1] == {"temperature": 0.5}
