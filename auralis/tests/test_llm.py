"""
LLM tests. Requires Ollama running locally with the configured model pulled.
Run with: pytest tests/test_llm.py
"""

import pytest
from backend.services import llm_service


def test_basic_question():
    history = llm_service.ConversationHistory()
    prompt = llm_service.build_prompt("What is artificial intelligence?", "neutral", 0.5, history)
    response = "".join(llm_service.stream_llm_response(prompt))
    assert len(response) > 0
    print("Response:", response)


def test_conversation_context_is_included():
    history = llm_service.ConversationHistory()
    history.add("My name is Alex.", "Nice to meet you, Alex.")
    prompt = llm_service.build_prompt("What's my name?", "neutral", 0.5, history)
    assert "Alex" in prompt


def test_emotion_context_is_included_in_prompt():
    history = llm_service.ConversationHistory()
    prompt = llm_service.build_prompt("I don't know what to do!", "fearful", 0.82, history)
    assert "fearful" in prompt
    assert "0.82" in prompt
