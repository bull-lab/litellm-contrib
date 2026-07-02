import os
from unittest.mock import patch

import pytest
import respx

import litellm
from litellm import completion

STEPFUN_API_BASE = "https://api.stepfun.ai/v1"


def test_stepfun_json_registry():
    from litellm.llms.openai_like.json_loader import JSONProviderRegistry

    assert litellm.LlmProviders.STEPFUN.value == "stepfun"
    assert litellm.LlmProviders("stepfun") == litellm.LlmProviders.STEPFUN
    assert JSONProviderRegistry.exists("stepfun")
    config = JSONProviderRegistry.get("stepfun")
    assert config is not None
    assert config.base_url == STEPFUN_API_BASE
    assert config.api_key_env == "STEPFUN_API_KEY"
    assert config.api_base_env == "STEPFUN_API_BASE"


def test_stepfun_dynamic_config_env_vars():
    from litellm.llms.openai_like.dynamic_config import create_config_class
    from litellm.llms.openai_like.json_loader import JSONProviderRegistry

    config = create_config_class(JSONProviderRegistry.get("stepfun"))()

    with patch.dict(
        os.environ,
        {
            "STEPFUN_API_KEY": "test-key",
            "STEPFUN_API_BASE": "https://custom.stepfun.example/v1",
        },
    ):
        api_base, api_key = config._get_openai_compatible_provider_info(None, None)

    assert api_base == "https://custom.stepfun.example/v1"
    assert api_key == "test-key"


def test_stepfun_provider_detection_by_prefix():
    from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

    model, provider, _, api_base = get_llm_provider("stepfun/step-3.5-flash")

    assert model == "step-3.5-flash"
    assert provider == "stepfun"
    assert api_base == STEPFUN_API_BASE


def test_stepfun_chat_complete_url():
    from litellm.llms.openai_like.dynamic_config import create_config_class
    from litellm.llms.openai_like.json_loader import JSONProviderRegistry

    config = create_config_class(JSONProviderRegistry.get("stepfun"))()

    assert (
        config.get_complete_url(
            api_base=None,
            api_key=None,
            model="step-3.5-flash",
            optional_params={},
            litellm_params={},
        )
        == "https://api.stepfun.ai/v1/chat/completions"
    )


@pytest.fixture
def stepfun_response():
    return {
        "id": "chatcmpl-stepfun-123",
        "object": "chat.completion",
        "created": 1677652288,
        "model": "step-3.5-flash",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Hello from StepFun",
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 15, "total_tokens": 25},
    }


@pytest.mark.asyncio
async def test_stepfun_completion_call(respx_mock, stepfun_response, monkeypatch):
    monkeypatch.setenv("STEPFUN_API_KEY", "test-api-key")
    litellm.disable_aiohttp_transport = True

    respx_mock.post("https://api.stepfun.ai/v1/chat/completions").respond(json=stepfun_response)

    response = await litellm.acompletion(
        model="stepfun/step-3.5-flash",
        messages=[{"role": "user", "content": "Hello"}],
        max_tokens=20,
    )

    assert response.choices[0].message.content == "Hello from StepFun"
    assert response.usage.total_tokens == 25

    assert len(respx_mock.calls) == 1
    request = respx_mock.calls[0].request
    assert request.method == "POST"
    assert "api.stepfun.ai" in str(request.url)
    assert request.headers["Authorization"] == "Bearer test-api-key"


def test_stepfun_sync_completion(respx_mock, stepfun_response, monkeypatch):
    monkeypatch.setenv("STEPFUN_API_KEY", "test-api-key")
    litellm.disable_aiohttp_transport = True

    respx_mock.post("https://api.stepfun.ai/v1/chat/completions").respond(json=stepfun_response)

    response = completion(
        model="stepfun/step-3.5-flash",
        messages=[{"role": "user", "content": "Hello"}],
        max_tokens=20,
    )

    assert response.choices[0].message.content == "Hello from StepFun"
    assert response.usage.total_tokens == 25
