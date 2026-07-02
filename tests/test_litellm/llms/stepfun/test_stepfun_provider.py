import math

import pytest
import respx

import litellm
from litellm import completion
from litellm.cost_calculator import cost_per_token


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


def test_get_llm_provider_stepfun():
    from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

    model, provider, api_key, api_base = get_llm_provider("stepfun/step-3.5-flash")
    assert model == "step-3.5-flash"
    assert provider == "stepfun"
    assert api_base == "https://api.stepfun.ai/v1"


def test_stepfun_in_provider_lists():
    assert "stepfun" in litellm.openai_compatible_providers
    assert "stepfun" in litellm.provider_list


def test_stepfun_models_in_model_cost():
    import os

    os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
    litellm.model_cost = litellm.get_model_cost_map(url="")

    stepfun_models = [
        "stepfun/step-3.7-flash",
        "stepfun/step-3.5-flash",
        "stepfun/step-3.5-flash-2603",
        "stepfun/stepaudio-2.5-tts",
        "stepfun/step-tts-2",
        "stepfun/stepaudio-2.5-asr",
        "stepfun/stepaudio-2.5-tts-voice-clone",
        "stepfun/step-tts-2-voice-clone",
    ]

    for model in stepfun_models:
        assert model in litellm.model_cost, f"Model {model} not found in model_cost"
        assert litellm.model_cost[model]["litellm_provider"] == "stepfun"


def test_stepfun_step_35_flash_cost_calculation():
    import os

    os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
    litellm.model_cost = litellm.get_model_cost_map(url="")

    prompt_cost, completion_cost = cost_per_token(
        model="stepfun/step-3.5-flash",
        prompt_tokens=1000000,
        completion_tokens=1000000,
    )

    assert math.isclose(prompt_cost, 0.10, rel_tol=1e-6)
    assert math.isclose(completion_cost, 0.30, rel_tol=1e-6)


def test_stepfun_step_37_flash_cost_calculation():
    import os

    os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
    litellm.model_cost = litellm.get_model_cost_map(url="")

    prompt_cost, completion_cost = cost_per_token(
        model="stepfun/step-3.7-flash",
        prompt_tokens=1000000,
        completion_tokens=1000000,
    )

    assert math.isclose(prompt_cost, 0.20, rel_tol=1e-6)
    assert math.isclose(completion_cost, 1.15, rel_tol=1e-6)


def test_stepfun_step_35_flash_cache_hit_cost():
    import os

    os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
    litellm.model_cost = litellm.get_model_cost_map(url="")

    prompt_cost, _ = cost_per_token(
        model="stepfun/step-3.5-flash",
        prompt_tokens=0,
        completion_tokens=0,
        cache_read_input_tokens=1000000,
    )

    assert math.isclose(prompt_cost, 0.02, rel_tol=1e-6)


def test_stepfun_reasoning_models_support_reasoning():
    import os

    os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
    litellm.model_cost = litellm.get_model_cost_map(url="")

    for model in ("stepfun/step-3.5-flash", "stepfun/step-3.5-flash-2603", "stepfun/step-3.7-flash"):
        assert litellm.model_cost[model]["supports_reasoning"] is True


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
