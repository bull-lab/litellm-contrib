import litellm
from litellm.secret_managers.main import get_secret_str
from litellm.types.llms.openai import AllMessageValues, ChatCompletionToolParam

from ...openai.chat.gpt_transformation import OpenAIGPTConfig

STEPFUN_API_BASE = "https://api.stepfun.ai/v1"


class StepFunChatConfig(OpenAIGPTConfig):
    @property
    def custom_llm_provider(self) -> str | None:
        return "stepfun"

    def _get_openai_compatible_provider_info(
        self, api_base: str | None, api_key: str | None
    ) -> tuple[str | None, str | None]:
        api_base = api_base or get_secret_str("STEPFUN_API_BASE") or STEPFUN_API_BASE
        dynamic_api_key = api_key or get_secret_str("STEPFUN_API_KEY")
        return api_base, dynamic_api_key

    def remove_cache_control_flag_from_messages_and_tools(
        self,
        model: str,
        messages: list[AllMessageValues],
        tools: list[ChatCompletionToolParam] | None = None,
    ) -> tuple[list[AllMessageValues], list[ChatCompletionToolParam] | None]:
        return messages, tools

    def get_supported_openai_params(self, model: str) -> list[str]:
        base_params = [
            "max_tokens",
            "stream",
            "stream_options",
            "temperature",
            "top_p",
            "stop",
            "tools",
            "tool_choice",
        ]

        try:
            if litellm.supports_reasoning(model=model, custom_llm_provider=self.custom_llm_provider):
                base_params.extend(["thinking", "reasoning_effort"])
        except (AttributeError, KeyError, TypeError, ValueError):
            pass

        return base_params
