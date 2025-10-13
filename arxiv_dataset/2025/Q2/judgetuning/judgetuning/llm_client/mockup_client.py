from dataclasses import dataclass
from typing import List

from openai.types import CompletionUsage
from openai.types.chat import ChatCompletion, ChatCompletionMessage
from openai.types.chat.chat_completion import Choice


@dataclass
class State:
    messages: list[str]
    current_index: int = 0


class MockUpClient:
    """
    A client that mocks OpenAI interface and return current messages from a static message buffer when being called.
    The nested classes are made to support being able to call `MockUpClient().chat.completions.create(...)`.
    This is done since property method only work for classes (ideally we could use functional programming).
    """

    @dataclass
    class Completions:
        state: State

        @dataclass
        class Create:
            state: State

            def create(self, n: int, **kwargs) -> ChatCompletion:
                res = ChatCompletion(
                    id="1",
                    choices=[
                        Choice(
                            finish_reason="stop",
                            index=0,
                            message=ChatCompletionMessage(
                                content=self.state.messages[
                                    i % len(self.state.messages)
                                ],
                                role="assistant",
                            ),
                        )
                        for i in range(
                            self.state.current_index, self.state.current_index + n
                        )
                    ],
                    created=1,
                    model="mockup",
                    object="chat.completion",
                    usage=CompletionUsage(
                        completion_tokens=1, prompt_tokens=1, total_tokens=2
                    ),
                )
                self.state.current_index += n
                return res

        @property
        def completions(self):
            return self.Create(state=self.state)

    def __init__(self, messages: list[str]):
        self.state = State(messages=messages, current_index=0)

    @property
    def chat(self):
        return self.Completions(state=self.state)
