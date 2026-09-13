"""Deterministic in-memory generation provider for tests and development.

It never contacts anything. Responses are scripted: either a fixed list
consumed in order, or a function of the request, so a test can state exactly
what "the model" says -- including malformed output and failures -- and then
assert how the service reacts.

It is deliberately not selectable through configuration. A production
deployment with fake generation would serve fabricated answers, so the only
way to get this provider is to construct it in code.
"""

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field

from memovo_ai.generation.models import GenerationRequest, GenerationResult

__all__ = ["FakeGenerationProvider", "ScriptedResponse"]

#: A scripted turn: text to return, or an exception to raise.
type ScriptedResponse = str | GenerationResult | BaseException

FAKE_MODEL = "fake/deterministic"


@dataclass(eq=False)
class FakeGenerationProvider:
    """A :class:`~memovo_ai.generation.base.GenerationProvider` that replays a script."""

    responses: list[ScriptedResponse] = field(default_factory=list)
    responder: Callable[[GenerationRequest], ScriptedResponse] | None = None
    calls: list[GenerationRequest] = field(default_factory=list)
    closed: bool = False

    def __init__(
        self,
        responses: Iterable[ScriptedResponse] = (),
        *,
        responder: Callable[[GenerationRequest], ScriptedResponse] | None = None,
    ) -> None:
        self.responses = list(responses)
        self.responder = responder
        self.calls = []
        self.closed = False

    async def generate(self, request: GenerationRequest) -> GenerationResult:
        self.calls.append(request)

        if self.responder is not None:
            scripted = self.responder(request)
        elif self.responses:
            scripted = self.responses.pop(0)
        else:
            message = "the fake generation provider has no scripted response left"
            raise RuntimeError(message)

        if isinstance(scripted, BaseException):
            raise scripted
        if isinstance(scripted, GenerationResult):
            return scripted

        return GenerationResult(
            text=scripted,
            finish_reason="stop",
            model=FAKE_MODEL,
            prompt_tokens=sum(len(m.content.split()) for m in request.messages),
            completion_tokens=len(scripted.split()),
        )

    async def aclose(self) -> None:
        self.closed = True
