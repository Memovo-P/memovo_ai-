"""The deterministic fake and the concurrency/deadline wrapper."""

import asyncio

import pytest

from memovo_ai.generation import (
    BoundedGenerationProvider,
    FakeGenerationProvider,
    GenerationMessage,
    GenerationProvider,
    GenerationRequest,
    GenerationResult,
    GenerationTimeoutError,
    GenerationUnavailableError,
)

pytestmark = pytest.mark.unit


def request(text: str = "hello", *, max_output_tokens: int = 64) -> GenerationRequest:
    return GenerationRequest(
        messages=(GenerationMessage(role="user", content=text),),
        max_output_tokens=max_output_tokens,
    )


# --------------------------------------------------------------------------
# Request invariants
# --------------------------------------------------------------------------
def test_a_request_needs_a_message() -> None:
    with pytest.raises(ValueError, match="at least one message"):
        GenerationRequest(messages=(), max_output_tokens=10)


def test_a_request_needs_a_positive_output_budget() -> None:
    with pytest.raises(ValueError, match="max_output_tokens"):
        request(max_output_tokens=0)


def test_the_default_temperature_is_zero() -> None:
    assert request().temperature == 0.0


# --------------------------------------------------------------------------
# The fake
# --------------------------------------------------------------------------
def test_the_fake_is_a_provider() -> None:
    assert isinstance(FakeGenerationProvider(), GenerationProvider)


async def test_scripted_text_is_replayed_in_order() -> None:
    fake = FakeGenerationProvider(["first", "second"])

    assert (await fake.generate(request())).text == "first"
    assert (await fake.generate(request())).text == "second"
    assert len(fake.calls) == 2


async def test_a_scripted_result_is_returned_verbatim() -> None:
    result = GenerationResult(text="{}", finish_reason="length", model="m")
    fake = FakeGenerationProvider([result])

    assert await fake.generate(request()) is result


async def test_a_scripted_exception_is_raised() -> None:
    fake = FakeGenerationProvider([GenerationUnavailableError("down")])

    with pytest.raises(GenerationUnavailableError):
        await fake.generate(request())


async def test_a_responder_sees_the_request() -> None:
    fake = FakeGenerationProvider(responder=lambda r: r.messages[-1].content.upper())

    assert (await fake.generate(request("echo me"))).text == "ECHO ME"


async def test_running_out_of_script_is_a_test_error_not_a_silent_answer() -> None:
    with pytest.raises(RuntimeError, match="no scripted response"):
        await FakeGenerationProvider().generate(request())


async def test_the_fake_reports_token_counts_and_stop() -> None:
    result = await FakeGenerationProvider(["two words"]).generate(request("one two three"))

    assert result.finish_reason == "stop"
    assert result.prompt_tokens == 3
    assert result.completion_tokens == 2


async def test_the_fake_records_closing() -> None:
    fake = FakeGenerationProvider()
    await fake.aclose()

    assert fake.closed is True


# --------------------------------------------------------------------------
# The bounded wrapper
# --------------------------------------------------------------------------
class SlowProvider:
    def __init__(self, delay: float) -> None:
        self.delay = delay
        self.in_flight = 0
        self.peak = 0

    async def generate(self, request: GenerationRequest) -> GenerationResult:
        self.in_flight += 1
        self.peak = max(self.peak, self.in_flight)
        try:
            await asyncio.sleep(self.delay)
        finally:
            self.in_flight -= 1
        return GenerationResult(text="ok", finish_reason="stop", model="slow")


@pytest.mark.parametrize(("concurrency", "timeout"), [(0, 1.0), (1, 0.0), (1, -1.0)])
def test_invalid_bounds_are_rejected(concurrency: int, timeout: float) -> None:
    with pytest.raises(ValueError):
        BoundedGenerationProvider(
            FakeGenerationProvider(), max_concurrency=concurrency, timeout_seconds=timeout
        )


async def test_the_wrapper_is_transparent_on_success() -> None:
    bounded = BoundedGenerationProvider(
        FakeGenerationProvider(["fine"]), max_concurrency=2, timeout_seconds=5
    )

    assert (await bounded.generate(request())).text == "fine"


async def test_concurrency_is_capped() -> None:
    slow = SlowProvider(0.05)
    bounded = BoundedGenerationProvider(slow, max_concurrency=2, timeout_seconds=5)

    await asyncio.gather(*(bounded.generate(request()) for _ in range(6)))

    assert slow.peak == 2


async def test_a_slow_call_times_out_as_a_generation_timeout() -> None:
    bounded = BoundedGenerationProvider(SlowProvider(1.0), max_concurrency=1, timeout_seconds=0.05)

    with pytest.raises(GenerationTimeoutError):
        await bounded.generate(request())


async def test_the_deadline_covers_waiting_for_a_slot() -> None:
    """A request queued behind a busy slot must not wait past the deadline.

    Both calls share one slot and a 0.1 s deadline against a 0.3 s provider:
    the first times out inside the provider, the second times out while
    still waiting for the slot and never reaches the provider at all.
    """
    slow = SlowProvider(0.3)
    bounded = BoundedGenerationProvider(slow, max_concurrency=1, timeout_seconds=0.1)

    first = asyncio.create_task(bounded.generate(request()))
    await asyncio.sleep(0.01)

    with pytest.raises(GenerationTimeoutError):
        await bounded.generate(request())

    with pytest.raises(GenerationTimeoutError):
        await first

    assert slow.peak == 1


async def test_a_timeout_releases_the_slot() -> None:
    slow = SlowProvider(0.2)
    bounded = BoundedGenerationProvider(slow, max_concurrency=1, timeout_seconds=0.05)

    with pytest.raises(GenerationTimeoutError):
        await bounded.generate(request())

    # The slot is free again: a fast provider behind the same wrapper works.
    assert slow.in_flight == 0


async def test_cancellation_propagates_and_releases_the_slot() -> None:
    slow = SlowProvider(1.0)
    bounded = BoundedGenerationProvider(slow, max_concurrency=1, timeout_seconds=5)

    task = asyncio.create_task(bounded.generate(request()))
    await asyncio.sleep(0.01)
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task
    assert slow.in_flight == 0


async def test_a_provider_timeout_is_passed_through_unchanged() -> None:
    fake = FakeGenerationProvider([GenerationTimeoutError("upstream")])
    bounded = BoundedGenerationProvider(fake, max_concurrency=1, timeout_seconds=5)

    with pytest.raises(GenerationTimeoutError, match="upstream"):
        await bounded.generate(request())


async def test_no_retry_is_ever_made() -> None:
    """One upstream attempt per call: the Backend owns retries."""
    fake = FakeGenerationProvider([GenerationUnavailableError("down"), "never reached"])
    bounded = BoundedGenerationProvider(fake, max_concurrency=1, timeout_seconds=5)

    with pytest.raises(GenerationUnavailableError):
        await bounded.generate(request())

    assert len(fake.calls) == 1


async def test_closing_the_wrapper_closes_the_provider() -> None:
    fake = FakeGenerationProvider()
    await BoundedGenerationProvider(fake, max_concurrency=1, timeout_seconds=1).aclose()

    assert fake.closed is True
