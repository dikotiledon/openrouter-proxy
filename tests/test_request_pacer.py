import unittest
from unittest.mock import patch


class FakeClock:
    def __init__(self, start: float = 100.0):
        self.now = start
        self.sleep_calls: list[float] = []

    def monotonic(self) -> float:
        return self.now

    async def sleep(self, delay: float) -> None:
        self.sleep_calls.append(delay)
        self.now += delay


class ProviderRequestPacerTests(unittest.IsolatedAsyncioTestCase):
    async def test_wait_skips_sleep_when_delay_disabled(self):
        from request_pacer import ProviderRequestPacer

        clock = FakeClock()
        pacer = ProviderRequestPacer(0, label="test")

        with patch("request_pacer.time.monotonic", new=clock.monotonic), patch(
            "request_pacer.asyncio.sleep",
            new=clock.sleep,
        ):
            await pacer.wait()
            await pacer.wait()

        self.assertEqual(clock.sleep_calls, [])
        self.assertEqual(clock.now, 100.0)

    async def test_wait_enforces_spacing_between_calls(self):
        from request_pacer import ProviderRequestPacer

        clock = FakeClock()
        pacer = ProviderRequestPacer(25, label="test")

        with patch("request_pacer.time.monotonic", new=clock.monotonic), patch(
            "request_pacer.asyncio.sleep",
            new=clock.sleep,
        ):
            await pacer.wait()
            await pacer.wait()
            await pacer.wait()

        self.assertEqual(clock.sleep_calls, [25.0, 25.0])
        self.assertEqual(clock.now, 150.0)
