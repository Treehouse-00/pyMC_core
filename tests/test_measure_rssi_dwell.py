"""
Tests for SX1262Radio.measure_rssi_dwell: the measurement half of a spectrum sweep.

The radio is built with every hardware boundary mocked, the same way
test_sx1262_wrapper_concurrency.py does it, so these run without an SX1262.
"""

import asyncio
from unittest.mock import MagicMock, patch

import pytest

IRQ_NONE = 0x0000
IRQ_RX_DONE = 0x0002
IRQ_PREAMBLE_DETECTED = 0x0004
IRQ_SYNC_WORD_VALID = 0x0008
IRQ_HEADER_VALID = 0x0010
IRQ_HEADER_ERR = 0x0020
IRQ_CRC_ERR = 0x0040
IRQ_TIMEOUT = 0x0200

MESH_FREQ = 910_525_000
SWEEP_FREQ = 903_400_000


def _make_mock_lora(rssi_raw_values) -> MagicMock:
    lora = MagicMock()
    lora.IRQ_NONE = IRQ_NONE
    lora.IRQ_RX_DONE = IRQ_RX_DONE
    lora.IRQ_PREAMBLE_DETECTED = IRQ_PREAMBLE_DETECTED
    lora.IRQ_SYNC_WORD_VALID = IRQ_SYNC_WORD_VALID
    lora.IRQ_HEADER_VALID = IRQ_HEADER_VALID
    lora.IRQ_HEADER_ERR = IRQ_HEADER_ERR
    lora.IRQ_CRC_ERR = IRQ_CRC_ERR
    lora.IRQ_TIMEOUT = IRQ_TIMEOUT
    lora.STANDBY_RC = 0x00
    lora.RX_CONTINUOUS = 0xFFFFFF
    # getRssiInst returns the raw register value; the wrapper converts -(raw/2).
    values = list(rssi_raw_values)

    def _next_raw():
        if not values:
            return 200  # -100 dBm, a plausible quiet floor
        return values.pop(0)

    lora.getRssiInst.side_effect = _next_raw
    return lora


@pytest.fixture(autouse=True)
def _reset_singleton():
    from openhop_core.hardware.sx1262_wrapper import SX1262Radio

    SX1262Radio._active_instance = None
    yield
    SX1262Radio._active_instance = None


async def _make_radio(rssi_raw_values=()):
    with (
        patch("openhop_core.hardware.sx1262_wrapper.GPIOPinManager", return_value=MagicMock()),
        patch("openhop_core.hardware.sx1262_wrapper.set_gpio_manager"),
    ):
        from openhop_core.hardware.sx1262_wrapper import SX1262Radio

        radio = SX1262Radio(radio_timing_delay=0.0, frequency=MESH_FREQ)
    radio.lora = _make_mock_lora(rssi_raw_values)
    radio._initialized = True
    radio._interrupt_setup = True
    radio._gpio_manager = MagicMock()
    radio._event_loop = asyncio.get_running_loop()
    return radio


async def test_dwell_returns_converted_samples_and_counts_unsettled_reads():
    # raw 200 -> -100 dBm (kept); raw 1 -> -0.5 dBm (>= -1, unsettled, discarded);
    # raw 253 -> -126.5 dBm (<= -126, bus noise, discarded); raw 180 -> -90 (kept).
    radio = await _make_radio([200, 1, 253, 180])

    result = await radio.measure_rssi_dwell(
        SWEEP_FREQ, dwell_s=0.02, sample_gap_s=0.0, settle_s=0.0
    )

    assert "error" not in result
    assert result["freq_hz"] == SWEEP_FREQ
    assert result["n"] == len(result["samples"])
    assert result["samples"][:2] == [-100.0, -90.0]
    assert result["discarded"] >= 2
    assert all(-126.0 < v < -1.0 for v in result["samples"])


async def test_dwell_retunes_then_restores_the_mesh_frequency():
    radio = await _make_radio()

    await radio.measure_rssi_dwell(SWEEP_FREQ, dwell_s=0.005, sample_gap_s=0.0, settle_s=0.0)

    freqs = [c.args[0] for c in radio.lora.setFrequency.call_args_list]
    # First retune is to the sweep channel, the last call puts the mesh back.
    assert freqs[0] == SWEEP_FREQ
    assert freqs[-1] == MESH_FREQ
    # The configured mesh frequency is never rewritten by a sweep.
    assert radio.frequency == MESH_FREQ


async def test_dwell_quiets_irq_routes_during_and_restores_rx_routes_after():
    radio = await _make_radio()
    lora = radio.lora
    rx_mask = radio._get_rx_irq_mask()
    log: list[tuple[str, tuple]] = []

    # The driver's request() re-arms the full RX mask on its way into RX (see
    # SX126x.request -> _irqSetup), which is exactly what a quiet written
    # before it would be undone by. The mock does the same, so this test can
    # only pass if the routes are quieted after the chip is in RX.
    lora.request.side_effect = lambda _timeout: (
        lora.setDioIrqParams(rx_mask, rx_mask, IRQ_NONE, IRQ_NONE) or True
    )
    lora.setDioIrqParams.side_effect = lambda *args: log.append(("irq", args))
    original = lora.getRssiInst.side_effect

    def _sample():
        log.append(("sample", ()))
        return original()

    lora.getRssiInst.side_effect = _sample

    await radio.measure_rssi_dwell(SWEEP_FREQ, dwell_s=0.005, sample_gap_s=0.0, settle_s=0.0)

    first_sample = next(i for i, (kind, _) in enumerate(log) if kind == "sample")
    irq_before = [args for kind, args in log[:first_sample] if kind == "irq"]
    assert irq_before, "IRQ routes were never touched"
    assert irq_before[-1] == (IRQ_NONE,) * 4, "routes were live when sampling began"
    irq_after = [args for kind, args in log[first_sample:] if kind == "irq"]
    assert irq_after[-1] == (rx_mask, rx_mask, IRQ_NONE, IRQ_NONE), "RX routes not restored"
    # And the radio is listening on the mesh again.
    assert lora.request.call_args_list[-1].args == (lora.RX_CONTINUOUS,)


async def test_dwell_holds_and_releases_the_tx_lock():
    radio = await _make_radio()
    seen_locked = []

    original = radio.lora.getRssiInst.side_effect

    def _spy():
        seen_locked.append(radio._tx_lock.locked())
        return original()

    radio.lora.getRssiInst.side_effect = _spy

    await radio.measure_rssi_dwell(SWEEP_FREQ, dwell_s=0.01, sample_gap_s=0.0, settle_s=0.0)

    assert seen_locked and all(seen_locked), "TX lock was not held while sampling"
    assert not radio._tx_lock.locked(), "TX lock leaked after the dwell"


async def test_dwell_gives_up_cleanly_when_a_transmit_holds_the_lock():
    radio = await _make_radio()
    await radio._tx_lock.acquire()
    try:
        result = await radio.measure_rssi_dwell(
            SWEEP_FREQ, dwell_s=0.01, sample_gap_s=0.0, lock_timeout=0.01
        )
    finally:
        radio._tx_lock.release()

    assert result == {"error": "waited_for_tx_lock_timeout", "freq_hz": SWEEP_FREQ}
    # It never touched the hardware while someone else owned it.
    radio.lora.setFrequency.assert_not_called()


async def test_dwell_restores_even_when_sampling_raises():
    radio = await _make_radio()
    radio.lora.getRssiInst.side_effect = RuntimeError("spi fault")

    with pytest.raises(RuntimeError, match="spi fault"):
        await radio.measure_rssi_dwell(SWEEP_FREQ, dwell_s=0.01, sample_gap_s=0.0, settle_s=0.0)

    assert radio.lora.setFrequency.call_args_list[-1].args[0] == MESH_FREQ
    assert not radio._tx_lock.locked()


async def test_dwell_requires_an_initialised_radio():
    radio = await _make_radio()
    radio._initialized = False
    with pytest.raises(RuntimeError, match="not initialized"):
        await radio.measure_rssi_dwell(SWEEP_FREQ, dwell_s=0.001)
