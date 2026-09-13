"""TX selectors that are told which radio a frame arrived on.

A selector used to see only the frame bytes, so a policy like "reply on the
other band" had to read the fabric's most recent RX at send time -- which, after
a retransmit delay, is whatever arrived in the meantime. Selectors may now take
``(data, rx_radio_id)`` instead, which makes the choice a function of the packet
in hand and lets a caller ask for it before the send.

The one-argument shape is still supported and is what an unrecognisable
callable is assumed to be.
"""

from __future__ import annotations

import pytest
from openhop_core.node.dispatcher import Dispatcher, _names_keyword
from openhop_core.protocol import Packet
from openhop_core.protocol.constants import PAYLOAD_TYPE_ADVERT
from openhop_core.protocol.packet_filter import PacketFilter
from openhop_core.rf_fabric import FabricRadio, RFFabric
from openhop_core.rf_fabric.fabric import _selector_takes_rx


class _MockRadio:
    def __init__(self, name: str = "r"):
        self.name = name
        self.rx_callback = None
        self.sent = []

    def set_rx_callback(self, callback):
        self.rx_callback = callback

    async def send(self, data: bytes):
        self.sent.append(data)
        return {"radio": self.name}

    def get_last_rssi(self):
        return -80

    def get_last_snr(self):
        return 7.5


class _LegacyRadio:
    """A radio from before ``rx_radio_id`` existed: one positional argument."""

    def __init__(self):
        self.calls = []

    def set_rx_callback(self, callback):
        self.rx_callback = callback

    async def send(self, data: bytes):
        self.calls.append(data)
        return {"ok": True}


def _bridge_pair():
    a, b = _MockRadio("a"), _MockRadio("b")
    fabric = RFFabric()
    fabric.register_radio(a, radio_id="ra")
    fabric.register_radio(b, radio_id="rb")
    return fabric, a, b


def _advert() -> Packet:
    pkt = Packet()
    pkt.header = PAYLOAD_TYPE_ADVERT << 2
    pkt.payload = bytearray(b"tx-context")
    pkt.payload_len = len(pkt.payload)
    pkt.path_len = 0
    return pkt


class TestSelectorArityDetection:
    def test_one_argument_selector_is_legacy(self):
        assert _selector_takes_rx(lambda data: None) is False

    def test_a_selector_naming_rx_radio_id_takes_rx(self):
        assert _selector_takes_rx(lambda data, rx_radio_id: None) is True

    def test_a_second_parameter_by_another_name_is_the_selectors_own(self):
        # The whole reason this is name-based. Counting positionals reads this
        # as context-aware, hands `options` a radio id in place of its default,
        # and then rejects whatever it returns.
        def choose(data, options=("alpha", "beta")):
            return options[0]

        assert _selector_takes_rx(choose) is False

    def test_star_args_is_not_evidence_of_intent(self):
        assert _selector_takes_rx(lambda *args: None) is False

    def test_kwargs_sponge_is_not_evidence_of_intent(self):
        assert _selector_takes_rx(lambda data, **kwargs: None) is False

    def test_keyword_only_rx_radio_id_counts(self):
        assert _selector_takes_rx(lambda data, *, rx_radio_id=None: None) is True

    def test_bound_method_is_read_by_name_too(self):
        class Policy:
            def one(self, data):
                return None

            def two(self, data, rx_radio_id):
                return None

        policy = Policy()
        assert _selector_takes_rx(policy.one) is False
        assert _selector_takes_rx(policy.two) is True

    def test_unreadable_signature_is_treated_as_legacy(self):
        # A C builtin has no introspectable signature. One argument is what
        # every selector written before this change expects.
        assert _selector_takes_rx(min) is False

    def test_none_clears_the_flag(self):
        fabric, _, _ = _bridge_pair()
        fabric.set_tx_selector(lambda data, rx_radio_id: "rb")
        assert fabric._tx_selector_takes_rx is True
        fabric.set_tx_selector(None)
        assert fabric._tx_selector_takes_rx is False


class TestResolveWithIngressRadio:
    def test_legacy_selector_is_called_with_data_only(self):
        fabric, _, _ = _bridge_pair()
        seen = []

        def _legacy(data):
            seen.append(data)
            return "rb"

        fabric.set_tx_selector(_legacy)
        assert fabric.resolve_tx_radio_id(b"frame", rx_radio_id="rb") == "rb"
        assert seen == [b"frame"]

    def test_context_selector_sees_the_ingress_radio(self):
        fabric, _, _ = _bridge_pair()
        seen = []

        def _bridge(data, rx_radio_id):
            seen.append((data, rx_radio_id))
            ids = [rid for rid in fabric.radios if rid != rx_radio_id]
            return ids[0] if ids else fabric.default_radio_id

        fabric.set_tx_selector(_bridge)
        assert fabric.resolve_tx_radio_id(b"f", rx_radio_id="ra") == "rb"
        assert fabric.resolve_tx_radio_id(b"f", rx_radio_id="rb") == "ra"
        assert seen == [(b"f", "ra"), (b"f", "rb")]

    def test_locally_originated_frames_pass_none(self):
        fabric, _, _ = _bridge_pair()
        seen = []
        fabric.set_tx_selector(lambda data, rx_radio_id: seen.append(rx_radio_id) or None)
        assert fabric.resolve_tx_radio_id(b"f") == "ra"
        assert seen == [None]

    def test_explicit_radio_id_still_wins(self):
        fabric, _, _ = _bridge_pair()
        fabric.set_tx_selector(lambda data, rx_radio_id: "ra")
        assert fabric.resolve_tx_radio_id(b"f", "rb", rx_radio_id="ra") == "rb"

    def test_answer_holds_when_another_radio_receives_in_between(self):
        # The reason for the argument: a selector reading the fabric's latest RX
        # answers about whatever arrived most recently, not about this packet.
        fabric, a, b = _bridge_pair()
        fabric.arm()
        fabric.set_tx_selector(
            lambda data, rx_radio_id: next(
                (rid for rid in fabric.radios if rid != rx_radio_id), None
            )
        )
        planned = fabric.resolve_tx_radio_id(b"f", rx_radio_id="ra")
        b.rx_callback(b"someone else", -90, 1.0)
        assert fabric.last_rx_radio_id == "rb"
        assert fabric.resolve_tx_radio_id(b"f", rx_radio_id="ra") == planned


class TestSendWithIngressRadio:
    @pytest.mark.asyncio
    async def test_send_routes_by_ingress_radio(self):
        fabric, a, b = _bridge_pair()
        fabric.set_tx_selector(
            lambda data, rx_radio_id: next(
                (rid for rid in fabric.radios if rid != rx_radio_id), None
            )
        )
        result = await fabric.send(b"from-a", rx_radio_id="ra")
        assert b.sent == [b"from-a"]
        assert result["radio_id"] == "rb"
        await fabric.send(b"from-b", rx_radio_id="rb")
        assert a.sent == [b"from-b"]

    @pytest.mark.asyncio
    async def test_fabric_radio_passes_the_keyword_through(self):
        a, b = _MockRadio("a"), _MockRadio("b")
        fr = FabricRadio(radios=[(a, "ra"), (b, "rb")], default_radio_id="ra")
        fr.fabric.set_tx_selector(
            lambda data, rx_radio_id: next(
                (rid for rid in fr.fabric.radios if rid != rx_radio_id), None
            )
        )
        await fr.send(b"d", rx_radio_id="ra")
        assert b.sent == [b"d"]
        assert fr.resolve_tx_radio_id(b"d", rx_radio_id="rb") == "ra"


class TestDispatcherThreadsIngressRadio:
    @pytest.mark.asyncio
    async def test_transmit_passes_the_packets_ingress_radio(self):
        a, b = _MockRadio("a"), _MockRadio("b")
        fr = FabricRadio(radios=[(a, "ra"), (b, "rb")], default_radio_id="ra")
        fr.fabric.set_tx_selector(
            lambda data, rx_radio_id: next(
                (rid for rid in fr.fabric.radios if rid != rx_radio_id), None
            )
        )
        dispatcher = Dispatcher(radio=fr, packet_filter=PacketFilter())

        pkt = _advert()
        pkt._rx_radio_id = "ra"
        assert await dispatcher.send_packet(pkt, wait_for_ack=False) is True
        # Heard on ra, so it leaves by rb -- and the metadata says so.
        assert b.sent and not a.sent
        assert pkt._tx_metadata["radio_id"] == "rb"

    @pytest.mark.asyncio
    async def test_locally_originated_packet_uses_the_default(self):
        a, b = _MockRadio("a"), _MockRadio("b")
        fr = FabricRadio(radios=[(a, "ra"), (b, "rb")], default_radio_id="ra")
        fr.fabric.set_tx_selector(
            lambda data, rx_radio_id: next(
                (rid for rid in fr.fabric.radios if rid != rx_radio_id), None
            )
            if rx_radio_id
            else None
        )
        dispatcher = Dispatcher(radio=fr, packet_filter=PacketFilter())
        assert await dispatcher.send_packet(_advert(), wait_for_ack=False) is True
        assert a.sent and not b.sent

    @pytest.mark.asyncio
    async def test_legacy_radio_is_called_exactly_as_before(self):
        radio = _LegacyRadio()
        dispatcher = Dispatcher(radio=radio, packet_filter=PacketFilter())
        pkt = _advert()
        pkt._rx_radio_id = "ra"
        assert await dispatcher.send_packet(pkt, wait_for_ack=False) is True
        assert len(radio.calls) == 1

    def test_names_keyword_ignores_kwargs_sponges(self):
        # A **kwargs wrapper would accept anything offered; that is not the same
        # as doing something with it, and every test double looks like one.
        async def sponge(data, **kwargs):
            return True

        async def explicit(data, *, rx_radio_id=None):
            return True

        assert _names_keyword(sponge, "rx_radio_id") is False
        assert _names_keyword(explicit, "rx_radio_id") is True


class TestLegacySelectorsAreNotMisread:
    """A one-argument selector may take arguments of its own.

    Reading arity as intent classified `def choose(data, options=...)` as
    context-aware, called it with a radio id in place of its default, and then
    raised KeyError on whatever it returned -- losing every transmission on a
    node that had been working.
    """

    @pytest.mark.asyncio
    async def test_a_defaulted_second_parameter_still_gets_one_argument(self):
        fabric, a, b = _bridge_pair()
        calls = []

        def choose(data, options=("ra", "rb")):
            calls.append(options)
            return options[0]

        fabric.set_tx_selector(choose)
        assert fabric.resolve_tx_radio_id(b"f", rx_radio_id="rb") == "ra"
        assert calls == [("ra", "rb")]

        await fabric.send(b"frame", rx_radio_id="rb")
        assert a.sent == [b"frame"]
        assert not b.sent

    def test_a_star_args_selector_is_called_the_way_it_always_was(self):
        fabric, _, _ = _bridge_pair()
        seen = []

        def anything(*args):
            seen.append(args)
            return "rb"

        fabric.set_tx_selector(anything)
        assert fabric.resolve_tx_radio_id(b"f", rx_radio_id="ra") == "rb"
        assert seen == [(b"f",)]

    def test_a_context_selector_receives_it_by_keyword(self):
        fabric, _, _ = _bridge_pair()
        seen = []

        def policy(data, *, rx_radio_id=None):
            seen.append(rx_radio_id)
            return "rb"

        fabric.set_tx_selector(policy)
        assert fabric.resolve_tx_radio_id(b"f", rx_radio_id="ra") == "rb"
        assert seen == ["ra"]
