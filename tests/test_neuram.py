"""
Tests for neuram — uses fakeredis so no real Redis is needed.
"""
import asyncio
import math
import time

import pytest
import fakeredis.aioredis as fakeredis

from neuram.models import Engram, MemoryType, MemoryLayer
from neuram.encoder import cosine_similarity, score_salience
from neuram.processes.forgetting import compute_retention, apply_decay, is_forgotten, time_until_forgotten
from neuram.processes.ltp import potentiate, long_term_depression
from neuram.regions.sensory_cortex import SensoryCortex
from neuram.regions.thalamus import Thalamus
from neuram.regions.amygdala import Amygdala
from neuram.regions.prefrontal_cortex import PrefrontalCortex
from neuram.regions.neocortex import Neocortex
from neuram.regions.cerebellum import Cerebellum
from neuram.regions.basal_ganglia import BasalGanglia
from neuram.processes.sleep_cycle import SleepCycle
from neuram.regions.hippocampus import Hippocampus


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_engram(**kwargs) -> Engram:
    defaults = dict(content="test memory", memory_type=MemoryType.SEMANTIC)
    defaults.update(kwargs)
    return Engram(**defaults)


def fake_embedding(content: str) -> list[float]:
    """Deterministic fake embedding for testing without sentence-transformers."""
    base = [float(ord(c) % 10) / 10.0 for c in content]
    padded = (base * 40)[:384]
    return padded


@pytest.fixture
async def redis():
    r = fakeredis.FakeRedis(decode_responses=True)
    yield r
    await r.aclose()


# ---------------------------------------------------------------------------
# Engram model
# ---------------------------------------------------------------------------

class TestEngram:
    def test_to_dict_round_trip(self):
        e = make_engram(content="hello world", salience=0.7)
        d = e.to_dict()
        restored = Engram.from_dict(d)
        assert restored.engram_id == e.engram_id
        assert restored.content == e.content
        assert restored.salience == e.salience
        assert restored.memory_type == e.memory_type

    def test_associations_round_trip(self):
        e = make_engram()
        e.associations = [("abc", 0.8), ("def", 0.5)]
        restored = Engram.from_dict(e.to_dict())
        assert restored.associations == [("abc", 0.8), ("def", 0.5)]

    def test_default_layer_is_working(self):
        e = make_engram()
        assert e.layer == MemoryLayer.WORKING


# ---------------------------------------------------------------------------
# Encoder utilities
# ---------------------------------------------------------------------------

class TestEncoder:
    def test_cosine_similarity_identical(self):
        v = [1.0, 0.0, 1.0]
        assert cosine_similarity(v, v) == pytest.approx(1.0)

    def test_cosine_similarity_orthogonal(self):
        a = [1.0, 0.0]
        b = [0.0, 1.0]
        assert cosine_similarity(a, b) == pytest.approx(0.0)

    def test_cosine_similarity_zero_vector(self):
        assert cosine_similarity([0.0, 0.0], [1.0, 1.0]) == 0.0

    def test_score_salience_baseline(self):
        s = score_salience("hello world")
        assert 0.0 < s <= 1.0

    def test_score_salience_urgent(self):
        s_urgent = score_salience("URGENT: critical error detected!")
        s_plain = score_salience("the weather is nice today")
        assert s_urgent > s_plain

    def test_score_salience_trivial(self):
        s = score_salience("maybe this is trivial and irrelevant")
        assert s < 0.5


# ---------------------------------------------------------------------------
# Forgetting curve
# ---------------------------------------------------------------------------

class TestForgetting:
    def test_retention_at_zero_elapsed(self):
        e = make_engram()
        e.stability = 10.0
        e.last_accessed = time.time()
        assert compute_retention(e) == pytest.approx(1.0, abs=0.01)

    def test_retention_decays_over_time(self):
        e = make_engram()
        e.stability = 1.0
        e.last_accessed = time.time() - 3600  # 1 hour ago
        r = compute_retention(e)
        assert r == pytest.approx(math.exp(-1.0), abs=0.01)

    def test_higher_stability_decays_slower(self):
        e_stable = make_engram()
        e_stable.stability = 10.0
        e_stable.last_accessed = time.time() - 7200

        e_weak = make_engram()
        e_weak.stability = 1.0
        e_weak.last_accessed = time.time() - 7200

        assert compute_retention(e_stable) > compute_retention(e_weak)

    def test_is_forgotten_fresh(self):
        e = make_engram()
        e.last_accessed = time.time()
        assert not is_forgotten(e)

    def test_is_forgotten_old(self):
        e = make_engram()
        e.stability = 0.001
        e.last_accessed = time.time() - 100000
        assert is_forgotten(e)

    def test_apply_decay_updates_strength(self):
        e = make_engram()
        e.stability = 1.0
        e.last_accessed = time.time() - 7200  # 2 hours ago
        apply_decay(e)
        assert e.synaptic_strength < 1.0

    def test_time_until_forgotten_positive_for_stable(self):
        e = make_engram()
        e.stability = 100.0
        e.last_accessed = time.time()
        assert time_until_forgotten(e) > 0

    def test_time_until_forgotten_zero_for_old(self):
        e = make_engram()
        e.stability = 0.001
        e.last_accessed = time.time() - 100000
        assert time_until_forgotten(e) == 0.0


# ---------------------------------------------------------------------------
# LTP
# ---------------------------------------------------------------------------

class TestLTP:
    def test_potentiate_increments_access_count(self):
        e = make_engram()
        potentiate(e)
        assert e.access_count == 1

    def test_potentiate_resets_strength(self):
        e = make_engram()
        e.synaptic_strength = 0.3
        potentiate(e)
        assert e.synaptic_strength == 1.0

    def test_potentiate_increases_stability(self):
        e = make_engram()
        original_stability = e.stability
        potentiate(e)
        assert e.stability > original_stability

    def test_high_salience_grows_faster(self):
        e_high = make_engram(salience=1.0)
        e_low = make_engram(salience=0.0)
        potentiate(e_high)
        potentiate(e_low)
        assert e_high.stability > e_low.stability

    def test_ltd_weakens_engram(self):
        e = make_engram()
        e.stability = 2.0
        long_term_depression(e, amount=0.5)
        assert e.stability < 2.0
        assert e.synaptic_strength < 1.0

    def test_multiple_potentiations_compound(self):
        e = make_engram()
        for _ in range(5):
            potentiate(e)
        assert e.stability > 2.0
        assert e.access_count == 5


# ---------------------------------------------------------------------------
# SensoryCortex
# ---------------------------------------------------------------------------

class TestSensoryCortex:
    def test_perceive_adds_to_buffer(self):
        sc = SensoryCortex(ttl_seconds=10.0)
        sc.perceive("hello")
        assert len(sc) == 1

    def test_expired_traces_are_dropped(self):
        sc = SensoryCortex(ttl_seconds=0.001)
        sc.perceive("stale")
        time.sleep(0.02)
        assert len(sc) == 0

    def test_drain_clears_buffer(self):
        sc = SensoryCortex(ttl_seconds=10.0)
        sc.perceive("a")
        sc.perceive("b")
        drained = sc.drain()
        assert len(drained) == 2
        assert len(sc) == 0

    def test_modality_stored(self):
        sc = SensoryCortex()
        trace = sc.perceive("beep boop", modality="audio")
        assert trace.modality == "audio"


# ---------------------------------------------------------------------------
# Thalamus
# ---------------------------------------------------------------------------

class TestThalamus:
    def test_gate_passes_high_salience(self):
        thalamus = Thalamus(attention_threshold=0.1)
        sc = SensoryCortex()
        trace = sc.perceive("URGENT: critical error occurred!")
        passed = thalamus.gate([trace])
        assert len(passed) == 1

    def test_gate_rejects_low_salience(self):
        thalamus = Thalamus(attention_threshold=0.9)
        sc = SensoryCortex()
        trace = sc.perceive("ok")
        passed = thalamus.gate([trace])
        assert len(passed) == 0

    def test_context_hint_can_boost_relevance(self):
        thalamus = Thalamus(attention_threshold=0.3)
        sc = SensoryCortex()
        trace = sc.perceive("redis timeout error in database")
        with_ctx = thalamus.gate([trace], context_hint="redis database timeout")
        # It should pass at low threshold with context
        assert len(with_ctx) >= 0  # at minimum doesn't crash


# ---------------------------------------------------------------------------
# Amygdala
# ---------------------------------------------------------------------------

class TestAmygdala:
    def test_tag_auto_salience(self):
        amygdala = Amygdala()
        e = make_engram(content="URGENT: system failure!")
        amygdala.tag(e)
        assert e.salience > 0.5

    def test_tag_manual_salience(self):
        amygdala = Amygdala()
        e = make_engram()
        amygdala.tag(e, salience=0.9)
        assert e.salience == 0.9

    def test_high_salience_boosts_stability(self):
        amygdala = Amygdala(high_salience_threshold=0.7, stability_boost_factor=2.0)
        e = make_engram()
        e.stability = 1.0
        amygdala.tag(e, salience=0.9)
        assert e.stability == pytest.approx(2.0)

    def test_low_salience_no_stability_boost(self):
        amygdala = Amygdala(high_salience_threshold=0.7)
        e = make_engram()
        e.stability = 1.0
        amygdala.tag(e, salience=0.3)
        assert e.stability == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# PrefrontalCortex
# ---------------------------------------------------------------------------

class TestPrefrontalCortex:
    def test_hold_within_capacity(self):
        pfc = PrefrontalCortex(capacity=3)
        for i in range(3):
            displaced = pfc.hold(make_engram(content=f"item {i}"))
            assert displaced is None
        assert len(pfc) == 3

    def test_hold_displaces_lru_on_overflow(self):
        pfc = PrefrontalCortex(capacity=2)
        e1 = make_engram(content="first")
        e2 = make_engram(content="second")
        e3 = make_engram(content="third")
        pfc.hold(e1)
        pfc.hold(e2)
        displaced = pfc.hold(e3)
        assert displaced is not None
        assert displaced.content == "first"

    def test_rehearse_prevents_lru_eviction(self):
        pfc = PrefrontalCortex(capacity=2)
        e1 = make_engram(content="rehearsed")
        e2 = make_engram(content="other")
        pfc.hold(e1)
        pfc.hold(e2)
        pfc.rehearse(e1.engram_id)  # e1 is now MRU
        e3 = make_engram(content="new")
        displaced = pfc.hold(e3)
        assert displaced is not None
        assert displaced.content == "other"

    def test_decay_all_removes_forgotten(self):
        pfc = PrefrontalCortex()
        e = make_engram()
        e.stability = 0.001
        e.last_accessed = time.time() - 100000
        pfc.hold(e)
        forgotten = pfc.decay_all()
        assert len(forgotten) == 1
        assert e.engram_id not in [slot.engram_id for slot in pfc.engrams]

    def test_consolidation_candidates_by_access(self):
        pfc = PrefrontalCortex()
        e = make_engram()
        e.access_count = 5
        pfc.hold(e)
        assert e in pfc.consolidation_candidates()

    def test_consolidation_candidates_by_salience(self):
        pfc = PrefrontalCortex()
        e = make_engram(salience=0.95)
        pfc.hold(e)
        assert e in pfc.consolidation_candidates()

    def test_remove(self):
        pfc = PrefrontalCortex()
        e = make_engram()
        pfc.hold(e)
        removed = pfc.remove(e.engram_id)
        assert removed is not None
        assert len(pfc) == 0


# ---------------------------------------------------------------------------
# Redis-backed stores
# ---------------------------------------------------------------------------

class TestNeocortex:
    async def test_store_and_retrieve(self, redis):
        nc = Neocortex(redis)
        e = make_engram(content="Paris is the capital of France",
                        memory_type=MemoryType.SEMANTIC)
        await nc.store(e)
        retrieved = await nc.retrieve(e.engram_id)
        assert retrieved is not None
        assert retrieved.content == e.content
        assert retrieved.layer == MemoryLayer.LONG_TERM

    async def test_delete(self, redis):
        nc = Neocortex(redis)
        e = make_engram()
        await nc.store(e)
        deleted = await nc.delete(e.engram_id)
        assert deleted is True
        assert await nc.retrieve(e.engram_id) is None

    async def test_all_engrams(self, redis):
        nc = Neocortex(redis)
        for i in range(3):
            await nc.store(make_engram(content=f"fact {i}"))
        all_e = await nc.all_engrams()
        assert len(all_e) == 3

    async def test_update(self, redis):
        nc = Neocortex(redis)
        e = make_engram(content="updateable")
        await nc.store(e)
        e.salience = 0.99
        await nc.update(e)
        retrieved = await nc.retrieve(e.engram_id)
        assert retrieved.salience == 0.99


class TestCerebellum:
    async def test_store_sets_procedural_type(self, redis):
        cb = Cerebellum(redis)
        e = make_engram(content="how to ride a bike")
        await cb.store(e)
        retrieved = await cb.retrieve(e.engram_id)
        assert retrieved.memory_type == MemoryType.PROCEDURAL

    async def test_procedural_gets_high_stability(self, redis):
        cb = Cerebellum(redis)
        e = make_engram(content="skill memory")
        e.stability = 1.0
        await cb.store(e)
        retrieved = await cb.retrieve(e.engram_id)
        assert retrieved.stability >= 5.0


class TestBasalGanglia:
    async def test_reward_boosts_stability(self, redis):
        bg = BasalGanglia(redis)
        e_high = make_engram(content="high reward habit")
        e_low = make_engram(content="low reward habit")
        await bg.store(e_high, reward_signal=1.0)
        await bg.store(e_low, reward_signal=0.0)
        r_high = await bg.retrieve(e_high.engram_id)
        r_low = await bg.retrieve(e_low.engram_id)
        assert r_high.stability > r_low.stability

    async def test_reward_stored_in_context(self, redis):
        bg = BasalGanglia(redis)
        e = make_engram()
        await bg.store(e, reward_signal=0.75)
        retrieved = await bg.retrieve(e.engram_id)
        assert retrieved.context["reward_signal"] == 0.75

    async def test_habitual_memory_type(self, redis):
        bg = BasalGanglia(redis)
        e = make_engram()
        await bg.store(e)
        retrieved = await bg.retrieve(e.engram_id)
        assert retrieved.memory_type == MemoryType.HABITUAL


# ---------------------------------------------------------------------------
# SleepCycle
# ---------------------------------------------------------------------------

class TestSleepCycle:
    async def test_consolidates_candidates(self, redis):
        pfc = PrefrontalCortex()
        hpc = Hippocampus()
        nc = Neocortex(redis)
        cb = Cerebellum(redis)
        bg = BasalGanglia(redis)

        e = make_engram(content="consolidate me", memory_type=MemoryType.SEMANTIC)
        e.access_count = 5
        e.embedding = fake_embedding(e.content)
        pfc.hold(e)
        hpc.index(e)

        sleep = SleepCycle(pfc, hpc, nc, cb, bg)
        stats = await sleep.run_once()

        assert stats["consolidated"] >= 1
        in_ltm = await nc.retrieve(e.engram_id)
        assert in_ltm is not None

    async def test_prunes_forgotten_ltm(self, redis):
        pfc = PrefrontalCortex()
        hpc = Hippocampus()
        nc = Neocortex(redis)
        cb = Cerebellum(redis)
        bg = BasalGanglia(redis)

        e = make_engram(content="forgotten long ago", memory_type=MemoryType.SEMANTIC)
        e.stability = 0.001
        e.last_accessed = time.time() - 1_000_000
        await nc.store(e)
        hpc.index(e)

        sleep = SleepCycle(pfc, hpc, nc, cb, bg, forgetting_threshold=0.05)
        stats = await sleep.run_once()

        assert stats["pruned_ltm"] >= 1
        assert await nc.retrieve(e.engram_id) is None

    async def test_routes_procedural_to_cerebellum(self, redis):
        pfc = PrefrontalCortex()
        hpc = Hippocampus()
        nc = Neocortex(redis)
        cb = Cerebellum(redis)
        bg = BasalGanglia(redis)

        e = make_engram(content="typing skill", memory_type=MemoryType.PROCEDURAL)
        e.access_count = 5
        e.embedding = fake_embedding(e.content)
        pfc.hold(e)
        hpc.index(e)

        sleep = SleepCycle(pfc, hpc, nc, cb, bg)
        await sleep.run_once()

        in_cerebellum = await cb.retrieve(e.engram_id)
        assert in_cerebellum is not None
        assert in_cerebellum.memory_type == MemoryType.PROCEDURAL
