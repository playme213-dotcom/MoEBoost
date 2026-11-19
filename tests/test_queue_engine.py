"""Tests for People Busyness Catalog & Heterogeneous Workstations, verifying SSD is storage only."""
import unittest
import numpy as np
import time

from moe_boost.queue_engine import (
    ExpertExecutionQueue,
    WorkstationType,
    BusynessLevel,
)


class MockRamCache:
    """Mock storage/cache demonstrating SSD -> RAM staging."""
    def __init__(self):
        self._ram = {}
        self._ssd = {f"exp_storage_{i}": np.eye(16, dtype=np.float32) for i in range(10)}

    def contains(self, key):
        return key in self._ram

    def fetch_from_storage(self, key):
        if key in self._ssd:
            self._ram[key] = self._ssd[key]

    def get_copy(self, key):
        return self._ram.get(key)

    def observe_hit(self, key):
        pass


class TestWorkshopWorkstationQueue(unittest.TestCase):
    def setUp(self):
        self.mock_cache = MockRamCache()
        self.queue = ExpertExecutionQueue(
            ram_cache=self.mock_cache,
            max_vram_slots=2,
            max_vram_line_length=3,
            hot_threshold_calls=4,
            warm_threshold_calls=2,
            auto_promote_to_ram=True,
            hand_labor_workers=4,
        )

    def tearDown(self):
        self.queue.shutdown()

    def test_ssd_is_storage_only_and_never_a_workstation(self):
        tokens = np.random.randn(1, 16).astype(np.float32)
        # Dispatch expert that starts on cold SSD storage
        job = self.queue.dispatch("exp_storage_1", tokens, sync=True)
        # Execution must only happen on Tool Workstation (VRAM) or Hand Workstation (RAM)
        self.assertIn(job.executed_on, [
            WorkstationType.TOOL_WORKSTATION_VRAM,
            WorkstationType.HAND_WORKSTATION_RAM,
        ])
        # Weights were pulled from SSD to RAM before work began
        self.assertTrue(job.staged_from_ssd_to_ram)
        self.assertGreater(self.queue.snapshot()["ssd_storage"]["fetches_from_ssd_to_ram"], 0)

    def test_stress_loop_workstations(self):
        """Stress loop with 100 job dispatches across varying expert keys."""
        tokens = np.random.randn(2, 16).astype(np.float32)
        for i in range(100):
            key = f"expert_{i % 8}"
            job = self.queue.dispatch(key, tokens, sync=True)
            self.assertIsNotNone(job.result)
            self.assertIn(job.executed_on, [
                WorkstationType.TOOL_WORKSTATION_VRAM,
                WorkstationType.HAND_WORKSTATION_RAM,
            ])

        snap = self.queue.snapshot()
        # Verify both workstations handled significant workloads
        self.assertGreater(snap["tool_workstation_vram"]["jobs_completed"], 0)
        self.assertGreater(snap["hand_workstation_ram"]["jobs_completed"], 0)
        self.assertGreater(snap["busyness_catalog"]["hot_people_count"], 0)

    def test_people_busyness_catalog(self):
        tokens = np.random.randn(1, 16).astype(np.float32)

        # 1 call -> Cold
        job1 = self.queue.dispatch("person_A", tokens, sync=True)
        self.assertEqual(job1.busyness, BusynessLevel.COLD)

        # 2 calls -> Warm
        job2 = self.queue.dispatch("person_A", tokens, sync=True)
        self.assertEqual(job2.busyness, BusynessLevel.WARM)

        # 4 calls -> Hot
        self.queue.dispatch("person_A", tokens, sync=True)
        job4 = self.queue.dispatch("person_A", tokens, sync=True)
        self.assertEqual(job4.busyness, BusynessLevel.HOT)

        snap = self.queue.snapshot()
        self.assertIn("person_A", snap["busyness_catalog"]["hot_people"])

    def test_batch_dispatch_parallelism(self):
        tokens = np.random.randn(1, 16).astype(np.float32)
        batch = [(f"batch_person_{i}", tokens) for i in range(8)]
        results = self.queue.dispatch_batch(batch, timeout=5.0)

        self.assertEqual(len(results), 8)
        for job in results:
            self.assertIsNotNone(job.result)
            self.assertIn(job.executed_on, [
                WorkstationType.TOOL_WORKSTATION_VRAM,
                WorkstationType.HAND_WORKSTATION_RAM,
            ])

    def test_transient_mode_stepping_out_of_slot(self):
        transient_q = ExpertExecutionQueue(
            max_vram_slots=2,
            max_vram_line_length=3,
            transient_mode=True,
        )
        try:
            tokens = np.random.randn(1, 16).astype(np.float32)
            transient_q.dispatch("transient_person", tokens, sync=True)
            snap = transient_q.snapshot()
            self.assertEqual(snap["tool_workstation_vram"]["occupied_tool_slots"], 0)
            self.assertGreater(snap["tool_workstation_vram"]["displaced_people"], 0)
        finally:
            transient_q.shutdown()


if __name__ == "__main__":
    unittest.main()
