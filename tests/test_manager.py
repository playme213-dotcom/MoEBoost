"""Tests for Workshop Manager directing experts to workstations."""
import unittest
import numpy as np
import time

from moe_boost.queue_engine import (
    ExpertExecutionQueue,
    WorkshopManager,
    ManagerDirective,
    WorkstationType,
    BusynessLevel,
)


class MockRamCache:
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


class TestWorkshopManagerDirective(unittest.TestCase):
    def setUp(self):
        self.mock_cache = MockRamCache()
        self.queue = ExpertExecutionQueue(
            ram_cache=self.mock_cache,
            max_vram_slots=2,
            max_vram_line_length=2,
            hot_threshold_calls=3,
            warm_threshold_calls=2,
            hand_labor_workers=4,
        )

    def tearDown(self):
        self.queue.shutdown()

    def test_manager_directive_attached_to_job(self):
        tokens = np.random.randn(1, 16).astype(np.float32)
        job = self.queue.dispatch("exp_mgr_1", tokens, sync=True)
        # Manager directive must be present
        self.assertIsNotNone(job.manager_directive)
        self.assertIn("Manager:", job.manager_directive.reason)
        self.assertEqual(job.executed_on, job.manager_directive.assigned_workstation)

    def test_manager_tells_hot_expert_to_use_tools(self):
        tokens = np.random.randn(1, 16).astype(np.float32)
        # Warm up expert to HOT in catalog
        for _ in range(3):
            self.queue.dispatch("vip_expert", tokens, sync=True)

        job = self.queue.dispatch("vip_expert", tokens, sync=True)
        self.assertEqual(job.busyness, BusynessLevel.HOT)
        self.assertEqual(job.manager_directive.assigned_workstation, WorkstationType.TOOL_WORKSTATION_VRAM)

    def test_manager_diverts_overflow_to_hand_workstation(self):
        tokens = np.random.randn(1, 16).astype(np.float32)
        # Dispatch multiple simultaneous jobs to trigger manager diversion
        jobs = []
        for i in range(8):
            j = self.queue.dispatch(f"divert_{i}", tokens, sync=True)
            jobs.append(j)

        # Some jobs were explicitly instructed to use Hand-Labor Workstation
        hand_jobs = [j for j in jobs if j.executed_on == WorkstationType.HAND_WORKSTATION_RAM]
        self.assertGreater(len(hand_jobs), 0)
        for hj in hand_jobs:
            self.assertIn("Tool Workstation is full/busy", hj.manager_directive.reason)

    def test_ssd_is_never_assigned_as_workstation(self):
        tokens = np.random.randn(1, 16).astype(np.float32)
        job = self.queue.dispatch("exp_storage_3", tokens, sync=True)
        # Must only be one of the two compute workstations
        self.assertIn(job.executed_on, [
            WorkstationType.TOOL_WORKSTATION_VRAM,
            WorkstationType.HAND_WORKSTATION_RAM,
        ])
        self.assertTrue(job.staged_from_ssd_to_ram)


if __name__ == "__main__":
    unittest.main()
