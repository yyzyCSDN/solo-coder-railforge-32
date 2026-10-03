import unittest
from datetime import datetime, timedelta, timezone

from railforge.signaling.block_occupancy import Occupancy
from railforge.signaling.interlocking import (
    Interlocking, RouteDemand, RouteRejected, SwitchPoint, decide,
)
from railforge.workflows.control import RailForgeWorkflowControl
from railforge.workflows.features import RailForgeFeatureService
from railforge.yard.switching import Route as YardRoute, Switch, can_set

U = timezone.utc
T0 = datetime(2026, 1, 1, 8, tzinfo=U)


class OwnershipDomainTests(unittest.TestCase):
    def setUp(self):
        self.box = Interlocking({
            "SW1": SwitchPoint("SW1", "normal"),
            "SW2": SwitchPoint("SW2", "reverse"),
        })

    def test_other_owner_cannot_set_route_over_locked_switch(self):
        self.box.request(RouteDemand("op-1", "dispatcher-A", {"SW1": "normal"}), "R1")
        with self.assertRaisesRegex(RouteRejected, "locked by dispatcher-A"):
            self.box.request(RouteDemand("op-2", "dispatcher-B", {"SW1": "normal"}), "R2")
        # lock ownership is unchanged after rejection
        self.assertEqual(self.box.switches["SW1"].locked_by, "dispatcher-A")
        self.assertIsNone(self.box.switches["SW2"].locked_by)

    def test_same_owner_can_extend_over_own_lock(self):
        self.box.request(RouteDemand("op-1", "dispatcher-A", {"SW1": "normal"}), "R1")
        grant = self.box.request(
            RouteDemand("op-2", "dispatcher-A", {"SW1": "normal", "SW2": "reverse"}), "R2")
        self.assertEqual(grant.locked_switches, ("SW1", "SW2"))

    def test_wrong_position_rejected_without_partial_locks(self):
        with self.assertRaisesRegex(RouteRejected, "position"):
            self.box.request(
                RouteDemand("op-x", "dispatcher-B", {"SW1": "normal", "SW2": "normal"}), "Rx")
        # neither switch may be locked after the rejection
        self.assertIsNone(self.box.switches["SW1"].locked_by)
        self.assertIsNone(self.box.switches["SW2"].locked_by)

    def test_unknown_switch_leaves_no_trace(self):
        with self.assertRaisesRegex(RouteRejected, "unknown switch"):
            self.box.request(RouteDemand("op-x", "dispatcher-B", {"SW9": "normal"}), "Rx")
        self.assertEqual(self.box.grant("op-x"), None)

    def test_block_conflict_rejects_before_locking(self):
        other = Occupancy("T-other", "B1", T0, T0 + timedelta(hours=1), 1)
        box = Interlocking({"SW1": SwitchPoint("SW1", "normal")}, [other])
        mine = Occupancy("T-mine", "B1", T0 + timedelta(minutes=30),
                         T0 + timedelta(hours=2), 1)
        with self.assertRaisesRegex(RouteRejected, "block conflict"):
            box.request(RouteDemand("op-x", "dispatcher-A", {"SW1": "normal"}, mine,
                                    timedelta(minutes=5)), "Rx")
        self.assertIsNone(box.switches["SW1"].locked_by)
        self.assertEqual(box.occupancies, [other])

    def test_route_and_block_commit_together(self):
        mine = Occupancy("T1", "B1", T0, T0 + timedelta(hours=1), 1)
        grant = self.box.request(
            RouteDemand("op-1", "dispatcher-A", {"SW1": "normal"}, mine), "R1")
        self.assertEqual(grant.locked_switches, ("SW1",))
        self.assertEqual([r.train for r in self.box.occupancies], ["T1"])

    def test_idempotent_retry_returns_same_grant_and_no_duplicate_occupancy(self):
        mine = Occupancy("T1", "B1", T0, T0 + timedelta(hours=1), 1)
        demand = RouteDemand("op-1", "dispatcher-A", {"SW1": "normal"}, mine)
        first = self.box.request(demand, "R1")
        second = self.box.request(demand, "R1")
        self.assertIs(first, second)
        self.assertEqual(len(self.box.occupancies), 1)

    def test_retry_cannot_steal_lock_by_switching_owner(self):
        # A succeeds; B retries with the SAME operation id but its own name.
        self.box.request(RouteDemand("op-1", "dispatcher-A", {"SW1": "normal"}), "R1")
        with self.assertRaises(RouteRejected):
            self.box.request(RouteDemand("op-1", "dispatcher-B", {"SW1": "normal"}), "R1")
        self.assertEqual(self.box.switches["SW1"].locked_by, "dispatcher-A")

    def test_retry_after_intervening_lock_still_returns_original_grant(self):
        # Even if another dispatcher meanwhile locks a different switch,
        # replaying op-1 must not re-evaluate against the new state.
        first = self.box.request(RouteDemand("op-1", "dispatcher-A", {"SW1": "normal"}), "R1")
        self.box.request(RouteDemand("op-9", "dispatcher-B", {"SW2": "reverse"}), "R9")
        again = self.box.request(RouteDemand("op-1", "dispatcher-A", {"SW1": "normal"}), "R1")
        self.assertIs(again, first)


class YardSwitchingGuardsRestoredTests(unittest.TestCase):
    def test_can_set_respects_ownership_again(self):
        switches = {"SW1": Switch("SW1", "normal", "dispatcher-A")}
        route = YardRoute("R1", {"SW1": "normal"})
        self.assertTrue(can_set(route, switches, "dispatcher-A"))
        self.assertFalse(can_set(route, switches, "dispatcher-B"))


class ControlPlaneIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.control = RailForgeWorkflowControl()
        self.control.provision_interlocking("yard-1", {
            "SW1": SwitchPoint("SW1", "normal"),
            "SW2": SwitchPoint("SW2", "reverse"),
        })
        self.features = RailForgeFeatureService(self.control)

    def test_full_workflow_owner_then_other_owner_rejected(self):
        receipt = self.features.request_switch_route(
            "wf-1", "yard-1", "R1", "dispatcher-A", {"SW1": "normal"}, T0)
        self.assertEqual(receipt.event.topic, "signaling.route.set")
        self.assertEqual(receipt.event.payload["status"], "committed")
        self.assertEqual(receipt.event.payload["control_path"], "atomic")
        self.assertEqual(receipt.owner, "dispatcher-A")

        with self.assertRaisesRegex(RouteRejected, "locked by dispatcher-A"):
            self.features.request_switch_route(
                "wf-2", "yard-1", "R2", "dispatcher-B", {"SW1": "normal"}, T0)

    def test_rejected_request_publishes_no_event_and_no_store_version(self):
        self.features.request_switch_route(
            "wf-1", "yard-1", "R1", "dispatcher-A", {"SW1": "normal"}, T0)
        events_before = self.control.stream.head
        with self.assertRaises(RouteRejected):
            self.features.request_switch_route(
                "wf-2", "yard-1", "R2", "dispatcher-B", {"SW1": "normal"}, T0)
        self.assertEqual(self.control.stream.head, events_before)
        self.assertNotIn("wf-2", self.control.receipts)
        self.assertTrue(self.control.stream.verify_chain())
        self.assertTrue(self.control.trail.verify())

    def test_idempotent_retry_same_operation_returns_same_receipt(self):
        r1 = self.features.request_switch_route(
            "wf-1", "yard-1", "R1", "dispatcher-A", {"SW1": "normal"}, T0)
        r2 = self.features.request_switch_route(
            "wf-1", "yard-1", "R1", "dispatcher-A", {"SW1": "normal"}, T0)
        self.assertIs(r1, r2)
        self.assertEqual(self.control.stream.head, 1)

    def test_same_operation_id_from_other_owner_is_permission_error(self):
        self.features.request_switch_route(
            "wf-1", "yard-1", "R1", "dispatcher-A", {"SW1": "normal"}, T0)
        with self.assertRaises(PermissionError):
            self.features.request_switch_route(
                "wf-1", "yard-1", "R1", "dispatcher-B", {"SW1": "normal"}, T0)
        # the original grant survives and no second event exists
        self.assertEqual(self.control.stream.head, 1)
        self.assertEqual(self.control.interlocking("yard-1").switches["SW1"].locked_by,
                         "dispatcher-A")

    def test_block_conflict_through_control_plane_leaves_nothing(self):
        control = RailForgeWorkflowControl()
        control.provision_interlocking("yard-2", {"SW1": SwitchPoint("SW1", "normal")}, [
            Occupancy("T-other", "B1", T0, T0 + timedelta(hours=1), 1),
        ])
        mine = Occupancy("T-mine", "B1", T0 + timedelta(minutes=30),
                         T0 + timedelta(hours=2), 1)
        with self.assertRaises(RouteRejected):
            control.set_switch_route("wf-x", "yard-2", "Rx", "dispatcher-A", T0,
                                     {"SW1": "normal"}, mine, timedelta(minutes=5))
        self.assertEqual(control.stream.head, 0)
        self.assertIsNone(control.interlocking("yard-2").switches["SW1"].locked_by)

    def test_stream_rejects_event_id_reuse_across_topics(self):
        self.features.request_switch_route(
            "wf-1", "yard-1", "R1", "dispatcher-A", {"SW1": "normal"}, T0)
        with self.assertRaises(ValueError):
            self.control.stream.publish(
                "wf-1", T0, "something.else", "yard-1", {"status": "committed"})


if __name__ == "__main__":
    unittest.main()
