from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import timedelta
from threading import RLock

from railforge.signaling.block_occupancy import Occupancy, conflicts as occupancy_conflicts


@dataclass(frozen=True)
class SwitchPoint:
    """Authoritative state of one switch (turnout/point)."""
    id: str
    position: str
    locked_by: str | None = None


@dataclass(frozen=True)
class RouteDemand:
    """A route-setting request: required switch positions plus an optional block claim."""
    operation_id: str
    owner: str
    requirements: dict[str, str]
    occupancy: Occupancy | None = None
    clearance: timedelta = timedelta(0)


@dataclass(frozen=True)
class RouteGrant:
    operation_id: str
    owner: str
    route_id: str
    locked_switches: tuple[str, ...]
    switches: dict[str, SwitchPoint]
    occupancies: list[Occupancy]


class RouteRejected(ValueError):
    """Raised when a route cannot be set. State is never mutated on rejection."""


def decide(demand: RouteDemand, route_id: str, switches: dict[str, SwitchPoint],
           occupancies: list[Occupancy]) -> RouteGrant:
    """Pure, all-or-nothing route-setting decision.

    Ownership of every required switch is checked against the *authoritative*
    current state, positions must match the route requirements, and the block
    claim must not conflict with other trains. Every validation happens before
    any new state is produced, so a rejection can never leave partial locks.
    """
    if not demand.requirements and demand.occupancy is None:
        raise RouteRejected('empty route demand')

    # Phase 1: validate everything against the supplied authoritative snapshot.
    for sid, required in demand.requirements.items():
        point = switches.get(sid)
        if point is None:
            raise RouteRejected(f'unknown switch: {sid}')
        if point.locked_by is not None and point.locked_by != demand.owner:
            raise RouteRejected(f'switch {sid} locked by {point.locked_by}')
        if point.position != required:
            raise RouteRejected(f'switch {sid} position {point.position} != {required}')

    next_occupancies = list(occupancies)
    if demand.occupancy is not None:
        hit = occupancy_conflicts(demand.occupancy, occupancies, demand.clearance)
        if hit:
            raise RouteRejected('block conflict: ' + ','.join(hit))
        # Domain-level idempotency: an exact re-request of the same occupancy
        # (e.g. a retried call) never duplicates the block reservation.
        if not any(row == demand.occupancy for row in occupancies):
            next_occupancies.append(demand.occupancy)
            next_occupancies.sort(key=lambda r: (r.block, r.enter, r.train))

    # Phase 2: only after every check passed, produce the new switch state.
    locked = tuple(sorted(demand.requirements))
    next_switches = dict(switches)
    for sid in locked:
        point = next_switches[sid]
        if point.locked_by is None:
            next_switches[sid] = replace(point, locked_by=demand.owner)

    return RouteGrant(demand.operation_id, demand.owner, route_id, locked,
                      next_switches, next_occupancies)


class Interlocking:
    """Authoritative, thread-safe aggregate owning switch locks and block claims.

    Retries are always evaluated against committed state: a caller cannot smuggle
    a stale switch snapshot past ownership, and a rejected demand changes nothing.
    """

    def __init__(self, switches: dict[str, SwitchPoint] | None = None,
                 occupancies: list[Occupancy] | None = None):
        self._switches = dict(switches or {})
        self._occupancies = list(occupancies or [])
        self._grants: dict[str, RouteGrant] = {}
        self._lock = RLock()

    @property
    def switches(self) -> dict[str, SwitchPoint]:
        with self._lock:
            return dict(self._switches)

    @property
    def occupancies(self) -> list[Occupancy]:
        with self._lock:
            return list(self._occupancies)

    def grant(self, operation_id: str) -> RouteGrant | None:
        with self._lock:
            return self._grants.get(operation_id)

    def request(self, demand: RouteDemand, route_id: str) -> RouteGrant:
        with self._lock:
            prior = self._grants.get(demand.operation_id)
            if prior is not None:
                # Idempotent replay: same operation returns the committed grant
                # unchanged. It must never re-evaluate against a later state
                # (which is how a retry could otherwise steal a lock).
                if prior.owner != demand.owner or prior.route_id != route_id:
                    raise RouteRejected('operation id reused for a different route request')
                return prior
            grant = decide(demand, route_id, self._switches, self._occupancies)
            self._switches = grant.switches
            self._occupancies = grant.occupancies
            self._grants[demand.operation_id] = grant
            return grant
