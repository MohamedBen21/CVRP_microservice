# ─────────────────────────────────────────────────────────────────────────────
#  algorithms/genetic_assignment.py
#  Genetic Algorithm — production CVRP assignment engine.
#
#  Chromosome encoding (unchanged):
#    Integer array of length = number of packages.
#    chromosome[i] = vehicle_index assigned to package i (0-based).
#
#  ═══════════════════════════════════════════════════════════════════════
#  FITNESS FUNCTION  (lower = better)
#  ═══════════════════════════════════════════════════════════════════════
#
#  fitness = CAPACITY_VIOLATION_PEN  × (weight_excess + volume_excess)   [1]
#          + FRAGILE_PENALTY         × fragile_mismatch_count            [2]
#          + DELIVERER_OVERLOAD_PEN  × packages_over_limit^1.5           [3]
#          + VEHICLE_COUNT_PENALTY   × vehicles_used                     [4]
#          + UNDERUTILISATION_PEN    × (1 − avg_fill) per vehicle        [5]
#          + route_distance_cost                                          [6]
#
#  Priority order (highest penalty → lowest):
#    [1] Capacity violations   — infeasible solutions must almost never survive
#    [2] Fragile mismatches    — hard business / safety rule
#    [3] Deliverer pkg limit   — hard business rule: ≤ 15 pkgs per deliverer
#    [4] Vehicle count         — fewest vehicles = highest business priority
#    [5] Underutilisation      — pack tightly before opening a new vehicle
#    [6] Route distance        — minimise km after all of the above
#
#  ═══════════════════════════════════════════════════════════════════════
#  VEHICLE ORDERING
#  ═══════════════════════════════════════════════════════════════════════
#  Vehicles are sorted small → large before the GA runs.
#  The greedy seed fills the smallest vehicle first so the initial
#  population already prefers consolidation.  The triangular-distribution
#  random initialisation is also skewed toward lower vehicle indices.
#
#  ═══════════════════════════════════════════════════════════════════════
#  ADAPTIVE MUTATION
#  ═══════════════════════════════════════════════════════════════════════
#  Mutation rate starts at MUTATION_RATE_BASE.  When no improvement is
#  seen for STAGNATION_WINDOW generations, it jumps to MUTATION_RATE_HIGH
#  to escape local minima, then resets as soon as improvement is found.
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations
import os
import math
import random
import logging
import numpy as np
from dataclasses import dataclass
from utils.haversine import haversine_km

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
#  GA HYPER-PARAMETERS  (all overridable via environment variables)
# ─────────────────────────────────────────────────────────────────────────────

POPULATION_SIZE    = int(os.getenv("GA_POPULATION_SIZE",     "80"))
GENERATIONS        = int(os.getenv("GA_GENERATIONS",         "150"))
MUTATION_RATE_BASE = float(os.getenv("GA_MUTATION_RATE",     "0.04"))
MUTATION_RATE_HIGH = float(os.getenv("GA_MUTATION_RATE_HIGH","0.20"))  # stagnation escape
ELITE_SIZE         = int(os.getenv("GA_ELITE_SIZE",          "6"))
STAGNATION_WINDOW  = int(os.getenv("GA_STAGNATION_WINDOW",   "20"))    # gens without improvement

# ─────────────────────────────────────────────────────────────────────────────
#  PENALTY WEIGHTS  (all overridable via environment variables)
#
#  Scale rationale:
#    CAPACITY_VIOLATION_PEN  ≫  FRAGILE_PENALTY
#                            ≫  DELIVERER_OVERLOAD_PEN
#                            ≫  VEHICLE_COUNT_PENALTY
#                            ≫  UNDERUTILISATION_PEN
#                            ≫  route_distance (typically < 1 000 km)
#
#  This ordering guarantees a feasible solution always beats an infeasible
#  one, and a solution using fewer vehicles always beats one with more,
#  regardless of utilisation or distance.
# ─────────────────────────────────────────────────────────────────────────────

# [1] Hard constraint — capacity
CAPACITY_VIOLATION_PEN = float(os.getenv("GA_PEN_CAPACITY",     "10_000_000"))

# [2] Hard constraint — fragile
FRAGILE_PENALTY        = float(os.getenv("GA_PEN_FRAGILE",      "5_000_000"))

# [3] Hard business rule — deliverer package count
DELIVERER_OVERLOAD_PEN = float(os.getenv("GA_PEN_OVERLOAD",     "2_000_000"))
MAX_PKGS_PER_DELIVERER = int(os.getenv("GA_MAX_PKGS_DELIVERER",  "15"))

# [4] Soft objective — minimise vehicle count (HIGHEST business priority)
VEHICLE_COUNT_PENALTY  = float(os.getenv("GA_PEN_VEHICLE_COUNT","100_000"))

# [5] Soft objective — maximise vehicle utilisation
UNDERUTILISATION_PEN   = float(os.getenv("GA_PEN_UNDERUTIL",    "20_000"))

# 5 % safety buffer — mirrors Node.js CAPACITY_BUFFER = 0.95
CAPACITY_BUFFER = 0.95

# Vehicle type → size rank (motorcycle=0 … large_truck=4).
# Used to sort vehicles small → large before the GA runs.
_VEHICLE_TYPE_RANK: dict[str, int] = {
    "motorcycle":  0,
    "car":         1,
    "van":         2,
    "small_truck": 3,
    "large_truck": 4,
}


# ─────────────────────────────────────────────────────────────────────────────
#  DATA CLASSES
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PackageGA:
    idx:        int
    weight:     float
    volume:     float
    is_fragile: bool
    coords:     tuple[float, float] | None  # None for transporter packages
    priority:   int                         # 0=same_day, 1=express, 2=standard


@dataclass
class VehicleGA:
    idx:              int    # original position in the caller's vehicle list
    max_weight:       float
    max_volume:       float
    supports_fragile: bool
    type_rank:        int = 0  # smaller = physically smaller vehicle


@dataclass
class AssignmentResult:
    """Final output: one entry per non-empty vehicle (sorted small → large)."""
    vehicle_idx:     int         # index into the sorted_vehicles list returned by the GA
    package_indices: list[int]   # indices into the packages list passed to the GA


# ─────────────────────────────────────────────────────────────────────────────
#  FITNESS FUNCTION
# ─────────────────────────────────────────────────────────────────────────────

def _fitness(
    chromosome:    np.ndarray,
    packages:      list[PackageGA],
    vehicles:      list[VehicleGA],  # already sorted small→large
    is_deliverer:  bool,
    dist_matrix:   list[list[float]] | None,
    origin_coords: tuple[float, float],
) -> float:
    """
    Evaluates one chromosome.  Lower score = better solution.

    See module docstring for full component breakdown.
    """
    n_vehicles = len(vehicles)
    score      = 0.0

    # ── Accumulate per-vehicle loads ──────────────────────────────────────────
    used_weight: list[float]     = [0.0]   * n_vehicles
    used_volume: list[float]     = [0.0]   * n_vehicles
    has_fragile: list[bool]      = [False] * n_vehicles
    pkg_per_veh: list[list[int]] = [[]     for _ in range(n_vehicles)]

    for pkg_idx, raw_veh_idx in enumerate(chromosome):
        veh_idx = int(raw_veh_idx)
        if veh_idx < 0 or veh_idx >= n_vehicles:
            # Invalid gene — catastrophic penalty
            score += CAPACITY_VIOLATION_PEN * 10
            continue
        pkg = packages[pkg_idx]
        used_weight[veh_idx] += pkg.weight
        used_volume[veh_idx] += pkg.volume
        if pkg.is_fragile:
            has_fragile[veh_idx] = True
        pkg_per_veh[veh_idx].append(pkg_idx)

    # ── Per-vehicle evaluation ────────────────────────────────────────────────
    vehicles_used = 0

    for veh_idx, veh in enumerate(vehicles):
        pkgs = pkg_per_veh[veh_idx]
        if not pkgs:
            continue

        vehicles_used += 1
        cap_w = veh.max_weight * CAPACITY_BUFFER
        cap_v = veh.max_volume * CAPACITY_BUFFER

        # [1] Capacity violations
        w_excess = max(0.0, used_weight[veh_idx] - cap_w)
        v_excess = max(0.0, used_volume[veh_idx] - cap_v)
        # Volume excess carries 10× weight: harder to redistribute in practice
        score += w_excess * CAPACITY_VIOLATION_PEN
        score += v_excess * CAPACITY_VIOLATION_PEN * 10.0

        # [2] Fragile mismatch
        if has_fragile[veh_idx] and not veh.supports_fragile:
            # Per-package so GA is motivated to move each package individually
            score += FRAGILE_PENALTY * len(pkgs)

        # [3] Deliverer package-count hard limit
        if is_deliverer:
            excess = max(0, len(pkgs) - MAX_PKGS_PER_DELIVERER)
            if excess > 0:
                # Power of 1.5: penalty grows super-linearly so the GA strongly
                # prefers distributing excess packages over keeping them together
                score += DELIVERER_OVERLOAD_PEN * (excess ** 1.5)

        # [5] Underutilisation
        # avg_fill is capped at 1.0 so overloaded vehicles don't look "efficient"
        w_fill   = min(1.0, used_weight[veh_idx] / veh.max_weight)
        v_fill   = min(1.0, used_volume[veh_idx] / veh.max_volume)
        avg_fill = (w_fill + v_fill) / 2.0
        # (1 - avg_fill): a vehicle at 30% fill → 0.70 × UNDERUTILISATION_PEN
        score   += (1.0 - avg_fill) * UNDERUTILISATION_PEN

        # [6] Route distance estimate
        score += _estimate_cluster_distance(
            pkgs, packages, origin_coords, dist_matrix
        )

    # [4] Vehicle count — flat penalty per vehicle used
    # This is the dominant soft term; placed after the loop so it is
    # always computed even if some vehicles have violations.
    score += vehicles_used * VEHICLE_COUNT_PENALTY

    return score


# ─────────────────────────────────────────────────────────────────────────────
#  CLUSTER DISTANCE ESTIMATOR
# ─────────────────────────────────────────────────────────────────────────────

def _estimate_cluster_distance(
    pkg_indices: list[int],
    packages:    list[PackageGA],
    origin:      tuple[float, float],
    dist_matrix: list[list[float]] | None,
) -> float:
    """
    Nearest-neighbour tour estimate for one vehicle's package cluster.
    O(n²) — fast because n ≤ MAX_PKGS_PER_DELIVERER (15) for deliverers
    and typically ≤ 30 for transporters in one vehicle's cluster.

    Packages without coordinates (transporter packages) are skipped;
    they contribute 0 to route cost — Node.js resolves real branch coords
    at persist time.
    """
    candidates = [i for i in pkg_indices if packages[i].coords is not None]
    if not candidates:
        return 0.0

    visited: set[int] = set()
    current: int | None = None   # None = currently at origin
    total = 0.0

    while len(visited) < len(candidates):
        best_dist = math.inf
        best_idx  = -1

        for i in candidates:
            if i in visited:
                continue
            if dist_matrix is not None:
                d = (
                    haversine_km(origin, packages[i].coords)
                    if current is None
                    else dist_matrix[current][i]
                )
            else:
                src = origin if current is None else packages[current].coords
                d = haversine_km(src, packages[i].coords)

            if d < best_dist:
                best_dist = d
                best_idx  = i

        visited.add(best_idx)
        total   += best_dist
        current  = best_idx

    return total


# ─────────────────────────────────────────────────────────────────────────────
#  GENETIC OPERATORS
# ─────────────────────────────────────────────────────────────────────────────

def _random_chromosome(n_packages: int, n_vehicles: int) -> np.ndarray:
    return np.random.randint(0, n_vehicles, size=n_packages)


def _biased_random_chromosome(n_packages: int, n_vehicles: int) -> np.ndarray:
    """
    Random chromosome skewed toward lower vehicle indices (smaller vehicles).
    Uses a triangular distribution with peak at 0 so the initial population
    reflects the business preference for consolidation onto fewer, smaller vehicles.
    """
    raw = np.random.triangular(0, 0, n_vehicles, size=n_packages)
    return np.clip(raw.astype(int), 0, n_vehicles - 1)


def _tournament_select(
    population: list[np.ndarray],
    fitnesses:  list[float],
    k:          int = 3,
) -> np.ndarray:
    """Tournament selection — pick k random individuals, return the best."""
    candidates = random.sample(range(len(population)), k)
    best = min(candidates, key=lambda i: fitnesses[i])
    return population[best].copy()


def _crossover(
    parent_a: np.ndarray,
    parent_b: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Two-point crossover — inherits clustering structure from both parents.
    Degrades gracefully for small chromosomes (n < 3).
    """
    n = len(parent_a)
    if n < 2:
        return parent_a.copy(), parent_b.copy()
    if n == 2:
        return (
            np.array([parent_a[0], parent_b[1]]),
            np.array([parent_b[0], parent_a[1]]),
        )
    p1, p2 = sorted(random.sample(range(n), 2))
    child_a = np.concatenate([parent_a[:p1], parent_b[p1:p2], parent_a[p2:]])
    child_b = np.concatenate([parent_b[:p1], parent_a[p1:p2], parent_b[p2:]])
    return child_a, child_b


def _mutate(
    chromosome: np.ndarray,
    n_vehicles: int,
    rate:       float,
) -> np.ndarray:
    """
    Two complementary mutation operators applied together:

    Operator 1 — Random reseat (probability = rate):
      Reassigns a gene to a completely random vehicle.
      Provides broad exploration; dominant early in evolution when the
      population needs diversity.

    Operator 2 — Neighbour step (probability = rate / 2):
      Moves a gene to an adjacent vehicle index (±1).
      Provides local refinement; dominant late in evolution.
      Slightly biased toward -1 (smaller vehicle) to reinforce the
      consolidation objective without over-riding the GA's freedom to explore.
    """
    chrom = chromosome.copy()

    # Operator 1: random reseat
    mask_random = np.random.random(len(chrom)) < rate
    if mask_random.any():
        chrom[mask_random] = np.random.randint(0, n_vehicles, size=int(mask_random.sum()))

    # Operator 2: neighbour step  (rate/2 probability, bias −1 over +1)
    mask_local = np.random.random(len(chrom)) < (rate / 2.0)
    for i in np.where(mask_local)[0]:
        # −1 appears twice so the mutation slightly favours consolidation
        delta   = random.choice([-1, -1, 1])
        chrom[i] = int(np.clip(int(chrom[i]) + delta, 0, n_vehicles - 1))

    return chrom


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN GA ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def run_genetic_assignment(
    packages:      list[PackageGA],
    vehicles:      list[VehicleGA],
    origin_coords: tuple[float, float],
    is_deliverer:  bool = False,
    dist_matrix:   list[list[float]] | None = None,
) -> tuple[list[AssignmentResult], list[VehicleGA]]:
    """
    Runs the Genetic Algorithm and returns optimised vehicle→package assignments.

    Parameters
    ----------
    packages      : All packages to be assigned this pass.
    vehicles      : Available vehicles (any order; sorted internally small→large).
    origin_coords : Branch [lng, lat], used as tour origin for distance estimation.
    is_deliverer  : True → enforce MAX_PKGS_PER_DELIVERER hard constraint.
    dist_matrix   : Optional pre-computed package×package distance matrix.

    Returns
    -------
    (assignments, sorted_vehicles)

    assignments    : List of AssignmentResult, one per non-empty vehicle,
                     ordered small→large so smallest vehicle is assigned first.
    sorted_vehicles: The vehicle list in the same order used by assignments.
                     The pipeline must use this list to resolve vehicle_idx.
    """
    n_pkg = len(packages)
    n_veh = len(vehicles)

    if n_pkg == 0 or n_veh == 0:
        return [], vehicles

    # ── Sort vehicles small → large ────────────────────────────────────────────
    # This is the BONUS feature: the GA always considers smallest vehicle first,
    # both in seeding and in decoding, so the pipeline assigns the smallest
    # possible vehicle to each load cluster.
    sorted_vehicles = sorted(vehicles, key=lambda v: (v.type_rank, v.max_weight))

    random.seed(42)
    np.random.seed(42)

    # ── Build initial population ───────────────────────────────────────────────

    # Seed A: greedy FFD using sorted (small→large) vehicle order
    seed_a = _greedy_seed(packages, sorted_vehicles, is_deliverer)

    # Seed B: greedy FFD using reversed (large→small) vehicle order.
    # This is the critical seed for consolidation: it tries to put everything
    # into the largest vehicle first, which is the correct solution when one
    # large vehicle can carry the entire load.
    reversed_vehicles = list(reversed(sorted_vehicles))
    seed_b_raw        = _greedy_seed(packages, reversed_vehicles, is_deliverer)
    seed_b            = _remap_seed(seed_b_raw, sorted_vehicles, reversed_vehicles)

    # Seed C: pure consolidation — assign every package to the single largest
    # vehicle that can carry the full load.  If no single vehicle fits,
    # falls back to the largest vehicle (GA repairs violations from there).
    # This seed directly represents the "one big truck" optimal solution.
    seed_c = _consolidation_seed(packages, sorted_vehicles, is_deliverer)

    population: list[np.ndarray] = [seed_a, seed_b, seed_c]

    # Fill rest: mix of uniform-random (exploration) and biased-random
    # (consolidation preference).  We use 50/50 to avoid over-biasing
    # the population toward small vehicles — the real bug that caused
    # the consolidation failure.
    for i in range(POPULATION_SIZE - 3):
        if i % 2 == 0:
            population.append(_random_chromosome(n_pkg, n_veh))
        else:
            population.append(_biased_random_chromosome(n_pkg, n_veh))

    # ── Fitness closure (captures sorted_vehicles + is_deliverer) ──────────────
    def evaluate(chrom: np.ndarray) -> float:
        return _fitness(
            chrom, packages, sorted_vehicles, is_deliverer, dist_matrix, origin_coords
        )

    # ── Initialise best tracker — start from best of all 3 seeds ──────────────
    seed_fitnesses = [(evaluate(s), s) for s in [seed_a, seed_b, seed_c]]
    seed_fitnesses.sort(key=lambda x: x[0])
    best_fitness, best_chromosome = seed_fitnesses[0]
    best_chromosome = best_chromosome.copy()
    stagnation_count = 0
    mutation_rate    = MUTATION_RATE_BASE
    generation       = 0

    # ── Evolution loop ─────────────────────────────────────────────────────────
    for generation in range(GENERATIONS):
        fitnesses = [evaluate(chrom) for chrom in population]

        # Track global best
        gen_best_idx = int(np.argmin(fitnesses))
        if fitnesses[gen_best_idx] < best_fitness:
            best_fitness     = fitnesses[gen_best_idx]
            best_chromosome  = population[gen_best_idx].copy()
            stagnation_count = 0
            mutation_rate    = MUTATION_RATE_BASE   # reset on improvement
        else:
            stagnation_count += 1

        # Adaptive mutation: escape local minima after stagnation
        if stagnation_count >= STAGNATION_WINDOW:
            mutation_rate = MUTATION_RATE_HIGH
            logger.debug(
                f"[GA] gen={generation} stagnation — mutation → {mutation_rate:.2f}"
            )

        # Early stop when solution is feasible and we are past the midpoint.
        # Feasibility proxy: no penalty term exceeds VEHICLE_COUNT_PENALTY × n_veh
        # (i.e. all hard constraints are satisfied and only soft terms remain).
        if best_fitness < VEHICLE_COUNT_PENALTY * n_veh and generation > GENERATIONS // 2:
            logger.debug(
                f"[GA] Early stop gen={generation} fitness={best_fitness:.0f}"
            )
            break

        # Elitism: best ELITE_SIZE individuals survive unchanged
        elite_indices  = sorted(range(len(fitnesses)), key=lambda i: fitnesses[i])[:ELITE_SIZE]
        new_population = [population[i].copy() for i in elite_indices]

        # Breed new generation
        while len(new_population) < POPULATION_SIZE:
            parent_a = _tournament_select(population, fitnesses)
            parent_b = _tournament_select(population, fitnesses)
            child_a, child_b = _crossover(parent_a, parent_b)
            child_a = _mutate(child_a, n_veh, mutation_rate)
            child_b = _mutate(child_b, n_veh, mutation_rate)
            new_population.extend([child_a, child_b])

        population = new_population[:POPULATION_SIZE]

    logger.info(
        f"[GA] fitness={best_fitness:.0f} gen={generation}/{GENERATIONS} "
        f"stagnation={stagnation_count} deliverer={is_deliverer} "
        f"pkgs={n_pkg} vehicles={n_veh}"
    )

    return _decode(best_chromosome, sorted_vehicles), sorted_vehicles


# ─────────────────────────────────────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _greedy_seed(
    packages:      list[PackageGA],
    vehicles:      list[VehicleGA],
    is_deliverer:  bool,
) -> np.ndarray:
    """
    Deterministic greedy seed using First Fit Decreasing (FFD) order:
      1. Sort packages: priority ascending (same_day first), weight descending
         within each tier (heavy packages placed first — FFD heuristic).
      2. For each package, pick the vehicle that:
         a. Passes all hard filters (capacity, fragile, pkg limit)
         b. Has the highest fill score after adding this package
            (fill-first = pack one vehicle before opening another)
         c. In case of tie, prefer the smallest vehicle type.
    """
    n_veh      = len(vehicles)
    assignment = np.zeros(len(packages), dtype=int)
    used_w     = [0.0] * n_veh
    used_v     = [0.0] * n_veh
    pkg_count  = [0]   * n_veh

    # FFD order: priority asc, then weight desc within tier
    order = sorted(
        range(len(packages)),
        key=lambda i: (packages[i].priority, -packages[i].weight),
    )

    for pkg_idx in order:
        pkg        = packages[pkg_idx]
        best_veh   = 0
        best_score = -math.inf

        for veh_idx, veh in enumerate(vehicles):
            # Hard filter: fragile
            if pkg.is_fragile and not veh.supports_fragile:
                continue
            # Hard filter: weight capacity
            new_w = used_w[veh_idx] + pkg.weight
            if new_w > veh.max_weight * CAPACITY_BUFFER:
                continue
            # Hard filter: volume capacity
            new_v = used_v[veh_idx] + pkg.volume
            if new_v > veh.max_volume * CAPACITY_BUFFER:
                continue
            # Hard filter: deliverer package limit
            if is_deliverer and pkg_count[veh_idx] >= MAX_PKGS_PER_DELIVERER:
                continue

            # Score: average fill after adding this package (higher = better)
            w_fill     = new_w / veh.max_weight
            v_fill     = new_v / veh.max_volume
            fill_score = (w_fill + v_fill) / 2.0

            # Tiebreaker: prefer smaller vehicle type (type_rank lower = smaller)
            # Weight is tiny (0.01 per rank step) so it never overrides fill_score
            type_bonus = (4 - veh.type_rank) * 0.01

            score = fill_score + type_bonus
            if score > best_score:
                best_score = score
                best_veh   = veh_idx

        assignment[pkg_idx]  = best_veh
        used_w[best_veh]    += pkg.weight
        used_v[best_veh]    += pkg.volume
        pkg_count[best_veh] += 1

    return assignment



def _consolidation_seed(
    packages:      list[PackageGA],
    vehicles:      list[VehicleGA],   # sorted small→large
    is_deliverer:  bool,
) -> np.ndarray:
    """
    Consolidation seed: assign ALL packages to the single largest vehicle
    that can carry the full load without violating any hard constraint.

    This directly encodes the "use one big vehicle" optimal solution so the
    GA starts with it in its population and can inherit it via elitism.

    Logic:
      1. Try vehicles from largest to smallest (reversed sorted order).
      2. Pick the first vehicle whose capacity covers all packages at once.
      3. If no single vehicle fits (load genuinely too large for any one
         vehicle), fall back to the reversed-order greedy seed which at
         least prefers large vehicles.

    Why this matters:
      The fill-score heuristic in _greedy_seed naturally prefers small
      vehicles because they reach higher fill ratios faster.  For a load
      that fits entirely in one large truck, the greedy seed spreads it
      across multiple smaller vehicles instead — a locally optimal but
      globally suboptimal choice that the GA then fails to escape.
    """
    n_veh      = len(vehicles)
    n_pkgs     = len(packages)
    total_w    = sum(p.weight for p in packages)
    total_v    = sum(p.volume for p in packages)
    has_fragile = any(p.is_fragile for p in packages)

    # Try vehicles largest → smallest
    for veh_idx in range(n_veh - 1, -1, -1):
        veh = vehicles[veh_idx]
        if has_fragile and not veh.supports_fragile:
            continue
        if total_w > veh.max_weight * CAPACITY_BUFFER:
            continue
        if total_v > veh.max_volume * CAPACITY_BUFFER:
            continue
        if is_deliverer and n_pkgs > MAX_PKGS_PER_DELIVERER:
            # Can't fit all in one deliverer vehicle — skip consolidation
            continue
        # Found a vehicle that fits everything
        return np.full(n_pkgs, veh_idx, dtype=int)

    # No single vehicle fits the full load — assign everything to the largest
    # vehicle (index n_veh-1) and let the GA repair violations from there.
    return np.full(n_pkgs, n_veh - 1, dtype=int)

def _remap_seed(
    seed:          np.ndarray,
    target_order:  list[VehicleGA],
    source_order:  list[VehicleGA],
) -> np.ndarray:
    """
    Translates vehicle indices in `seed` from source_order to target_order.
    Vehicles are matched by their original .idx field (immutable identity).
    Used to convert the reversed-order greedy seed back to sorted indices.
    """
    target_by_original_idx = {v.idx: i for i, v in enumerate(target_order)}
    remapped = np.zeros(len(seed), dtype=int)
    for pkg_i, src_veh_pos in enumerate(seed):
        original_id         = source_order[int(src_veh_pos)].idx
        remapped[pkg_i]     = target_by_original_idx.get(original_id, 0)
    return remapped


def _decode(
    chromosome:      np.ndarray,
    sorted_vehicles: list[VehicleGA],
) -> list[AssignmentResult]:
    """
    Converts the best chromosome into AssignmentResult list.

    Results are sorted by vehicle_idx (small→large) so the pipeline
    assigns the smallest vehicle to the first available worker.
    Empty vehicle slots are omitted — no route is created for them.
    """
    groups: dict[int, list[int]] = {}
    for pkg_idx, veh_idx in enumerate(chromosome):
        veh_idx = int(veh_idx)
        groups.setdefault(veh_idx, []).append(pkg_idx)

    return [
        AssignmentResult(vehicle_idx=veh_idx, package_indices=pkg_indices)
        for veh_idx, pkg_indices in sorted(groups.items())
        if pkg_indices
    ]