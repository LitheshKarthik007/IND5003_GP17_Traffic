"""
TraFlow AI - Four-Way Traffic Simulation Engine

Simulates a four-way intersection with:

    North
      |
      |
West--+--East
      |
      |
    South

The simulation models:
    - Vehicle arrivals
    - Traffic queues
    - Traffic signal state
    - Green-time allocation
    - Vehicle departures
    - Waiting time
    - Throughput
    - Queue growth
    - Corridor flow

This is a prototype simulation.
It does NOT control real-world traffic signals.
"""

import random
from dataclasses import dataclass


# ============================================================
# CONFIGURATION
# ============================================================

DIRECTIONS = [
    "North",
    "East",
    "South",
    "West",
]

MIN_GREEN_TIME = 15
MAX_GREEN_TIME = 60

VEHICLES_PER_GREEN_STEP = 3


# ============================================================
# VEHICLE
# ============================================================

@dataclass
class Vehicle:

    direction: str

    position: float = 100.0

    speed: float = 1.0

    waiting: bool = True

    passed: bool = False


# ============================================================
# FOUR-WAY TRAFFIC SIMULATION
# ============================================================

class FourWayTrafficSimulation:

    def __init__(
        self,
        traffic=None,
        seed=42,
    ):

        random.seed(seed)

        self.time = 0

        # ----------------------------------------------------
        # Traffic demand
        # ----------------------------------------------------

        self.traffic = {
            direction: 10
            for direction in DIRECTIONS
        }

        if traffic is not None:

            for direction in DIRECTIONS:

                if direction in traffic:

                    self.traffic[direction] = int(
                        max(
                            0,
                            traffic[direction]
                        )
                    )

        # ----------------------------------------------------
        # Queues
        # ----------------------------------------------------

        self.queues = {
            direction: 0
            for direction in DIRECTIONS
        }

        self.previous_queues = {
            direction: 0
            for direction in DIRECTIONS
        }

        # ----------------------------------------------------
        # Arrivals / departures
        # ----------------------------------------------------

        self.total_arrivals = {
            direction: 0
            for direction in DIRECTIONS
        }

        self.total_departures = {
            direction: 0
            for direction in DIRECTIONS
        }

        # ----------------------------------------------------
        # Waiting time
        # ----------------------------------------------------

        self.total_waiting_time = {
            direction: 0
            for direction in DIRECTIONS
        }

        # ----------------------------------------------------
        # Signal
        # ----------------------------------------------------

        self.signal_direction = "North"

        self.signal_state = "GREEN"

        self.green_time = 30

        self.signal_elapsed = 0

        # ----------------------------------------------------
        # Vehicle list
        # ----------------------------------------------------

        self.vehicles = []

    # ========================================================
    # SET TRAFFIC DEMAND
    # ========================================================

    def set_traffic(
        self,
        traffic,
    ):

        for direction in DIRECTIONS:

            value = traffic.get(
                direction,
                0
            )

            self.traffic[direction] = int(
                max(
                    0,
                    value
                )
            )

    # ========================================================
    # SET SIGNAL
    # ========================================================

    def set_signal(
        self,
        direction,
        green_time=30,
    ):

        if direction not in DIRECTIONS:

            raise ValueError(
                f"Invalid direction: {direction}"
            )

        self.signal_direction = direction

        self.signal_state = "GREEN"

        self.green_time = int(
            max(
                MIN_GREEN_TIME,
                min(
                    green_time,
                    MAX_GREEN_TIME
                )
            )
        )

        self.signal_elapsed = 0

    # ========================================================
    # GENERATE VEHICLE ARRIVALS
    # ========================================================

    def generate_arrivals(
        self,
        arrival_scale=1.0,
    ):

        for direction in DIRECTIONS:

            demand = self.traffic[direction]

            # Base arrival rate.
            #
            # Higher traffic demand produces
            # more vehicles entering the queue.

            arrivals = int(
                round(
                    demand
                    / 20.0
                    * arrival_scale
                )
            )

            # Small random variation.

            if random.random() < 0.25:

                arrivals += 1

            arrivals = max(
                0,
                arrivals
            )

            # Add to queue.

            self.queues[direction] += arrivals

            # Record arrivals.

            self.total_arrivals[
                direction
            ] += arrivals

            # Add simulated vehicles.

            for _ in range(arrivals):

                self.vehicles.append(
                    Vehicle(
                        direction=direction
                    )
                )

    # ========================================================
    # PROCESS GREEN SIGNAL
    # ========================================================

    def process_signal(
        self,
    ):

        # No vehicles can leave during red/yellow.

        if self.signal_state != "GREEN":

            return 0

        direction = self.signal_direction

        # Maximum vehicles that can pass
        # during one simulation step.

        capacity = VEHICLES_PER_GREEN_STEP

        departures = min(
            self.queues[direction],
            capacity
        )

        # Remove vehicles from queue.

        self.queues[direction] -= departures

        # Record departures.

        self.total_departures[
            direction
        ] += departures

        # Mark simulated vehicles as passed.

        remaining = departures

        for vehicle in self.vehicles:

            if remaining <= 0:

                break

            if (
                vehicle.direction == direction
                and not vehicle.passed
            ):

                vehicle.passed = True

                vehicle.waiting = False

                vehicle.position = 0

                remaining -= 1

        return departures

    # ========================================================
    # UPDATE WAITING TIME
    # ========================================================

    def update_waiting_time(
        self,
    ):

        for direction in DIRECTIONS:

            queue = self.queues[direction]

            # Vehicles not receiving green
            # accumulate waiting time.

            if direction != self.signal_direction:

                self.total_waiting_time[
                    direction
                ] += queue

            else:

                # Some vehicles may still remain
                # even while green is active.

                remaining = max(
                    0,
                    queue - VEHICLES_PER_GREEN_STEP
                )

                self.total_waiting_time[
                    direction
                ] += remaining

    # ========================================================
    # UPDATE SIGNAL TIMER
    # ========================================================

    def update_signal_timer(
        self,
    ):

        self.signal_elapsed += 1

        # When green time expires,
        # switch briefly to yellow.

        if (
            self.signal_elapsed
            >= self.green_time
        ):

            self.signal_state = "YELLOW"

            self.signal_elapsed = 0

    # ========================================================
    # SIMULATION STEP
    # ========================================================

    def step(
        self,
        arrival_scale=1.0,
    ):

        # Save previous queues.

        self.previous_queues = (
            self.queues.copy()
        )

        # Generate new vehicles.

        self.generate_arrivals(
            arrival_scale
        )

        # Allow vehicles through
        # the active green direction.

        departures = (
            self.process_signal()
        )

        # Update waiting time.

        self.update_waiting_time()

        # Update signal timer.

        self.update_signal_timer()

        # For this prototype we use a very
        # short yellow transition.

        if self.signal_state == "YELLOW":

            self.signal_state = "GREEN"

            self.signal_elapsed = 0

        self.time += 1

        return departures

    # ========================================================
    # QUEUE GROWTH
    # ========================================================

    def queue_growth(
        self,
    ):

        growth = {}

        for direction in DIRECTIONS:

            growth[direction] = (
                self.queues[direction]
                -
                self.previous_queues[direction]
            )

        return growth

    # ========================================================
    # TOTAL QUEUE
    # ========================================================

    def total_queue(
        self,
    ):

        return sum(
            self.queues.values()
        )

    # ========================================================
    # TOTAL THROUGHPUT
    # ========================================================

    def total_throughput(
        self,
    ):

        return sum(
            self.total_departures.values()
        )

    # ========================================================
    # TOTAL WAITING TIME
    # ========================================================

    def total_waiting(
        self,
    ):

        return sum(
            self.total_waiting_time.values()
        )

    # ========================================================
    # CORRIDOR FLOW SCORE
    # ========================================================

    def corridor_flow_score(
        self,
    ):

        total_arrivals = sum(
            self.total_arrivals.values()
        )

        total_departures = sum(
            self.total_departures.values()
        )

        if total_arrivals == 0:

            return 0.0

        score = (
            total_departures
            /
            total_arrivals
        ) * 100

        return round(
            score,
            2
        )

    # ========================================================
    # APPROACH DEMAND
    # ========================================================

    def demand_scores(
        self,
    ):

        scores = {}

        for direction in DIRECTIONS:

            vehicle_score = min(
                self.traffic[direction]
                / 100.0,
                1.0
            )

            queue_score = min(
                self.queues[direction]
                / 50.0,
                1.0
            )

            growth = self.queue_growth()[
                direction
            ]

            growth_score = min(
                max(growth, 0)
                / 10.0,
                1.0
            )

            score = (
                vehicle_score * 0.45
                +
                queue_score * 0.35
                +
                growth_score * 0.20
            )

            scores[direction] = round(
                score * 100,
                2
            )

        return scores

    # ========================================================
    # GET CURRENT STATE
    # ========================================================

    def get_state(
        self,
    ):

        return {

            "time":
                self.time,

            "signal_direction":
                self.signal_direction,

            "signal_state":
                self.signal_state,

            "green_time":
                self.green_time,

            "signal_elapsed":
                self.signal_elapsed,

            "traffic":
                self.traffic.copy(),

            "queues":
                self.queues.copy(),

            "queue_growth":
                self.queue_growth(),

            "demand_scores":
                self.demand_scores(),

            "total_queue":
                self.total_queue(),

            "throughput":
                self.total_throughput(),

            "waiting_time":
                self.total_waiting(),

            "corridor_flow":
                self.corridor_flow_score(),
        }


# ============================================================
# DEMO TRAFFIC SCENARIOS
# ============================================================

def create_demo_scenario(
    scenario="imbalanced",
):
    """
    Create traffic scenarios for testing.

    balanced:
        Similar traffic on all approaches.

    imbalanced:
        One direction has significantly
        higher demand.

    extreme:
        One direction has extremely
        high demand.
    """

    if scenario == "balanced":

        return {

            "North": 20,

            "East": 18,

            "South": 21,

            "West": 19,

        }

    if scenario == "extreme":

        return {

            "North": 90,

            "East": 10,

            "South": 12,

            "West": 8,

        }

    # Default scenario.

    return {

        "North": 65,

        "East": 12,

        "South": 15,

        "West": 10,

    }


# ============================================================
# SIMPLE STANDALONE TEST
# ============================================================

if __name__ == "__main__":

    print("=" * 60)

    print(
        "TRAFLOW FOUR-WAY TRAFFIC SIMULATION"
    )

    print("=" * 60)

    traffic = create_demo_scenario(
        "imbalanced"
    )

    print("\nTraffic demand:")

    for direction in DIRECTIONS:

        print(
            f"{direction}: "
            f"{traffic[direction]}"
        )

    simulation = FourWayTrafficSimulation(
        traffic=traffic
    )

    # Deliberately give green to
    # a low-demand direction.

    simulation.set_signal(
        "East",
        green_time=30
    )

    print(
        "\nInitial signal:"
    )

    print(
        "East = GREEN"
    )

    # Run simulation.

    for _ in range(30):

        simulation.step(
            arrival_scale=1.0
        )

    state = simulation.get_state()

    print(
        "\nQueues:"
    )

    for direction in DIRECTIONS:

        print(
            f"{direction}: "
            f"{state['queues'][direction]}"
        )

    print(
        "\nQueue growth:"
    )

    for direction in DIRECTIONS:

        print(
            f"{direction}: "
            f"{state['queue_growth'][direction]}"
        )

    print(
        "\nDemand scores:"
    )

    for direction in DIRECTIONS:

        print(
            f"{direction}: "
            f"{state['demand_scores'][direction]}"
        )

    print(
        "\nTotal queue:",
        state["total_queue"]
    )

    print(
        "Throughput:",
        state["throughput"]
    )

    print(
        "Waiting time:",
        state["waiting_time"]
    )

    print(
        "Corridor flow:",
        f"{state['corridor_flow']}%"
    )

    print(
        "\nCurrent signal:",
        state["signal_direction"]
    )

    print(
        "Green time:",
        state["green_time"],
        "seconds"
    )

    print("\nDONE.")