"""
TraFlow AI - Before vs After Adaptive Signal Simulation

Purpose
-------
Compare:

1. Fixed / inefficient signal timing
2. AI-adaptive signal timing

Metrics:
    - Total queue
    - Waiting time
    - Throughput
    - Corridor flow
    - Queue reduction
    - Waiting-time reduction
    - Throughput improvement
    - Corridor-flow improvement

This is a prototype simulation.
It does NOT control real traffic infrastructure.
"""

from traffic_simulation import (
    FourWayTrafficSimulation,
    create_demo_scenario,
    DIRECTIONS,
)

from signal_engine import (
    generate_signal_recommendation,
)


# ============================================================
# CONFIGURATION
# ============================================================

SIMULATION_STEPS = 30

ARRIVAL_SCALE = 1.0

# Deliberately inefficient signal:
# East has low traffic but receives green.
BASELINE_SIGNAL = "East"

BASELINE_GREEN_TIME = 30


# ============================================================
# PRINT SECTION
# ============================================================

def print_section(title):

    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


# ============================================================
# RUN SIMULATION
# ============================================================

def run_simulation(
    traffic,
    signal_direction,
    green_time,
    steps=SIMULATION_STEPS,
):
    """
    Run a traffic simulation using a fixed signal plan.
    """

    simulation = FourWayTrafficSimulation(
        traffic=traffic,
        seed=42,
    )

    simulation.set_signal(
        signal_direction,
        green_time,
    )

    for _ in range(steps):

        simulation.step(
            arrival_scale=ARRIVAL_SCALE
        )

    return simulation.get_state()


# ============================================================
# PERCENTAGE REDUCTION
# ============================================================

def percentage_reduction(
    before,
    after,
):
    """
    Calculate percentage reduction.

    Positive value = improvement.
    """

    if before == 0:

        return 0.0

    return round(
        (
            (before - after)
            / before
        )
        * 100,
        2,
    )


# ============================================================
# PERCENTAGE INCREASE
# ============================================================

def percentage_increase(
    before,
    after,
):
    """
    Calculate percentage increase.

    Positive value = improvement.
    """

    if before == 0:

        return 0.0

    return round(
        (
            (after - before)
            / before
        )
        * 100,
        2,
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print_section(
        "TRAFLOW BEFORE vs AFTER SIGNAL SIMULATION"
    )

    # ========================================================
    # 1. CREATE TRAFFIC SCENARIO
    # ========================================================

    traffic = create_demo_scenario(
        "imbalanced"
    )

    print(
        "\nTraffic scenario:"
    )

    for direction in DIRECTIONS:

        print(
            f"{direction:>6} : "
            f"{traffic[direction]}"
        )

    # ========================================================
    # 2. SHOW PROBLEM
    # ========================================================

    print_section(
        "INITIAL TRAFFIC PROBLEM"
    )

    print(
        "Current signal:",
        BASELINE_SIGNAL,
        "= GREEN"
    )

    print(
        "\nTraffic demand:"
    )

    for direction in DIRECTIONS:

        print(
            f"{direction:>6} : "
            f"{traffic[direction]}"
        )

    print(
        "\nObservation:"
    )

    print(
        "North has the highest traffic demand,"
    )

    print(
        "but East currently receives the green signal."
    )

    print(
        "\nThis represents inefficient signal allocation."
    )

    # ========================================================
    # 3. RUN BASELINE
    # ========================================================

    print_section(
        "BASELINE SIMULATION"
    )

    baseline = run_simulation(

        traffic=traffic,

        signal_direction=BASELINE_SIGNAL,

        green_time=BASELINE_GREEN_TIME,
    )

    print(
        "\nBaseline signal:"
    )

    print(
        f"{BASELINE_SIGNAL} = GREEN"
    )

    print(
        f"Green time = "
        f"{BASELINE_GREEN_TIME} seconds"
    )

    print(
        "\nQueues:"
    )

    for direction in DIRECTIONS:

        print(
            f"{direction:>6} : "
            f"{baseline['queues'][direction]}"
        )

    print(
        "\nTotal queue:",
        baseline["total_queue"]
    )

    print(
        "Waiting time:",
        baseline["waiting_time"]
    )

    print(
        "Throughput:",
        baseline["throughput"]
    )

    print(
        "Corridor flow:",
        f"{baseline['corridor_flow']}%"
    )

    # ========================================================
    # 4. GET QUEUE INFORMATION FOR AI
    # ========================================================

    queue_growth = (
        baseline["queue_growth"]
    )

    queues = (
        baseline["queues"]
    )

    # ========================================================
    # 5. GENERATE AI SIGNAL DECISION
    # ========================================================

    print_section(
        "TRAFLOW AI SIGNAL DECISION"
    )

    # For this prototype we assume that
    # the downstream model detected severe
    # future congestion and the propagation
    # analysis produced a high risk.

    predicted_congestion = 3

    propagation_risk = 75.0

    recommendation = (
        generate_signal_recommendation(

            traffic=traffic,

            queues=queues,

            queue_growth=queue_growth,

            current_green=BASELINE_GREEN_TIME,

            predicted_congestion=
                predicted_congestion,

            propagation_risk=
                propagation_risk,

            current_direction=
                BASELINE_SIGNAL,
        )
    )

    print(
        "Current signal:",
        BASELINE_SIGNAL
    )

    print(
        "Predicted downstream congestion:",
        predicted_congestion
    )

    print(
        "Propagation risk:",
        propagation_risk
    )

    print(
        "\nAI action:",
        recommendation.action
    )

    print(
        "AI selected direction:",
        recommendation.selected_direction
    )

    print(
        "Previous green time:",
        recommendation.previous_green_time,
        "seconds"
    )

    print(
        "Recommended green time:",
        recommendation.green_time,
        "seconds"
    )

    print(
        "Timing adjustment:",
        recommendation.adjustment,
        "seconds"
    )

    print(
        "\nReason:"
    )

    print(
        recommendation.reason
    )

    # ========================================================
    # 6. RUN AI SIMULATION
    # ========================================================

    print_section(
        "AI-ADAPTIVE SIMULATION"
    )

    ai_state = run_simulation(

        traffic=traffic,

        signal_direction=
            recommendation.selected_direction,

        green_time=
            recommendation.green_time,
    )

    print(
        "\nAI signal:"
    )

    print(
        f"{recommendation.selected_direction}"
        f" = GREEN"
    )

    print(
        f"Green time = "
        f"{recommendation.green_time} seconds"
    )

    print(
        "\nQueues:"
    )

    for direction in DIRECTIONS:

        print(
            f"{direction:>6} : "
            f"{ai_state['queues'][direction]}"
        )

    print(
        "\nTotal queue:",
        ai_state["total_queue"]
    )

    print(
        "Waiting time:",
        ai_state["waiting_time"]
    )

    print(
        "Throughput:",
        ai_state["throughput"]
    )

    print(
        "Corridor flow:",
        f"{ai_state['corridor_flow']}%"
    )

    # ========================================================
    # 7. CALCULATE IMPROVEMENTS
    # ========================================================

    print_section(
        "BEFORE vs AFTER RESULTS"
    )

    queue_reduction = percentage_reduction(

        baseline["total_queue"],

        ai_state["total_queue"],
    )

    waiting_reduction = percentage_reduction(

        baseline["waiting_time"],

        ai_state["waiting_time"],
    )

    throughput_improvement = percentage_increase(

        baseline["throughput"],

        ai_state["throughput"],
    )

    corridor_flow_improvement = (
        round(
            ai_state["corridor_flow"]
            -
            baseline["corridor_flow"],
            2,
        )
    )

    print(
        "\nMETRIC"
    )

    print(
        "-" * 70
    )

    print(
        f"{'Total Queue':<30}"
        f"{baseline['total_queue']:>10}"
        f"{ai_state['total_queue']:>10}"
        f"{queue_reduction:>15.2f}%"
    )

    print(
        f"{'Waiting Time':<30}"
        f"{baseline['waiting_time']:>10}"
        f"{ai_state['waiting_time']:>10}"
        f"{waiting_reduction:>15.2f}%"
    )

    print(
        f"{'Throughput':<30}"
        f"{baseline['throughput']:>10}"
        f"{ai_state['throughput']:>10}"
        f"{throughput_improvement:>15.2f}%"
    )

    print(
        f"{'Corridor Flow (%)':<30}"
        f"{baseline['corridor_flow']:>10.2f}"
        f"{ai_state['corridor_flow']:>10.2f}"
        f"{corridor_flow_improvement:>14.2f}"
    )

    # ========================================================
    # 8. FINAL INTERPRETATION
    # ========================================================

    print_section(
        "TRAFLOW CORRIDOR IMPACT"
    )

    if queue_reduction > 0:

        print(
            f"✓ Queue reduced by "
            f"{queue_reduction}%."
        )

    else:

        print(
            "⚠ Queue did not decrease."
        )

    if waiting_reduction > 0:

        print(
            f"✓ Waiting time reduced by "
            f"{waiting_reduction}%."
        )

    else:

        print(
            "⚠ Waiting time did not decrease."
        )

    if throughput_improvement > 0:

        print(
            f"✓ Throughput improved by "
            f"{throughput_improvement}%."
        )

    else:

        print(
            "⚠ Throughput did not improve."
        )

    if corridor_flow_improvement > 0:

        print(
            f"✓ Corridor flow improved by "
            f"{corridor_flow_improvement} percentage points."
        )

    else:

        print(
            "⚠ Corridor flow did not improve."
        )

    # ========================================================
    # 9. FINAL DECISION
    # ========================================================

    print_section(
        "FINAL TRAFLOW DECISION"
    )

    print(
        "BEFORE:"
    )

    print(
        f"{BASELINE_SIGNAL} = GREEN"
    )

    print(
        "\nAFTER:"
    )

    print(
        f"{recommendation.selected_direction}"
        f" = GREEN"
    )

    print(
        f"\nGreen time changed:"
        f" {BASELINE_GREEN_TIME}s"
        f" → "
        f"{recommendation.green_time}s"
    )

    print(
        "\nTraFlow used:"
    )

    print(
        "Traffic demand"
    )

    print(
        "+ Queue pressure"
    )

    print(
        "+ Queue growth"
    )

    print(
        "+ Future congestion prediction"
    )

    print(
        "+ Corridor propagation risk"
    )

    print(
        "\n→ Adaptive signal recommendation"
    )

    print(
        "\n" + "=" * 70
    )

    print(
        "BEFORE vs AFTER SIMULATION COMPLETE"
    )

    print(
        "=" * 70
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()