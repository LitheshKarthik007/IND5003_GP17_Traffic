"""
TraFlow AI - ML + Propagation + Adaptive Signal Pipeline

Connects the existing TraFlow inference result with:
    - Traffic simulation
    - Adaptive signal engine
    - Before/after comparison
"""

import json
from pathlib import Path

from traffic_simulation import (
    FourWayTrafficSimulation,
    create_demo_scenario,
    DIRECTIONS,
)

from signal_engine import (
    generate_signal_recommendation,
)


# ============================================================
# PATHS
# ============================================================

RESULT_FILE = Path(
    "scripts/artifacts/final_inference/final_result.json"
)


# ============================================================
# LOAD REAL TRAFLOW RESULT
# ============================================================

def load_traflow_result():

    if not RESULT_FILE.exists():

        raise FileNotFoundError(
            f"TraFlow result not found:\n"
            f"{RESULT_FILE}"
        )

    with open(
        RESULT_FILE,
        "r",
        encoding="utf-8",
    ) as file:

        return json.load(file)


# ============================================================
# CONVERT PREDICTION TO STATE
# ============================================================

def get_predicted_state(
    result,
):
    """
    Use the 30-minute prediction.

    0 = Very Light
    1 = Light
    2 = Moderate
    3 = Severe
    """

    values = result.get(
        "predicted_state_values",
        [],
    )

    if not values:

        raise ValueError(
            "No predicted_state_values found "
            "in final_result.json"
        )

    return int(
        values[-1]
    )


# ============================================================
# RUN SIMULATION
# ============================================================

def run_simulation(
    traffic,
    direction,
    green_time,
    steps=30,
):

    simulation = FourWayTrafficSimulation(
        traffic=traffic,
        seed=42,
    )

    simulation.set_signal(
        direction,
        green_time,
    )

    for _ in range(steps):

        simulation.step(
            arrival_scale=1.0
        )

    return simulation.get_state()


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)

    print(
        "TRAFLOW REAL AI SIGNAL PIPELINE"
    )

    print("=" * 70)

    # ========================================================
    # 1. LOAD REAL ML RESULT
    # ========================================================

    result = load_traflow_result()

    print(
        "\n✓ Loaded:"
    )

    print(
        RESULT_FILE
    )

    # ========================================================
    # 2. READ REAL TRAFLOW VALUES
    # ========================================================

    propagation_risk = float(
        result.get(
            "propagation_risk",
            0,
        )
    )

    trend = result.get(
        "trend",
        "UNKNOWN",
    )

    strongest_upstream = result.get(
        "strongest_upstream_camera",
        None,
    )

    target_camera = result.get(
        "target_camera",
        2704,
    )

    predicted_state = get_predicted_state(
        result
    )

    # ========================================================
    # 3. PRINT REAL ML INTELLIGENCE
    # ========================================================

    print(
        "\n" + "=" * 70
    )

    print(
        "REAL TRAFLOW INTELLIGENCE"
    )

    print(
        "=" * 70
    )

    print(
        "\nTarget camera:",
        target_camera,
    )

    print(
        "Predicted 30-minute state:",
        predicted_state,
    )

    print(
        "Propagation risk:",
        propagation_risk,
    )

    print(
        "Traffic trend:",
        trend,
    )

    print(
        "Strongest upstream:",
        strongest_upstream,
    )

    # ========================================================
    # 4. CREATE TRAFFIC SCENARIO
    # ========================================================
    #
    # This is currently the simulation input.
    #
    # Later the video/detection layer will replace
    # this with measured directional traffic.
    # ========================================================

    traffic = create_demo_scenario(
        "imbalanced"
    )

    print(
        "\n" + "=" * 70
    )

    print(
        "SIMULATED INTERSECTION TRAFFIC"
    )

    print(
        "=" * 70
    )

    for direction in DIRECTIONS:

        print(
            f"{direction:>6}: "
            f"{traffic[direction]}"
        )

    # ========================================================
    # 5. BASELINE SIGNAL
    # ========================================================

    baseline_direction = "East"

    baseline_green_time = 30

    print(
        "\nCurrent signal:",
        baseline_direction,
        "= GREEN",
    )

    # ========================================================
    # 6. RUN BASELINE
    # ========================================================

    baseline = run_simulation(

        traffic=traffic,

        direction=baseline_direction,

        green_time=baseline_green_time,
    )

    # ========================================================
    # 7. GET QUEUE DATA
    # ========================================================

    queues = baseline[
        "queues"
    ]

    queue_growth = baseline[
        "queue_growth"
    ]

    # ========================================================
    # 8. REAL AI SIGNAL DECISION
    # ========================================================

    print(
        "\n" + "=" * 70
    )

    print(
        "ADAPTIVE SIGNAL DECISION"
    )

    print(
        "=" * 70
    )

    recommendation = (
        generate_signal_recommendation(

            traffic=traffic,

            queues=queues,

            queue_growth=queue_growth,

            current_green=
                baseline_green_time,

            predicted_congestion=
                predicted_state,

            propagation_risk=
                propagation_risk,

            current_direction=
                baseline_direction,
        )
    )

    print(
        "\nAI Action:",
        recommendation.action,
    )

    print(
        "Selected direction:",
        recommendation.selected_direction,
    )

    print(
        "Recommended green:",
        recommendation.green_time,
        "seconds",
    )

    print(
        "Adjustment:",
        recommendation.adjustment,
        "seconds",
    )

    print(
        "Priority score:",
        recommendation.score,
    )

    print(
        "\nReason:"
    )

    print(
        recommendation.reason
    )

    # ========================================================
    # 9. RUN AI SIMULATION
    # ========================================================

    ai_state = run_simulation(

        traffic=traffic,

        direction=
            recommendation.selected_direction,

        green_time=
            recommendation.green_time,
    )

    # ========================================================
    # 10. BEFORE / AFTER
    # ========================================================

    print(
        "\n" + "=" * 70
    )

    print(
        "BEFORE vs AFTER"
    )

    print(
        "=" * 70
    )

    print(
        f"{'Metric':<25}"
        f"{'Before':>12}"
        f"{'After':>12}"
    )

    print(
        "-" * 50
    )

    print(
        f"{'Total Queue':<25}"
        f"{baseline['total_queue']:>12}"
        f"{ai_state['total_queue']:>12}"
    )

    print(
        f"{'Waiting Time':<25}"
        f"{baseline['waiting_time']:>12}"
        f"{ai_state['waiting_time']:>12}"
    )

    print(
        f"{'Throughput':<25}"
        f"{baseline['throughput']:>12}"
        f"{ai_state['throughput']:>12}"
    )

    print(
        f"{'Corridor Flow (%)':<25}"
        f"{baseline['corridor_flow']:>12.2f}"
        f"{ai_state['corridor_flow']:>12.2f}"
    )

    # ========================================================
    # 11. IMPROVEMENT
    # ========================================================

    if baseline["total_queue"] > 0:

        queue_improvement = (
            (
                baseline["total_queue"]
                -
                ai_state["total_queue"]
            )
            /
            baseline["total_queue"]
        ) * 100

    else:

        queue_improvement = 0

    if baseline["waiting_time"] > 0:

        waiting_improvement = (
            (
                baseline["waiting_time"]
                -
                ai_state["waiting_time"]
            )
            /
            baseline["waiting_time"]
        ) * 100

    else:

        waiting_improvement = 0

    throughput_improvement = (
        ai_state["throughput"]
        -
        baseline["throughput"]
    )

    corridor_flow_change = (
        ai_state["corridor_flow"]
        -
        baseline["corridor_flow"]
    )

    # ========================================================
    # 12. FINAL RESULTS
    # ========================================================

    print(
        "\n" + "=" * 70
    )

    print(
        "TRAFLOW CORRIDOR IMPACT"
    )

    print(
        "=" * 70
    )

    print(
        f"Queue reduction:"
        f" {queue_improvement:.2f}%"
    )

    print(
        f"Waiting-time reduction:"
        f" {waiting_improvement:.2f}%"
    )

    print(
        f"Throughput change:"
        f" +{throughput_improvement}"
    )

    print(
        f"Corridor-flow change:"
        f" +{corridor_flow_change:.2f}"
        f" percentage points"
    )

    # ========================================================
    # 13. FINAL DECISION
    # ========================================================

    print(
        "\n" + "=" * 70
    )

    print(
        "FINAL TRAFLOW DECISION"
    )

    print(
        "=" * 70
    )

    print(
        f"Current signal:"
        f" {baseline_direction} GREEN"
    )

    print(
        f"AI signal:"
        f" {recommendation.selected_direction} GREEN"
    )

    print(
        f"Green time:"
        f" {baseline_green_time}s"
        f" → "
        f"{recommendation.green_time}s"
    )

    print(
        "\nDecision based on:"
    )

    print(
        "✓ Directional traffic demand"
    )

    print(
        "✓ Queue pressure"
    )

    print(
        "✓ Queue growth"
    )

    print(
        "✓ Real LSTM prediction"
    )

    print(
        "✓ Real propagation risk"
    )

    print(
        "✓ Corridor-level intelligence"
    )

    print(
        "\n" + "=" * 70
    )

    print(
        "REAL AI SIGNAL PIPELINE COMPLETE"
    )

    print(
        "=" * 70
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()