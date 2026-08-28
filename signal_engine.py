"""
TraFlow AI - Adaptive Traffic Signal Decision Engine

This module decides which approach of a four-way
intersection should receive green signal.

Decision factors:
    1. Traffic demand
    2. Queue length
    3. Queue growth
    4. Predicted downstream congestion
    5. Corridor propagation risk

This is a SIMULATION / DECISION-SUPPORT system.
It does not control real-world traffic signals.
"""

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


# ============================================================
# SIGNAL RECOMMENDATION
# ============================================================

@dataclass
class SignalRecommendation:

    selected_direction: str

    action: str

    green_time: int

    previous_green_time: int

    adjustment: int

    reason: str

    score: float


# ============================================================
# HELPER
# ============================================================

def clamp(
    value,
    minimum,
    maximum,
):

    return max(
        minimum,
        min(
            value,
            maximum
        )
    )


# ============================================================
# DIRECTION DEMAND SCORE
# ============================================================

def calculate_direction_score(
    vehicle_count,
    queue_length,
    queue_growth,
):
    """
    Calculate the priority score of one approach.

    Score =

        Vehicle demand  -> 45%
        Queue length    -> 35%
        Queue growth    -> 20%

    Higher score means higher signal priority.
    """

    # --------------------------------------------------------
    # Vehicle component
    # --------------------------------------------------------

    vehicle_component = (
        min(
            max(vehicle_count, 0) / 100.0,
            1.0
        )
        * 45
    )

    # --------------------------------------------------------
    # Queue component
    # --------------------------------------------------------

    queue_component = (
        min(
            max(queue_length, 0) / 100.0,
            1.0
        )
        * 35
    )

    # --------------------------------------------------------
    # Queue growth component
    # --------------------------------------------------------

    growth_component = (
        min(
            max(queue_growth, 0) / 20.0,
            1.0
        )
        * 20
    )

    score = (
        vehicle_component
        + queue_component
        + growth_component
    )

    return round(
        score,
        2
    )


# ============================================================
# CALCULATE ALL DEMAND SCORES
# ============================================================

def calculate_all_scores(
    traffic,
    queues,
    queue_growth,
):
    """
    Calculate demand score for all four directions.
    """

    scores = {}

    for direction in DIRECTIONS:

        scores[direction] = (
            calculate_direction_score(

                traffic.get(
                    direction,
                    0
                ),

                queues.get(
                    direction,
                    0
                ),

                queue_growth.get(
                    direction,
                    0
                ),
            )
        )

    return scores


# ============================================================
# SIGNAL DECISION
# ============================================================

def generate_signal_recommendation(
    traffic,
    queues,
    queue_growth,
    current_green,
    predicted_congestion,
    propagation_risk,
    current_direction,
):
    """
    Generate an adaptive signal recommendation.

    Parameters
    ----------

    traffic:
        Current vehicle demand for each approach.

    queues:
        Current queue length.

    queue_growth:
        Change in queue length.

    current_green:
        Current green time in seconds.

    predicted_congestion:
        Predicted downstream state.

        0 = Very Light
        1 = Light
        2 = Moderate
        3 = Severe

    propagation_risk:
        Corridor propagation risk from 0 to 100.

    current_direction:
        Current green direction.
    """

    # ========================================================
    # STEP 1
    # Calculate base traffic demand
    # ========================================================

    scores = calculate_all_scores(
        traffic,
        queues,
        queue_growth,
    )

    # ========================================================
    # STEP 2
    # Add downstream prediction influence
    # ========================================================

    if predicted_congestion >= 3:

        for direction in DIRECTIONS:

            scores[direction] += 10

    elif predicted_congestion == 2:

        for direction in DIRECTIONS:

            scores[direction] += 5

    elif predicted_congestion == 1:

        for direction in DIRECTIONS:

            scores[direction] += 2

    # ========================================================
    # STEP 3
    # Add corridor propagation risk
    # ========================================================

    if propagation_risk >= 75:

        for direction in DIRECTIONS:

            scores[direction] += 10

    elif propagation_risk >= 50:

        for direction in DIRECTIONS:

            scores[direction] += 6

    elif propagation_risk >= 30:

        for direction in DIRECTIONS:

            scores[direction] += 3

    # ========================================================
    # STEP 4
    # Find highest priority direction
    # ========================================================

    selected_direction = max(
        scores,
        key=scores.get
    )

    selected_score = scores[
        selected_direction
    ]

    current_score = scores.get(
        current_direction,
        0
    )

    # Difference between best direction
    # and currently green direction.

    score_difference = (
        selected_score
        - current_score
    )

    # ========================================================
    # STEP 5
    # DECISION LOGIC
    # ========================================================

    # --------------------------------------------------------
    # CASE A
    # Different direction has clearly higher demand
    # --------------------------------------------------------

    if (
        selected_direction
        != current_direction
        and score_difference >= 8
    ):

        action = "SWITCH GREEN"

        green_time = int(
            clamp(
                30
                + (
                    selected_score
                    * 0.35
                ),
                MIN_GREEN_TIME,
                MAX_GREEN_TIME,
            )
        )

        reason = (
            f"{selected_direction} has the highest "
            f"combined traffic demand, queue pressure "
            f"and queue-growth priority."
        )

    # --------------------------------------------------------
    # CASE B
    # Difference is small
    # --------------------------------------------------------

    elif (
        selected_direction
        != current_direction
    ):

        selected_direction = (
            current_direction
        )

        action = "MAINTAIN"

        green_time = int(
            clamp(
                current_green,
                MIN_GREEN_TIME,
                MAX_GREEN_TIME,
            )
        )

        reason = (
            "Traffic demand is relatively balanced. "
            "Maintaining the current signal avoids "
            "unnecessary switching."
        )

    # --------------------------------------------------------
    # CASE C
    # Current direction is highest priority
    # --------------------------------------------------------

    else:

        if selected_score >= 70:

            action = "EXTEND GREEN"

            green_time = int(
                clamp(
                    current_green + 15,
                    MIN_GREEN_TIME,
                    MAX_GREEN_TIME,
                )
            )

            reason = (
                f"{current_direction} has very high "
                f"traffic pressure. Green time should "
                f"be extended."
            )

        elif selected_score >= 45:

            action = "MAINTAIN"

            green_time = int(
                clamp(
                    current_green,
                    MIN_GREEN_TIME,
                    MAX_GREEN_TIME,
                )
            )

            reason = (
                "Current green direction has sufficient "
                "traffic demand. Maintain the current timing."
            )

        else:

            action = "NORMALIZE"

            green_time = 30

            reason = (
                "Traffic demand is relatively low. "
                "Use normal signal timing."
            )

    # ========================================================
    # STEP 6
    # Corridor risk override
    # ========================================================

    if (
        propagation_risk >= 75
        and predicted_congestion >= 2
    ):

        green_time = int(
            clamp(
                green_time + 10,
                MIN_GREEN_TIME,
                MAX_GREEN_TIME,
            )
        )

        if action == "MAINTAIN":

            action = "EXTEND GREEN"

        reason += (
            " High downstream congestion and "
            "propagation risk require additional "
            "green capacity."
        )

    # ========================================================
    # STEP 7
    # Calculate timing adjustment
    # ========================================================

    adjustment = (
        green_time
        - current_green
    )

    return SignalRecommendation(

        selected_direction=
            selected_direction,

        action=
            action,

        green_time=
            green_time,

        previous_green_time=
            current_green,

        adjustment=
            adjustment,

        reason=
            reason,

        score=
            round(
                selected_score,
                2
            ),
    )


# ============================================================
# CONVERT RESULT TO DICTIONARY
# ============================================================

def recommendation_to_dict(
    recommendation,
):

    return {

        "selected_direction":
            recommendation.selected_direction,

        "action":
            recommendation.action,

        "green_time":
            recommendation.green_time,

        "previous_green_time":
            recommendation.previous_green_time,

        "adjustment":
            recommendation.adjustment,

        "reason":
            recommendation.reason,

        "score":
            recommendation.score,
    }


# ============================================================
# STANDALONE TEST
# ============================================================

if __name__ == "__main__":

    print("=" * 65)

    print(
        "TRAFLOW ADAPTIVE SIGNAL ENGINE TEST"
    )

    print("=" * 65)

    # --------------------------------------------------------
    # Scenario
    # --------------------------------------------------------
    #
    # North is heavily congested.
    #
    # East currently has green.
    #
    # This is intentionally an inefficient
    # signal allocation.
    # --------------------------------------------------------

    traffic = {

        "North": 65,

        "East": 12,

        "South": 15,

        "West": 10,
    }

    queues = {

        "North": 96,

        "East": 0,

        "South": 40,

        "West": 9,
    }

    queue_growth = {

        "North": 3,

        "East": 0,

        "South": 1,

        "West": 1,
    }

    current_direction = "East"

    current_green_time = 30

    predicted_congestion = 3

    propagation_risk = 75

    # --------------------------------------------------------
    # Print input
    # --------------------------------------------------------

    print(
        "\nCURRENT TRAFFIC"
    )

    for direction in DIRECTIONS:

        print(
            f"{direction:>6} : "
            f"{traffic[direction]}"
        )

    print(
        "\nCURRENT QUEUES"
    )

    for direction in DIRECTIONS:

        print(
            f"{direction:>6} : "
            f"{queues[direction]}"
        )

    print(
        "\nQUEUE GROWTH"
    )

    for direction in DIRECTIONS:

        print(
            f"{direction:>6} : "
            f"+{queue_growth[direction]}"
        )

    print(
        "\nCURRENT SIGNAL"
    )

    print(
        f"{current_direction} = GREEN"
    )

    print(
        f"Green time = "
        f"{current_green_time} seconds"
    )

    print(
        "\nPredicted downstream congestion:",
        predicted_congestion
    )

    print(
        "Propagation risk:",
        propagation_risk
    )

    # --------------------------------------------------------
    # Calculate base scores
    # --------------------------------------------------------

    scores = calculate_all_scores(

        traffic,

        queues,

        queue_growth,
    )

    print(
        "\nBASE DEMAND SCORES"
    )

    for direction in DIRECTIONS:

        print(
            f"{direction:>6} : "
            f"{scores[direction]}"
        )

    # --------------------------------------------------------
    # Generate recommendation
    # --------------------------------------------------------

    recommendation = (
        generate_signal_recommendation(

            traffic=traffic,

            queues=queues,

            queue_growth=queue_growth,

            current_green=current_green_time,

            predicted_congestion=
                predicted_congestion,

            propagation_risk=
                propagation_risk,

            current_direction=
                current_direction,
        )
    )

    # --------------------------------------------------------
    # Print result
    # --------------------------------------------------------

    print(
        "\n" + "=" * 65
    )

    print(
        "AI SIGNAL DECISION"
    )

    print(
        "=" * 65
    )

    print(
        "Action:",
        recommendation.action
    )

    print(
        "Selected direction:",
        recommendation.selected_direction
    )

    print(
        "Previous green:",
        recommendation.previous_green_time,
        "seconds"
    )

    print(
        "Recommended green:",
        recommendation.green_time,
        "seconds"
    )

    print(
        "Adjustment:",
        recommendation.adjustment,
        "seconds"
    )

    print(
        "Priority score:",
        recommendation.score
    )

    print(
        "\nReason:"
    )

    print(
        recommendation.reason
    )

    print(
        "\n" + "=" * 65
    )

    print(
        "SIGNAL ENGINE TEST COMPLETE"
    )

    print(
        "=" * 65
    )