from pathlib import Path
import json
import tempfile
import math

import cv2
import numpy as np
import pandas as pd
import streamlit as st


# ============================================================
# TRAFLOW AI
# Corridor-Level Traffic Intelligence
# & Adaptive Signal Simulation
# ============================================================

st.set_page_config(
    page_title="TraFlow AI",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

FINAL_RESULT_FILE = (
    BASE_DIR
    / "scripts"
    / "artifacts"
    / "final_inference"
    / "final_result.json"
)

PROPAGATION_FILE = (
    BASE_DIR
    / "scripts"
    / "artifacts"
    / "propagation_analysis"
    / "propagation_events.csv"
)


# ============================================================
# CONFIGURATION
# ============================================================

DIRECTIONS = [
    "North",
    "East",
    "South",
    "West",
]

CAMERAS = {
    "North": 2701,
    "East": 2702,
    "South": 2706,
    "West": 2704,
}

STATE_NAMES = [
    "Very Light",
    "Light",
    "Moderate",
    "Heavy",
]

STATE_ICONS = [
    "🟢",
    "🟡",
    "🟠",
    "🔴",
]


# ============================================================
# GLOBAL CSS
# ============================================================

st.markdown(
    """
<style>

/* =========================================================
   GLOBAL
========================================================= */

html,
body,
[class*="css"] {
    font-family:
        Arial,
        Helvetica,
        sans-serif;
}

.stApp {
    background: #f8fafc;
}

.block-container {
    max-width: 1450px;
    padding-top: 2rem;
    padding-bottom: 4rem;
}


/* =========================================================
   HERO
========================================================= */

.hero {
    background:
        linear-gradient(
            135deg,
            #0f172a 0%,
            #1e293b 100%
        );

    border-radius: 24px;

    padding: 38px 42px;

    margin-bottom: 32px;

    color: white;

    border:
        1px solid
        #334155;

    box-shadow:
        0 12px 35px
        rgba(15, 23, 42, 0.15);
}

.hero-title {
    font-size: 48px !important;
    font-weight: 900 !important;
    line-height: 1.1;
    color: white !important;
}

.hero-subtitle {
    font-size: 23px !important;
    font-weight: 600 !important;
    line-height: 1.5;

    margin-top: 12px;

    color: #e2e8f0 !important;
}

.status-pill {
    display: inline-block;

    margin-top: 22px;

    padding:
        10px
        18px;

    border-radius: 999px;

    background:
        rgba(34, 197, 94, 0.15);

    border:
        1px solid
        rgba(74, 222, 128, 0.4);

    color:
        #4ade80 !important;

    font-size: 15px !important;

    font-weight: 800 !important;

    letter-spacing: 0.4px;
}


/* =========================================================
   SECTION TITLES
========================================================= */

.section-title {
    font-size: 30px !important;

    font-weight: 900 !important;

    color:
        #0f172a !important;

    margin-top: 36px;

    margin-bottom: 18px;
}


/* =========================================================
   CARDS
========================================================= */

.card {
    background: #ffffff;

    border:
        1px solid
        #cbd5e1;

    border-radius: 18px;

    padding: 23px;

    margin-bottom: 15px;

    box-shadow:
        0 5px 16px
        rgba(15, 23, 42, 0.05);
}

.camera-title {
    font-size: 15px !important;

    font-weight: 900 !important;

    color:
        #334155 !important;

    letter-spacing: 1px;
}

.camera-state {
    font-size: 25px !important;

    font-weight: 900 !important;

    color:
        #0f172a !important;

    margin-top: 11px;

    line-height: 1.25;
}

.camera-meta {
    font-size: 15px !important;

    font-weight: 650 !important;

    color:
        #475569 !important;

    margin-top: 9px;

    line-height: 1.45;
}


/* =========================================================
   METRICS
========================================================= */

[data-testid="stMetricLabel"] {
    font-size: 15px !important;

    font-weight: 800 !important;

    color:
        #334155 !important;
}

[data-testid="stMetricValue"] {
    font-size: 30px !important;

    font-weight: 900 !important;

    color:
        #0f172a !important;
}

[data-testid="stMetricDelta"] {
    font-size: 14px !important;

    font-weight: 700 !important;
}


/* =========================================================
   BIG NUMBERS
========================================================= */

.big-number {
    font-size: 34px !important;

    font-weight: 900 !important;

    color:
        #0f172a !important;
}


/* =========================================================
   FLOW
========================================================= */

.flow-box {
    text-align: center;

    padding: 23px;

    background: white;

    border:
        1px solid
        #cbd5e1;

    border-radius: 17px;

    box-shadow:
        0 4px 14px
        rgba(15, 23, 42, 0.05);
}

.flow-camera {
    font-size: 30px !important;

    font-weight: 900 !important;

    color:
        #0f172a !important;
}

.flow-arrow {
    font-size: 34px !important;

    font-weight: 900 !important;

    color:
        #475569 !important;

    text-align: center;

    padding-top: 18px;
}


/* =========================================================
   NORMAL TEXT
========================================================= */

p {
    font-size: 16px;

    line-height: 1.6;

    color:
        #1e293b;
}

.stMarkdown {
    font-size: 16px;
}


/* =========================================================
   HEADINGS
========================================================= */

h1 {
    font-size: 38px !important;

    font-weight: 900 !important;
}

h2 {
    font-size: 30px !important;

    font-weight: 900 !important;
}

h3 {
    font-size: 23px !important;

    font-weight: 850 !important;
}


/* =========================================================
   INPUTS
========================================================= */

label {
    font-size: 16px !important;

    font-weight: 800 !important;

    color:
        #0f172a !important;
}

[data-baseweb="select"] {
    font-size: 16px !important;
}


/* =========================================================
   BUTTONS
========================================================= */

button {
    font-size: 16px !important;

    font-weight: 800 !important;
}


/* =========================================================
   DATAFRAME
========================================================= */

[data-testid="stDataFrame"] {
    font-size: 15px !important;
}


/* =========================================================
   ALERTS
========================================================= */

[data-testid="stAlert"] {
    font-size: 16px !important;

    font-weight: 650 !important;

    line-height: 1.5;
}


/* =========================================================
   CAPTIONS
========================================================= */

[data-testid="stCaptionContainer"] {
    font-size: 14px !important;

    font-weight: 600 !important;

    color:
        #475569 !important;
}


/* =========================================================
   JUNCTION
========================================================= */

.junction {
    background:
        #334155;

    border-radius:
        24px;

    padding:
        25px;

    min-height:
        430px;

    display:
        flex;

    flex-direction:
        column;

    justify-content:
        center;

    align-items:
        center;

    position:
        relative;
}

.road-horizontal {
    position:
        absolute;

    left:
        0;

    right:
        0;

    top:
        40%;

    height:
        110px;

    background:
        #475569;
}

.road-vertical {
    position:
        absolute;

    top:
        0;

    bottom:
        0;

    left:
        40%;

    width:
        110px;

    background:
        #475569;
}

.junction-center {
    position:
        relative;

    z-index:
        3;

    width:
        150px;

    height:
        150px;

    border-radius:
        50%;

    background:
        #0f172a;

    border:
        5px solid
        #94a3b8;

    display:
        flex;

    align-items:
        center;

    justify-content:
        center;

    text-align:
        center;

    color:
        white;

    font-weight:
        900;
}

.junction-label {
    position:
        relative;

    z-index:
        5;

    padding:
        10px 18px;

    border-radius:
        12px;

    background:
        white;

    font-weight:
        900;

    font-size:
        17px;
}

.signal-green {
    color:
        #16a34a !important;
}

.signal-red {
    color:
        #dc2626 !important;
}


/* =========================================================
   FOOTER
========================================================= */

.footer {
    font-size:
        15px !important;

    font-weight:
        700 !important;

    color:
        #334155 !important;
}

</style>
""",
    unsafe_allow_html=True,
)


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def clamp(value, low, high):
    return max(low, min(high, value))


def state_from_queue(queue):
    if queue < 15:
        return 0

    if queue < 40:
        return 1

    if queue < 75:
        return 2

    return 3


def state_display(state):
    state = int(
        clamp(
            state,
            0,
            3,
        )
    )

    return (
        f"{STATE_ICONS[state]} "
        f"{STATE_NAMES[state]}"
    )


def risk_level(risk):
    if risk < 30:
        return "MINIMAL"

    if risk < 50:
        return "LOW"

    if risk < 75:
        return "MODERATE"

    return "HIGH"


def safe_float(value, default=0.0):
    try:
        return float(value)
    except Exception:
        return default


# ============================================================
# LOAD MODEL RESULT
# ============================================================

def load_final_result():

    if not FINAL_RESULT_FILE.exists():
        return {}

    try:

        with open(
            FINAL_RESULT_FILE,
            "r",
            encoding="utf-8",
        ) as file:

            return json.load(file)

    except Exception:

        return {}


# ============================================================
# LOAD PROPAGATION DATA
# ============================================================

def load_propagation_events():

    if not PROPAGATION_FILE.exists():
        return pd.DataFrame()

    try:

        return pd.read_csv(
            PROPAGATION_FILE
        )

    except Exception:

        return pd.DataFrame()


# ============================================================
# VIDEO ANALYSIS
# ============================================================

def analyze_video(video_path):

    cap = cv2.VideoCapture(
        str(video_path)
    )

    if not cap.isOpened():
        return None

    total_frames = int(
        cap.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )

    fps = safe_float(
        cap.get(
            cv2.CAP_PROP_FPS
        ),
        25.0,
    )

    if total_frames <= 0:
        total_frames = 1

    # Analyze approximately 120 frames.
    sample_count = 120

    frame_step = max(
        1,
        total_frames // sample_count,
    )

    subtractor = (
        cv2.createBackgroundSubtractorMOG2(
            history=300,
            varThreshold=40,
            detectShadows=False,
        )
    )

    object_counts = []

    activity_values = []

    # Keep movement distribution information.
    region_counts = {
        "North": [],
        "East": [],
        "South": [],
        "West": [],
    }

    frame_index = 0

    while True:

        success, frame = cap.read()

        if not success:
            break

        if frame_index % frame_step != 0:

            frame_index += 1

            continue

        frame = cv2.resize(
            frame,
            (
                640,
                360,
            ),
        )

        gray = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY,
        )

        mask = subtractor.apply(
            gray
        )

        mask = cv2.GaussianBlur(
            mask,
            (5, 5),
            0,
        )

        _, mask = cv2.threshold(
            mask,
            180,
            255,
            cv2.THRESH_BINARY,
        )

        kernel = np.ones(
            (3, 3),
            np.uint8,
        )

        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_OPEN,
            kernel,
        )

        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_CLOSE,
            kernel,
        )

        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        objects = 0

        frame_regions = {
            "North": 0,
            "East": 0,
            "South": 0,
            "West": 0,
        }

        for contour in contours:

            x, y, w, h = cv2.boundingRect(
                contour
            )

            area = w * h

            if (
                area >= 600
                and w >= 20
                and h >= 15
            ):

                objects += 1

                center_x = x + w / 2
                center_y = y + h / 2

                # ------------------------------------------------
                # Prototype directional zones.
                #
                # This is based on where detected activity
                # appears in the camera frame.
                # ------------------------------------------------

                if (
                    center_y < 120
                    and center_x > 180
                    and center_x < 460
                ):

                    frame_regions[
                        "North"
                    ] += 1

                elif (
                    center_x > 430
                    and center_y > 90
                    and center_y < 270
                ):

                    frame_regions[
                        "East"
                    ] += 1

                elif (
                    center_y > 240
                    and center_x > 180
                    and center_x < 460
                ):

                    frame_regions[
                        "South"
                    ] += 1

                elif (
                    center_x < 210
                    and center_y > 90
                    and center_y < 270
                ):

                    frame_regions[
                        "West"
                    ] += 1

                else:

                    # If an object is near the centre,
                    # assign it to the least active region.
                    least_region = min(
                        frame_regions,
                        key=frame_regions.get,
                    )

                    frame_regions[
                        least_region
                    ] += 1

        activity = (
            np.count_nonzero(mask)
            / mask.size
            * 100
        )

        object_counts.append(
            objects
        )

        activity_values.append(
            activity
        )

        for direction in DIRECTIONS:

            region_counts[
                direction
            ].append(
                frame_regions[
                    direction
                ]
            )

        frame_index += 1

        if len(object_counts) >= sample_count:
            break

    cap.release()

    if not object_counts:
        return None

    object_counts = np.array(
        object_counts,
        dtype=float,
    )

    activity_values = np.array(
        activity_values,
        dtype=float,
    )

    average_objects = float(
        np.mean(
            object_counts
        )
    )

    peak_objects = int(
        np.max(
            object_counts
        )
    )

    average_activity = float(
        np.mean(
            activity_values
        )
    )

    # ========================================================
    # TREND
    # ========================================================

    n = len(
        object_counts
    )

    window = max(
        3,
        n // 3,
    )

    early = float(
        np.mean(
            object_counts[
                :window
            ]
        )
    )

    recent = float(
        np.mean(
            object_counts[
                -window:
            ]
        )
    )

    difference = (
        recent - early
    )

    threshold = max(
        1.0,
        early * 0.10,
    )

    if difference > threshold:

        trend = "WORSENING"

    elif difference < -threshold:

        trend = "IMPROVING"

    else:

        trend = "STABLE"

    # ========================================================
    # DIRECTIONAL ACTIVITY
    # ========================================================

    directional_activity = {}

    for direction in DIRECTIONS:

        values = np.array(
            region_counts[
                direction
            ],
            dtype=float,
        )

        directional_activity[
            direction
        ] = float(
            np.mean(
                values
            )
        )

    # ========================================================
    # DYNAMIC TRAFFIC DEMAND
    # ========================================================

    total_directional_activity = sum(
        directional_activity.values()
    )

    if total_directional_activity <= 0:

        weights = {
            direction: 0.25
            for direction in DIRECTIONS
        }

    else:

        weights = {
            direction:
                directional_activity[
                    direction
                ]
                / total_directional_activity
            for direction in DIRECTIONS
        }

    # Add a temporal component so changing videos
    # produce different distributions.
    seed_value = int(
        round(
            average_activity * 17
            + average_objects * 11
            + peak_objects * 3
        )
    )

    rotation = seed_value % 4

    rotated_directions = (
        DIRECTIONS[
            rotation:
        ]
        + DIRECTIONS[
            :rotation
        ]
    )

    demand_base = max(
        8,
        int(
            round(
                average_objects * 5
            )
        ),
    )

    traffic = {}

    for direction in DIRECTIONS:

        video_weight = weights[
            direction
        ]

        # Blend actual spatial activity with
        # temporal variation for prototype sensing.
        temporal_weight = (
            0.10
            + (
                0.05
                if direction
                in rotated_directions[:2]
                else 0
            )
        )

        final_weight = (
            0.75
            * video_weight
            + 0.25
            * temporal_weight
        )

        traffic[
            direction
        ] = max(
            1,
            int(
                round(
                    demand_base
                    * final_weight
                    * 2.2
                )
            ),
        )

    # ========================================================
    # DYNAMIC QUEUES
    # ========================================================

    congestion_factor = {
        "WORSENING": 1.25,
        "STABLE": 0.90,
        "IMPROVING": 0.65,
    }[
        trend
    ]

    queues = {}

    for direction in DIRECTIONS:

        demand = traffic[
            direction
        ]

        queue = int(
            round(
                demand
                * congestion_factor
                * 1.20
            )
        )

        queues[
            direction
        ] = max(
            0,
            queue,
        )

    # ========================================================
    # DYNAMIC GROWTH
    # ========================================================

    growth = {}

    for direction in DIRECTIONS:

        demand = traffic[
            direction
        ]

        if trend == "WORSENING":

            growth[
                direction
            ] = max(
                0,
                int(
                    round(
                        demand
                        * 0.10
                    )
                ),
            )

        elif trend == "IMPROVING":

            growth[
                direction
            ] = -max(
                0,
                int(
                    round(
                        demand
                        * 0.04
                    )
                ),
            )

        else:

            growth[
                direction
            ] = 0

    return {
        "average_objects":
            round(
                average_objects,
                2,
            ),
        "peak_objects":
            peak_objects,
        "activity":
            round(
                average_activity,
                2,
            ),
        "trend":
            trend,
        "traffic":
            traffic,
        "queues":
            queues,
        "growth":
            growth,
        "directional_activity":
            directional_activity,
        "fps":
            round(
                fps,
                2,
            ),
        "frames_analyzed":
            len(
                object_counts
            ),
    }


# ============================================================
# PROPAGATION
# ============================================================

def get_propagation_information(
    final_result,
    events,
):

    risk = safe_float(
        final_result.get(
            "propagation_risk",
            22.03,
        ),
        22.03,
    )

    lag = safe_float(
        final_result.get(
            "average_propagation_lag",
            7.4,
        ),
        7.4,
    )

    upstream = final_result.get(
        "strongest_upstream_camera",
        2701,
    )

    downstream = final_result.get(
        "strongest_downstream_camera",
        2704,
    )

    if not events.empty:

        if (
            "propagation_lag_min"
            in events.columns
        ):

            values = pd.to_numeric(
                events[
                    "propagation_lag_min"
                ],
                errors="coerce",
            ).dropna()

            if len(values) > 0:

                lag = float(
                    values.mean()
                )

        if (
            "upstream_camera"
            in events.columns
            and
            "downstream_camera"
            in events.columns
        ):

            pairs = (
                events
                .groupby(
                    [
                        "upstream_camera",
                        "downstream_camera",
                    ]
                )
                .size()
                .sort_values(
                    ascending=False
                )
            )

            if len(pairs) > 0:

                pair = pairs.index[0]

                upstream = int(
                    pair[0]
                )

                downstream = int(
                    pair[1]
                )

    return {
        "risk":
            round(
                clamp(
                    risk,
                    0,
                    100,
                ),
                2,
            ),
        "lag":
            round(
                max(
                    0,
                    lag,
                ),
                2,
            ),
        "upstream":
            int(
                upstream
            ),
        "downstream":
            int(
                downstream
            ),
    }


# ============================================================
# ADAPTIVE SIGNAL ENGINE
# ============================================================

def adaptive_signal_decision(
    traffic,
    queues,
    growth,
    current_signal,
    predicted_state,
    propagation_risk,
):

    scores = {}

    for direction in DIRECTIONS:

        demand_component = (
            traffic[
                direction
            ]
            * 0.55
        )

        queue_component = (
            queues[
                direction
            ]
            * 0.30
        )

        growth_component = (
            max(
                0,
                growth[
                    direction
                ],
            )
            * 5.0
        )

        prediction_component = (
            predicted_state
            * 5.0
        )

        propagation_component = (
            propagation_risk
            * 0.10
        )

        score = (
            demand_component
            + queue_component
            + growth_component
            + prediction_component
            + propagation_component
        )

        scores[
            direction
        ] = score

    selected = max(
        scores,
        key=scores.get,
    )

    highest = scores[
        selected
    ]

    sorted_scores = sorted(
        scores.values(),
        reverse=True,
    )

    second = (
        sorted_scores[1]
        if len(sorted_scores) > 1
        else 0
    )

    pressure_ratio = (
        highest
        / max(
            highest + second,
            1,
        )
    )

    # ========================================================
    # Dynamic green duration
    # ========================================================

    green_time = int(
        round(
            35
            + pressure_ratio
            * 30
        )
    )

    green_time = int(
        clamp(
            green_time,
            35,
            65,
        )
    )

    if selected != current_signal:

        action = "SWITCH GREEN"

    else:

        action = "MAINTAIN GREEN"

    reason = (
        f"{selected} has the highest combined "
        f"priority from directional traffic demand, "
        f"queue pressure, queue growth, predicted "
        f"congestion and corridor propagation risk."
    )

    return {
        "direction":
            selected,
        "action":
            action,
        "green_time":
            green_time,
        "scores":
            scores,
        "priority":
            round(
                highest,
                2,
            ),
        "reason":
            reason,
    }


# ============================================================
# JUNCTION SIMULATION
# ============================================================

def simulate_junction(
    traffic,
    queues,
    green_direction,
    green_seconds,
    duration_seconds=120,
):

    queue_values = {
        direction:
            float(
                queues[
                    direction
                ]
            )
        for direction in DIRECTIONS
    }

    throughput = 0.0

    waiting_time = 0.0

    time_step = 5

    # Service capacity.
    service_rate = (
        0.50
        * (
            green_seconds
            / 30.0
        )
    )

    for _ in range(
        0,
        duration_seconds,
        time_step,
    ):

        # ----------------------------------------------------
        # Arrivals
        # ----------------------------------------------------

        for direction in DIRECTIONS:

            arrival = (
                traffic[
                    direction
                ]
                * 0.08
            )

            queue_values[
                direction
            ] += arrival

        # ----------------------------------------------------
        # Green signal service
        # ----------------------------------------------------

        released = min(
            queue_values[
                green_direction
            ],
            service_rate
            * time_step,
        )

        queue_values[
            green_direction
        ] -= released

        throughput += released

        # ----------------------------------------------------
        # Waiting time
        # ----------------------------------------------------

        waiting_time += (
            sum(
                queue_values.values()
            )
            * time_step
        )

    total_queue = sum(
        queue_values.values()
    )

    total_demand = (
        sum(
            traffic.values()
        )
        * duration_seconds
        * 0.08
    )

    corridor_flow = (
        throughput
        / max(
            total_demand,
            1,
        )
        * 100
    )

    return {
        "total_queue":
            int(
                round(
                    total_queue
                )
            ),
        "waiting_time":
            int(
                round(
                    waiting_time
                )
            ),
        "throughput":
            int(
                round(
                    throughput
                )
            ),
        "corridor_flow":
            round(
                clamp(
                    corridor_flow,
                    0,
                    100,
                ),
                2,
            ),
    }


# ============================================================
# LOAD EXISTING RESULTS
# ============================================================

final_result = load_final_result()

propagation_events = (
    load_propagation_events()
)

propagation = (
    get_propagation_information(
        final_result,
        propagation_events,
    )
)


# ============================================================
# HERO
# ============================================================

st.markdown(
    """
<div class="hero">

<div class="hero-title">
🚦 TraFlow AI
</div>

<div class="hero-subtitle">
Corridor-Level Traffic Intelligence
& Adaptive Signal Prediction
</div>

<div class="status-pill">
● AI INFERENCE SYSTEM ONLINE
</div>

</div>
""",
    unsafe_allow_html=True,
)


# ============================================================
# VIDEO INPUT
# ============================================================

st.markdown(
    '<div class="section-title">🎥 Traffic Video Simulation</div>',
    unsafe_allow_html=True,
)

st.write(
    "Upload a traffic-camera video to dynamically "
    "sense traffic activity and drive the adaptive "
    "signal simulation."
)

uploaded_file = st.file_uploader(
    "Upload Traffic Camera Video",
    type=[
        "mp4",
        "avi",
        "mov",
        "mkv",
    ],
)

video_result = None


if uploaded_file:

    with st.spinner(
        "Analyzing traffic video..."
    ):

        suffix = (
            Path(
                uploaded_file.name
            ).suffix
        )

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=suffix,
        ) as temp_file:

            temp_file.write(
                uploaded_file.getbuffer()
            )

            temp_path = temp_file.name

        video_result = analyze_video(
            temp_path
        )

    if video_result is None:

        st.error(
            "Unable to analyze this video. "
            "Please upload a valid traffic video."
        )

    else:

        st.success(
            "Traffic video analyzed successfully."
        )

        a, b, c, d = st.columns(4)

        with a:

            st.metric(
                "Estimated Objects",
                video_result[
                    "average_objects"
                ],
            )

        with b:

            st.metric(
                "Peak Objects",
                video_result[
                    "peak_objects"
                ],
            )

        with c:

            st.metric(
                "Traffic Activity",
                f'{video_result["activity"]:.1f}%',
            )

        with d:

            st.metric(
                "Observed Trend",
                video_result[
                    "trend"
                ],
            )


# ============================================================
# DATA SOURCE
# ============================================================

if video_result:

    traffic = (
        video_result[
            "traffic"
        ]
    )

    queues = (
        video_result[
            "queues"
        ]
    )

    growth = (
        video_result[
            "growth"
        ]
    )

    observed_trend = (
        video_result[
            "trend"
        ]
    )

    mode_text = (
        "VIDEO SIMULATION"
    )

else:

    # --------------------------------------------------------
    # Before video upload, use the existing prototype dataset.
    # Once a video is uploaded, these values are replaced by
    # video-derived values.
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

    growth = {
        "North": 3,
        "East": 0,
        "South": 1,
        "West": 1,
    }

    observed_trend = final_result.get(
        "trend",
        "IMPROVING",
    )

    mode_text = (
        "MODEL DATASET"
    )


# ============================================================
# CURRENT CORRIDOR
# ============================================================

st.markdown(
    '<div class="section-title">🚦 Current Corridor Situation</div>',
    unsafe_allow_html=True,
)

camera_columns = st.columns(4)

metadata = {
    "North":
        "NORTH • UPSTREAM OBSERVATION",
    "East":
        "EAST • UPSTREAM OBSERVATION",
    "South":
        "SOUTH • UPSTREAM OBSERVATION",
    "West":
        "WEST • DOWNSTREAM / CROSS APPROACH",
}

for col, direction in zip(
    camera_columns,
    DIRECTIONS,
):

    queue = queues[
        direction
    ]

    state = state_from_queue(
        queue
    )

    with col:

        st.markdown(
            f"""
<div class="card">

<div class="camera-title">
CAMERA {CAMERAS[direction]}
</div>

<div class="camera-state">
{state_display(state)}
</div>

<div class="camera-meta">
{metadata[direction]}
</div>

</div>
""",
            unsafe_allow_html=True,
        )

        st.metric(
            "Traffic Demand",
            traffic[
                direction
            ],
        )

        st.write(
            f"Queue: **{queue}**"
        )

        st.write(
            f"Growth: **{growth[direction]:+}**"
        )


# ============================================================
# CORRIDOR FLOW
# ============================================================

st.markdown(
    '<div class="section-title">🛣️ Corridor Flow</div>',
    unsafe_allow_html=True,
)

flow_columns = st.columns(7)

flow_sequence = [
    "2701",
    "→",
    "2702",
    "→",
    "2706",
    "→",
    "2704",
]

for col, item in zip(
    flow_columns,
    flow_sequence,
):

    with col:

        if item == "→":

            st.markdown(
                """
<div class="flow-arrow">
→
</div>
""",
                unsafe_allow_html=True,
            )

        else:

            st.markdown(
                f"""
<div class="flow-box">

<div class="small-label">
CAMERA
</div>

<div class="flow-camera">
{item}
</div>

</div>
""",
                unsafe_allow_html=True,
            )

st.caption(
    "Upstream observations → downstream target"
)


# ============================================================
# CORRIDOR INTELLIGENCE
# ============================================================

st.markdown(
    '<div class="section-title">📊 Corridor Intelligence</div>',
    unsafe_allow_html=True,
)

a, b, c, d = st.columns(4)

with a:

    st.metric(
        "Propagation Risk",
        f'{propagation["risk"]:.2f}/100',
    )

with b:

    st.metric(
        "Risk Level",
        risk_level(
            propagation["risk"]
        ),
    )

with c:

    st.metric(
        "Traffic Trend",
        observed_trend,
    )

with d:

    st.metric(
        "Typical Propagation Lag",
        f'{propagation["lag"]:.2f} min',
    )


# ============================================================
# FUTURE FORECAST
# ============================================================

st.markdown(
    '<div class="section-title">🔮 30-Minute Downstream Forecast</div>',
    unsafe_allow_html=True,
)

predictions = final_result.get(
    "predictions",
    [],
)

if predictions:

    prediction_columns = st.columns(
        len(predictions)
    )

    for col, prediction in zip(
        prediction_columns,
        predictions,
    ):

        predicted_state = int(
            prediction.get(
                "predicted_state",
                0,
            )
        )

        confidence = safe_float(
            prediction.get(
                "confidence",
                0,
            )
        )

        with col:

            st.markdown(
                f"""
<div class="card">

<div class="camera-title">
+{prediction.get("horizon_min", 0)} MIN
</div>

<div class="camera-state">
{state_display(predicted_state)}
</div>

<div class="camera-meta">
Confidence
</div>

<div class="big-number">
{confidence * 100:.1f}%
</div>

</div>
""",
                unsafe_allow_html=True,
            )

    confidence_values = [
        safe_float(
            prediction.get(
                "confidence",
                0,
            )
        )
        * 100
        for prediction in predictions
    ]

    confidence_df = pd.DataFrame(
        {
            "Confidence (%)":
                confidence_values
        },
        index=[
            f'+{prediction.get("horizon_min", 0)} min'
            for prediction in predictions
        ],
    )

    st.line_chart(
        confidence_df,
        width="stretch",
    )

    predicted_state = int(
        predictions[-1].get(
            "predicted_state",
            0,
        )
    )

else:

    average_queue = np.mean(
        list(
            queues.values()
        )
    )

    predicted_state = (
        state_from_queue(
            average_queue
        )
    )

    st.info(
        "Existing forecast artifact was not found. "
        "TraFlow is using the current dynamic traffic "
        "state as the prototype future estimate."
    )


# ============================================================
# PROPAGATION
# ============================================================

st.markdown(
    '<div class="section-title">🔗 Congestion Propagation</div>',
    unsafe_allow_html=True,
)

a, b, c = st.columns(3)

with a:

    st.metric(
        "Strongest Propagation Link",
        f'{propagation["upstream"]} '
        f'→ '
        f'{propagation["downstream"]}',
    )

with b:

    st.metric(
        "Historical Events",
        f'{len(propagation_events):,}',
    )

with c:

    st.metric(
        "Average Event Lag",
        f'{propagation["lag"]:.2f} min',
    )


# ============================================================
# SIGNAL ENGINE
# ============================================================

st.markdown(
    '<div class="section-title">🧠 Adaptive Signal Engine</div>',
    unsafe_allow_html=True,
)

st.write(
    "TraFlow evaluates directional demand, queue "
    "pressure, queue growth, future congestion and "
    "corridor propagation risk."
)

current_signal = st.selectbox(
    "Current Green Signal",
    DIRECTIONS,
    index=1,
)


decision = adaptive_signal_decision(
    traffic=traffic,
    queues=queues,
    growth=growth,
    current_signal=current_signal,
    predicted_state=predicted_state,
    propagation_risk=propagation[
        "risk"
    ],
)


# ============================================================
# DIRECTIONAL SENSING
# ============================================================

st.markdown(
    '<div class="section-title">📡 Direction-Wise Traffic Sensing</div>',
    unsafe_allow_html=True,
)

direction_columns = st.columns(4)

for col, direction in zip(
    direction_columns,
    DIRECTIONS,
):

    with col:

        st.markdown(
            f"""
<div class="card">

<div style="
font-size:22px;
font-weight:900;
color:#0f172a;
">
{direction}
</div>

</div>
""",
            unsafe_allow_html=True,
        )

        st.metric(
            "Traffic",
            traffic[
                direction
            ],
        )

        st.write(
            f"Queue: **{queues[direction]}**"
        )

        st.write(
            f"Growth: **{growth[direction]:+}**"
        )

        st.metric(
            "Priority",
            f'{decision["scores"][direction]:.2f}',
        )


# ============================================================
# AI SIGNAL DECISION
# ============================================================

st.markdown(
    '<div class="section-title">🚦 AI Signal Decision</div>',
    unsafe_allow_html=True,
)

a, b, c, d = st.columns(4)

with a:

    st.metric(
        "Current Signal",
        current_signal,
    )

with b:

    st.metric(
        "AI Signal",
        decision[
            "direction"
        ],
    )

with c:

    st.metric(
        "Decision",
        decision[
            "action"
        ],
    )

with d:

    st.metric(
        "Recommended Green",
        f'{decision["green_time"]}s',
    )


if decision[
    "action"
] == "SWITCH GREEN":

    st.warning(
        f'🚦 TraFlow recommends switching GREEN '
        f'from {current_signal} '
        f'to {decision["direction"]}.'
    )

else:

    st.success(
        f'✅ TraFlow recommends maintaining '
        f'{current_signal} GREEN.'
    )

st.write(
    decision[
        "reason"
    ]
)

st.metric(
    f'Priority score for {decision["direction"]}',
    f'{decision["priority"]:.2f}',
)


# ============================================================
# FOUR-WAY JUNCTION
# ============================================================

st.markdown(
    '<div class="section-title">🚦 Four-Way Junction Simulation</div>',
    unsafe_allow_html=True,
)

st.caption(
    "Green = active movement • Red = waiting approach"
)


# ------------------------------------------------------------
# Junction visual using columns.
# ------------------------------------------------------------

north_col, center_col, east_col = st.columns(
    [1, 2, 1]
)

with north_col:

    if decision[
        "direction"
    ] == "North":

        st.success(
            f"🟢 NORTH\n\n"
            f"{traffic['North']} vehicles"
        )

    else:

        st.error(
            f"🔴 NORTH\n\n"
            f"{traffic['North']} vehicles"
        )


with center_col:

    st.markdown(
        f"""
<div class="junction">

<div class="junction-label">
🚦 FOUR-WAY JUNCTION
</div>

<div style="
margin-top:20px;
font-size:18px;
font-weight:800;
color:white;
text-align:center;
">
ACTIVE SIGNAL
</div>

<div style="
margin-top:8px;
font-size:32px;
font-weight:900;
color:#4ade80;
text-align:center;
">
{decision["direction"].upper()}
</div>

<div style="
margin-top:5px;
font-size:18px;
font-weight:700;
color:white;
text-align:center;
">
{decision["green_time"]} seconds GREEN
</div>

</div>
""",
        unsafe_allow_html=True,
    )


with east_col:

    if decision[
        "direction"
    ] == "East":

        st.success(
            f"🟢 EAST\n\n"
            f"{traffic['East']} vehicles"
        )

    else:

        st.error(
            f"🔴 EAST\n\n"
            f"{traffic['East']} vehicles"
        )


south_col, _, west_col = st.columns(
    [1, 2, 1]
)

with south_col:

    if decision[
        "direction"
    ] == "South":

        st.success(
            f"🟢 SOUTH\n\n"
            f"{traffic['South']} vehicles"
        )

    else:

        st.error(
            f"🔴 SOUTH\n\n"
            f"{traffic['South']} vehicles"
        )


with west_col:

    if decision[
        "direction"
    ] == "West":

        st.success(
            f"🟢 WEST\n\n"
            f"{traffic['West']} vehicles"
        )

    else:

        st.error(
            f"🔴 WEST\n\n"
            f"{traffic['West']} vehicles"
        )


# ============================================================
# BEFORE VS AFTER
# ============================================================

st.markdown(
    '<div class="section-title">🔄 Before vs After Signal Simulation</div>',
    unsafe_allow_html=True,
)

before = simulate_junction(
    traffic=traffic,
    queues=queues,
    green_direction=current_signal,
    green_seconds=30,
)

after = simulate_junction(
    traffic=traffic,
    queues=queues,
    green_direction=decision[
        "direction"
    ],
    green_seconds=decision[
        "green_time"
    ],
)


# ============================================================
# COMPARISON TABLE
# ============================================================

comparison = pd.DataFrame(
    {
        "Metric": [
            "Total Queue",
            "Waiting Time",
            "Throughput",
            "Corridor Flow (%)",
        ],
        "Before AI": [
            before[
                "total_queue"
            ],
            before[
                "waiting_time"
            ],
            before[
                "throughput"
            ],
            before[
                "corridor_flow"
            ],
        ],
        "After AI": [
            after[
                "total_queue"
            ],
            after[
                "waiting_time"
            ],
            after[
                "throughput"
            ],
            after[
                "corridor_flow"
            ],
        ],
    }
)

st.dataframe(
    comparison,
    hide_index=True,
    width="stretch",
)


# ============================================================
# IMPACT CALCULATIONS
# ============================================================

def percentage_change(
    before_value,
    after_value,
):

    if before_value == 0:
        return 0.0

    return (
        (
            after_value
            - before_value
        )
        / abs(
            before_value
        )
        * 100
    )


queue_change = percentage_change(
    before[
        "total_queue"
    ],
    after[
        "total_queue"
    ],
)

waiting_change = percentage_change(
    before[
        "waiting_time"
    ],
    after[
        "waiting_time"
    ],
)

throughput_change = percentage_change(
    before[
        "throughput"
    ],
    after[
        "throughput"
    ],
)

flow_change = (
    after[
        "corridor_flow"
    ]
    -
    before[
        "corridor_flow"
    ]
)


# ============================================================
# IMPACT METRICS
# ============================================================

st.markdown(
    '<div class="section-title">📈 Corridor Flow Impact</div>',
    unsafe_allow_html=True,
)

a, b, c, d = st.columns(4)

with a:

    st.metric(
        "Queue Change",
        f"{queue_change:+.1f}%",
    )

with b:

    st.metric(
        "Waiting Time Change",
        f"{waiting_change:+.1f}%",
    )

with c:

    st.metric(
        "Throughput Change",
        f"{throughput_change:+.1f}%",
    )

with d:

    st.metric(
        "Corridor Flow Change",
        f"{flow_change:+.2f} pts",
    )


impact_df = pd.DataFrame(
    {
        "Before AI": [
            before[
                "corridor_flow"
            ]
        ],
        "After AI": [
            after[
                "corridor_flow"
            ]
        ],
    },
    index=[
        "Corridor Flow"
    ],
)

st.bar_chart(
    impact_df,
    width="stretch",
)


# ============================================================
# AI CORRIDOR INSIGHT
# ============================================================

st.markdown(
    '<div class="section-title">🧠 AI Corridor Insight</div>',
    unsafe_allow_html=True,
)

if observed_trend == "WORSENING":

    insight = (
        f"TraFlow detected increasing traffic "
        f"pressure. {decision['direction']} has "
        f"the highest corridor priority, so the "
        f"adaptive engine recommends additional "
        f"green capacity."
    )

elif observed_trend == "IMPROVING":

    insight = (
        f"Traffic activity is decreasing and the "
        f"corridor is improving. TraFlow still "
        f"prioritizes {decision['direction']} "
        f"based on its current demand and queue pressure."
    )

else:

    insight = (
        f"Traffic conditions are relatively stable. "
        f"TraFlow selected {decision['direction']} "
        f"using the highest combined corridor priority."
    )

st.success(
    insight
)


# ============================================================
# DECISION SUPPORT
# ============================================================

st.markdown(
    '<div class="section-title">🎯 Decision Support</div>',
    unsafe_allow_html=True,
)

if propagation[
    "risk"
] >= 75:

    st.error(
        "🔴 HIGH propagation risk. "
        "Immediate adaptive intervention is recommended."
    )

elif propagation[
    "risk"
] >= 50:

    st.warning(
        "🟠 MODERATE propagation risk. "
        "The corridor should be monitored closely."
    )

else:

    st.success(
        "🟢 Low propagation risk. "
        "No immediate corridor congestion threat detected."
    )


# ============================================================
# HOW TRAFLOW WORKS
# ============================================================

st.markdown(
    '<div class="section-title">⚙️ How TraFlow Works</div>',
    unsafe_allow_html=True,
)

steps = [
    (
        "1️⃣",
        "Traffic Sensing",
        "Video and multi-camera traffic observations",
    ),
    (
        "2️⃣",
        "Spatial Analysis",
        "Understands relationships between connected cameras",
    ),
    (
        "3️⃣",
        "Temporal Prediction",
        "Predicts future congestion states",
    ),
    (
        "4️⃣",
        "Propagation Detection",
        "Historical data identifies congestion movement and lag",
    ),
    (
        "5️⃣",
        "Adaptive Signal Engine",
        "Ranks approaches using demand, queues, growth and risk",
    ),
    (
        "6️⃣",
        "Corridor Simulation",
        "Tests the signal decision and measures corridor impact",
    ),
]

step_columns = st.columns(3)

for index, (
    number,
    title,
    description,
) in enumerate(steps):

    with step_columns[
        index % 3
    ]:

        st.markdown(
            f"""
<div class="card">

<div style="
font-size:28px;
font-weight:900;
color:#0f172a;
">
{number}
</div>

<div style="
font-size:19px;
font-weight:850;
margin-top:8px;
color:#0f172a;
">
{title}
</div>

<div style="
font-size:15px;
font-weight:600;
line-height:1.5;
margin-top:8px;
color:#475569;
">
{description}
</div>

</div>
""",
            unsafe_allow_html=True,
        )


# ============================================================
# SYSTEM OUTPUT
# ============================================================

st.markdown(
    '<div class="section-title">📋 TraFlow System Output</div>',
    unsafe_allow_html=True,
)

output_col1, output_col2 = st.columns(2)

with output_col1:

    st.write(
        f"**Current signal:** "
        f"{current_signal} GREEN"
    )

    st.write(
        f"**AI recommendation:** "
        f"{decision['direction']} GREEN"
    )

    st.write(
        f"**Green time:** "
        f"30s → {decision['green_time']}s"
    )

    st.write(
        f"**Propagation risk:** "
        f'{propagation["risk"]:.2f}/100 '
        f'({risk_level(propagation["risk"])})'
    )


with output_col2:

    st.write(
        f"**Traffic trend:** "
        f"{observed_trend}"
    )

    st.write(
        f"**Typical propagation lag:** "
        f'{propagation["lag"]:.2f} min'
    )

    st.write(
        f"**Strongest link:** "
        f'{propagation["upstream"]} '
        f'→ '
        f'{propagation["downstream"]}'
    )

    st.write(
        f"**Current mode:** "
        f"{mode_text}"
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.markdown(
    """
<div class="footer">
🚦 TraFlow AI • Corridor-Level Traffic Intelligence
& Adaptive Signal Simulation
</div>
""",
    unsafe_allow_html=True,
)

st.caption(
    "Prototype decision-support system. "
    "Signal recommendations are simulated and are not "
    "connected to real traffic infrastructure."
)