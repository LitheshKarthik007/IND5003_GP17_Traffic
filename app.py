"""
TraFlow AI
Corridor-Level Traffic Intelligence & Adaptive Signal Simulation

Dashboard:
1. Traffic video input
2. Video activity analysis
3. Current corridor state
4. Corridor flow
5. Propagation intelligence
6. LSTM forecast from final_result.json
7. Four-way junction simulation
8. Adaptive signal decision
9. Before/after corridor impact
"""

from pathlib import Path
import json
import tempfile
import math

import cv2
import numpy as np
import pandas as pd
import streamlit as st


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="TraFlow AI",
    page_icon="🚦",
    layout="wide",
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

FINAL_RESULT_PATH = (
    BASE_DIR
    / "scripts"
    / "artifacts"
    / "final_inference"
    / "final_result.json"
)

PROPAGATION_PATH = (
    BASE_DIR
    / "scripts"
    / "artifacts"
    / "propagation_analysis"
    / "propagation_events.csv"
)


# ============================================================
# CONSTANTS
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

STATE_NAMES = {
    0: "Very Light",
    1: "Light",
    2: "Moderate",
    3: "Severe",
}

STATE_EMOJI = {
    "Very Light": "🟢",
    "Light": "🟡",
    "Moderate": "🟠",
    "Severe": "🔴",
}


# ============================================================
# DEFAULT SCENARIO
# ============================================================

DEFAULT_TRAFFIC = {
    "North": 65,
    "East": 12,
    "South": 15,
    "West": 10,
}

DEFAULT_QUEUES = {
    "North": 96,
    "East": 0,
    "South": 40,
    "West": 9,
}

DEFAULT_GROWTH = {
    "North": 3,
    "East": 0,
    "South": 1,
    "West": 1,
}


# ============================================================
# CSS
#
# CSS is kept separate from the actual UI.
# No HTML is used for camera cards or metrics.
# ============================================================

st.markdown(
    """
<style>

.main-title {
    font-size: 3rem;
    font-weight: 900;
    margin-bottom: 0.2rem;
}

.subtitle {
    font-size: 1.15rem;
    opacity: 0.75;
    margin-bottom: 1rem;
}

.small-muted {
    opacity: 0.65;
    font-size: 0.85rem;
}

.signal-box {
    text-align: center;
    padding: 16px;
    border-radius: 18px;
    border: 1px solid rgba(128,128,128,0.35);
}

.junction-road {
    background: #3b4252;
    border-radius: 20px;
    padding: 25px;
    text-align: center;
}

.big-signal {
    font-size: 3rem;
    line-height: 1.1;
}

.decision-box {
    padding: 20px;
    border-radius: 18px;
    border: 1px solid rgba(128,128,128,0.35);
}

</style>
""",
    unsafe_allow_html=True,
)


# ============================================================
# FUNCTIONS
# ============================================================

def load_final_result():

    if not FINAL_RESULT_PATH.exists():
        return {}

    try:
        with open(
            FINAL_RESULT_PATH,
            "r",
            encoding="utf-8",
        ) as f:
            return json.load(f)

    except Exception:
        return {}


def load_propagation():

    if not PROPAGATION_PATH.exists():
        return pd.DataFrame()

    try:
        return pd.read_csv(
            PROPAGATION_PATH
        )

    except Exception:
        return pd.DataFrame()


def safe_float(value, default=0.0):

    try:
        return float(value)
    except Exception:
        return default


def state_from_activity(activity):

    if activity < 20:
        return 0

    if activity < 40:
        return 1

    if activity < 65:
        return 2

    return 3


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

    if total_frames <= 0:
        cap.release()
        return None

    sample_count = min(
        total_frames,
        180
    )

    step = max(
        1,
        total_frames // sample_count
    )

    previous = None
    activities = []

    frame_number = 0

    while True:

        ok, frame = cap.read()

        if not ok:
            break

        if frame_number % step != 0:

            frame_number += 1
            continue

        frame = cv2.resize(
            frame,
            (640, 360)
        )

        gray = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY
        )

        gray = cv2.GaussianBlur(
            gray,
            (5, 5),
            0
        )

        if previous is not None:

            difference = cv2.absdiff(
                previous,
                gray
            )

            _, threshold = cv2.threshold(
                difference,
                25,
                255,
                cv2.THRESH_BINARY
            )

            activity = (
                np.count_nonzero(
                    threshold
                )
                /
                threshold.size
            ) * 100

            activities.append(
                activity
            )

        previous = gray
        frame_number += 1

        if len(activities) >= 180:
            break

    cap.release()

    if not activities:
        return None

    values = np.array(
        activities,
        dtype=float
    )

    average = float(
        np.mean(values)
    )

    peak = float(
        np.percentile(
            values,
            95
        )
    )

    third = max(
        1,
        len(values) // 3
    )

    beginning = float(
        np.mean(
            values[:third]
        )
    )

    ending = float(
        np.mean(
            values[-third:]
        )
    )

    difference = ending - beginning

    if difference > 2:
        trend = "WORSENING"

    elif difference < -2:
        trend = "IMPROVING"

    else:
        trend = "STABLE"

    estimated_objects = max(
        1,
        int(round(average / 5))
    )

    peak_objects = max(
        estimated_objects,
        int(round(peak / 3))
    )

    return {
        "activity": min(
            average,
            100
        ),
        "peak_activity": min(
            peak,
            100
        ),
        "estimated_objects":
            estimated_objects,
        "peak_objects":
            peak_objects,
        "trend":
            trend,
    }


def traffic_from_video(video_result):

    if not video_result:
        return DEFAULT_TRAFFIC.copy()

    activity = video_result[
        "activity"
    ]

    scale = np.clip(
        activity / 45.0,
        0.5,
        2.0
    )

    traffic = {
        "North": int(
            DEFAULT_TRAFFIC["North"]
            * scale
        ),

        "East": int(
            DEFAULT_TRAFFIC["East"]
            * scale
        ),

        "South": int(
            DEFAULT_TRAFFIC["South"]
            * scale
        ),

        "West": int(
            DEFAULT_TRAFFIC["West"]
            * scale
        ),
    }

    return traffic


def calculate_priority(
    traffic,
    queues,
    growth,
    predicted_congestion,
    propagation_risk,
):

    scores = {}

    for direction in DIRECTIONS:

        demand = traffic[
            direction
        ]

        queue = queues[
            direction
        ]

        queue_growth = growth[
            direction
        ]

        score = (
            demand * 0.55
            +
            queue * 0.25
            +
            queue_growth * 4
            +
            predicted_congestion * 3
            +
            propagation_risk * 0.10
        )

        scores[direction] = score

    return scores


def calculate_green_time(
    score,
    maximum_score,
):

    if maximum_score <= 0:
        return 30

    ratio = (
        score
        /
        maximum_score
    )

    green = (
        35
        +
        ratio * 25
    )

    return int(
        np.clip(
            round(green),
            30,
            60
        )
    )


def simulate(
    traffic,
    current_signal,
    green_time,
    duration,
):

    queues = {}

    for direction in DIRECTIONS:

        base_queue = (
            DEFAULT_QUEUES[direction]
        )

        demand = traffic[
            direction
        ]

        queues[direction] = max(
            0,
            int(
                base_queue
                +
                demand * 0.25
                -
                8
            )
        )

    active_demand = traffic[
        current_signal
    ]

    total_demand = sum(
        traffic.values()
    )

    capacity = (
        green_time
        / 60
        *
        110
    )

    throughput = min(
        total_demand,
        max(
            active_demand,
            capacity
        )
    )

    if current_signal in queues:

        released = int(
            capacity * 0.55
        )

        queues[
            current_signal
        ] = max(
            0,
            queues[
                current_signal
            ]
            - released
        )

    total_queue = sum(
        queues.values()
    )

    waiting_time = int(
        total_queue
        *
        max(
            1,
            duration // 2
        )
    )

    corridor_flow = (
        throughput
        /
        max(
            total_demand,
            1
        )
        *
        100
    )

    return {
        "queues": queues,
        "total_queue":
            total_queue,
        "waiting_time":
            waiting_time,
        "throughput":
            int(throughput),
        "corridor_flow":
            round(
                corridor_flow,
                2
            ),
    }


def improvement(before, after):

    if before == 0:
        return 0

    return (
        (after - before)
        /
        abs(before)
        *
        100
    )


# ============================================================
# LOAD REAL TRAFLOW OUTPUT
# ============================================================

result = load_final_result()
events = load_propagation()


# ============================================================
# HERO
# ============================================================

st.markdown(
    '<div class="main-title">🚦 TraFlow AI</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="subtitle">'
    'Corridor-Level Traffic Intelligence '
    '& Adaptive Signal Prediction'
    '</div>',
    unsafe_allow_html=True,
)

st.success(
    "● AI INFERENCE SYSTEM ONLINE"
)


# ============================================================
# VIDEO
# ============================================================

st.header(
    "🎥 Traffic Video Simulation"
)

st.write(
    "Upload a traffic-camera video to "
    "simulate real-time traffic sensing."
)

uploaded_video = st.file_uploader(
    "Upload traffic video",
    type=[
        "mp4",
        "avi",
        "mov",
        "mkv",
    ],
)

video_result = None


if uploaded_video:

    st.video(
        uploaded_video
    )

    extension = Path(
        uploaded_video.name
    ).suffix

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=extension,
    ) as temp:

        temp.write(
            uploaded_video.getbuffer()
        )

        temp_path = temp.name

    with st.spinner(
        "Analyzing traffic activity..."
    ):

        video_result = analyze_video(
            temp_path
        )

    if video_result:

        st.success(
            "Traffic video analyzed successfully."
        )

        v1, v2, v3, v4 = st.columns(4)

        with v1:

            st.metric(
                "Estimated Objects",
                video_result[
                    "estimated_objects"
                ],
            )

        with v2:

            st.metric(
                "Peak Objects",
                video_result[
                    "peak_objects"
                ],
            )

        with v3:

            st.metric(
                "Traffic Activity",
                f"{video_result['activity']:.1f}%",
            )

        with v4:

            st.metric(
                "Observed Trend",
                video_result[
                    "trend"
                ],
            )

        st.caption(
            "OpenCV foreground-motion analysis "
            "is used as the video sensing prototype."
        )


# ============================================================
# TRAFFIC INPUT
# ============================================================

traffic = traffic_from_video(
    video_result
)


# ============================================================
# CURRENT CORRIDOR
# ============================================================

st.header(
    "🚦 Current Corridor Situation"
)

current_states = result.get(
    "current_states",
    {}
)

camera_columns = st.columns(4)

for column, direction in zip(
    camera_columns,
    DIRECTIONS
):

    camera = CAMERAS[
        direction
    ]

    state = current_states.get(
        str(camera),
        "Light"
    )

    with column:

        st.subheader(
            f"CAMERA {camera}"
        )

        st.markdown(
            f"### "
            f"{STATE_EMOJI.get(state, '⚪')} "
            f"{state}"
        )

        if direction == "West":

            st.caption(
                "DOWNSTREAM TARGET"
            )

        else:

            st.caption(
                f"{direction.upper()} • "
                "UPSTREAM OBSERVATION"
            )


# ============================================================
# CORRIDOR FLOW
# ============================================================

st.header(
    "🛣️ Corridor Flow"
)

flow_columns = st.columns(7)

flow_items = [
    "2701",
    "→",
    "2702",
    "→",
    "2706",
    "→",
    "2704",
]

for column, item in zip(
    flow_columns,
    flow_items
):

    with column:

        if item == "→":

            st.markdown(
                "## →"
            )

        else:

            st.metric(
                "Camera",
                item
            )

st.caption(
    "Upstream observations → downstream target"
)


# ============================================================
# PROPAGATION INTELLIGENCE
# ============================================================

risk = safe_float(
    result.get(
        "propagation_risk",
        22.03
    ),
    22.03,
)

risk_level = (
    "HIGH"
    if risk >= 75
    else
    "MODERATE"
    if risk >= 50
    else
    "LOW"
    if risk >= 30
    else
    "MINIMAL"
)

trend = result.get(
    "trend",
    "IMPROVING"
)

average_lag = safe_float(
    result.get(
        "average_propagation_lag",
        7.4
    ),
    7.4,
)

st.header(
    "📊 Corridor Intelligence"
)

c1, c2, c3, c4 = st.columns(4)

with c1:

    st.metric(
        "Propagation Risk",
        f"{risk:.2f}/100"
    )

with c2:

    st.metric(
        "Risk Level",
        risk_level
    )

with c3:

    st.metric(
        "Traffic Trend",
        trend
    )

with c4:

    st.metric(
        "Typical Propagation Lag",
        f"{average_lag:.1f} min"
    )


# ============================================================
# FORECAST
# ============================================================

st.header(
    "🔮 30-Minute Downstream Forecast"
)

predictions = result.get(
    "predictions",
    []
)

if predictions:

    forecast_columns = st.columns(
        len(predictions)
    )

    confidence_values = []

    for column, prediction in zip(
        forecast_columns,
        predictions
    ):

        horizon = prediction.get(
            "horizon_min",
            0
        )

        state_name = prediction.get(
            "predicted_state_name",
            "Very Light"
        )

        confidence = (
            safe_float(
                prediction.get(
                    "confidence",
                    0
                )
            )
            * 100
        )

        confidence_values.append(
            confidence
        )

        with column:

            st.subheader(
                f"+{horizon} min"
            )

            st.markdown(
                f"## "
                f"{STATE_EMOJI.get(state_name, '⚪')}"
            )

            st.write(
                f"**{state_name}**"
            )

            st.metric(
                "Confidence",
                f"{confidence:.1f}%"
            )

else:

    st.info(
        "No forecast data found in "
        "final_result.json."
    )

    confidence_values = []


# ============================================================
# CONFIDENCE CHART
# ============================================================

if predictions:

    confidence_df = pd.DataFrame(
        {
            "Horizon (min)": [
                p["horizon_min"]
                for p in predictions
            ],

            "Confidence (%)": [
                safe_float(
                    p["confidence"]
                ) * 100
                for p in predictions
            ],
        }
    )

    st.subheader(
        "📈 Forecast Confidence"
    )

    st.line_chart(
        confidence_df.set_index(
            "Horizon (min)"
        ),
        width="stretch",
    )


# ============================================================
# PROPAGATION
# ============================================================

st.header(
    "🔗 Congestion Propagation"
)

strongest_upstream = result.get(
    "strongest_upstream_camera",
    2701
)

target_camera = result.get(
    "strongest_downstream_camera",
    result.get(
        "target_camera",
        2704
    )
)

event_count = len(
    events
)

p1, p2, p3 = st.columns(3)

with p1:

    st.metric(
        "Strongest Link",
        f"{strongest_upstream} → {target_camera}"
    )

with p2:

    st.metric(
        "Historical Events",
        f"{event_count:,}"
    )

with p3:

    st.metric(
        "Average Event Lag",
        f"{average_lag:.2f} min"
    )


if not events.empty:

    link_counts = (
        events
        .groupby(
            [
                "upstream_camera",
                "downstream_camera",
            ]
        )
        .size()
        .reset_index(
            name="events"
        )
    )

    link_counts["Link"] = (
        link_counts[
            "upstream_camera"
        ].astype(str)
        + " → "
        + link_counts[
            "downstream_camera"
        ].astype(str)
    )

    link_counts = (
        link_counts
        .set_index("Link")
        [["events"]]
    )

    st.bar_chart(
        link_counts,
        width="stretch",
    )


# ============================================================
# ADAPTIVE SIGNAL ENGINE
# ============================================================

st.header(
    "🧠 Adaptive Signal Engine"
)

st.write(
    "TraFlow evaluates directional demand, "
    "queue pressure, queue growth, predicted "
    "downstream congestion and propagation risk."
)


# Current signal default from your scenario
current_signal = st.selectbox(
    "Current Green Signal",
    DIRECTIONS,
    index=1,
)


# Current queue and growth
queues = DEFAULT_QUEUES.copy()
growth = DEFAULT_GROWTH.copy()


# Predicted downstream state
predicted_congestion = 0

if predictions:

    predicted_congestion = int(
        predictions[-1].get(
            "predicted_state",
            0
        )
    )


# Calculate AI scores
scores = calculate_priority(
    traffic,
    queues,
    growth,
    predicted_congestion,
    risk,
)


recommended_direction = max(
    scores,
    key=scores.get
)

maximum_score = max(
    scores.values()
)

recommended_green = calculate_green_time(
    scores[
        recommended_direction
    ],
    maximum_score,
)


if (
    recommended_direction
    != current_signal
):

    action = "SWITCH GREEN"

else:

    action = "MAINTAIN GREEN"


# ============================================================
# DIRECTIONAL TRAFFIC
# ============================================================

st.subheader(
    "📡 Direction-Wise Traffic Sensing"
)

traffic_columns = st.columns(4)

for column, direction in zip(
    traffic_columns,
    DIRECTIONS
):

    with column:

        st.metric(
            direction,
            traffic[direction]
        )

        st.write(
            f"Queue: "
            f"**{queues[direction]}**"
        )

        st.write(
            f"Growth: "
            f"**{growth[direction]:+}**"
        )

        st.write(
            f"Priority: "
            f"**{scores[direction]:.2f}**"
        )


# ============================================================
# AI DECISION
# ============================================================

st.subheader(
    "🚦 AI Signal Decision"
)

d1, d2, d3, d4 = st.columns(4)

with d1:

    st.metric(
        "Current Signal",
        current_signal
    )

with d2:

    st.metric(
        "AI Signal",
        recommended_direction
    )

with d3:

    st.metric(
        "Decision",
        action
    )

with d4:

    st.metric(
        "Recommended Green",
        f"{recommended_green}s"
    )


if action == "SWITCH GREEN":

    st.warning(
        f"🚦 TraFlow recommends "
        f"switching GREEN from "
        f"{current_signal} to "
        f"{recommended_direction}."
    )

else:

    st.success(
        f"✅ TraFlow recommends "
        f"maintaining "
        f"{current_signal} GREEN."
    )


st.info(
    f"Priority score for "
    f"{recommended_direction}: "
    f"{scores[recommended_direction]:.2f}"
)


# ============================================================
# FOUR-WAY JUNCTION
# ============================================================

st.header(
    "🚦 Four-Way Junction Simulation"
)

st.caption(
    "Green = active movement • "
    "Red = waiting approach"
)


# Junction display
north_color = (
    "🟢"
    if recommended_direction == "North"
    else "🔴"
)

east_color = (
    "🟢"
    if recommended_direction == "East"
    else "🔴"
)

south_color = (
    "🟢"
    if recommended_direction == "South"
    else "🔴"
)

west_color = (
    "🟢"
    if recommended_direction == "West"
    else "🔴"
)


row1 = st.columns(
    [1, 2, 1]
)

with row1[1]:

    st.markdown(
        f"### "
        f"{north_color} NORTH"
    )

    st.write(
        f"🚗 × {traffic['North']}"
    )


row2 = st.columns(
    [2, 1, 2]
)

with row2[0]:

    st.markdown(
        f"### "
        f"{west_color} WEST"
    )

    st.write(
        f"🚗 × {traffic['West']}"
    )


with row2[1]:

    st.markdown(
        "## 🚦"
    )

    st.write(
        f"**{recommended_direction}**"
    )

    st.caption(
        f"{recommended_green}s GREEN"
    )


with row2[2]:

    st.markdown(
        f"### "
        f"{east_color} EAST"
    )

    st.write(
        f"🚗 × {traffic['East']}"
    )


row3 = st.columns(
    [1, 2, 1]
)

with row3[1]:

    st.markdown(
        f"### "
        f"{south_color} SOUTH"
    )

    st.write(
        f"🚗 × {traffic['South']}"
    )


# ============================================================
# SIMULATION
# ============================================================

st.header(
    "🔄 Before vs After Signal Simulation"
)

before = simulate(
    traffic,
    current_signal,
    30,
    30,
)

after = simulate(
    traffic,
    recommended_direction,
    recommended_green,
    30,
)


comparison = pd.DataFrame(
    {
        "Metric": [
            "Total Queue",
            "Waiting Time",
            "Throughput",
            "Corridor Flow (%)",
        ],

        "Before": [
            before["total_queue"],
            before["waiting_time"],
            before["throughput"],
            before["corridor_flow"],
        ],

        "After AI": [
            after["total_queue"],
            after["waiting_time"],
            after["throughput"],
            after["corridor_flow"],
        ],
    }
)

st.dataframe(
    comparison,
    hide_index=True,
    width="stretch",
)


# ============================================================
# IMPACT
# ============================================================

queue_change = (
    before["total_queue"]
    -
    after["total_queue"]
)

waiting_change = (
    before["waiting_time"]
    -
    after["waiting_time"]
)

throughput_change = (
    after["throughput"]
    -
    before["throughput"]
)

flow_change = (
    after["corridor_flow"]
    -
    before["corridor_flow"]
)


i1, i2, i3, i4 = st.columns(4)

with i1:

    st.metric(
        "Queue Change",
        f"{queue_change:+}"
    )

with i2:

    st.metric(
        "Waiting Time Change",
        f"{waiting_change:+}"
    )

with i3:

    st.metric(
        "Throughput Change",
        f"{throughput_change:+}"
    )

with i4:

    st.metric(
        "Corridor Flow Change",
        f"{flow_change:+.2f} pts"
    )


# ============================================================
# IMPACT CHART
# ============================================================

impact_df = pd.DataFrame(
    {
        "Scenario": [
            "Before AI",
            "After AI",
        ],

        "Corridor Flow (%)": [
            before["corridor_flow"],
            after["corridor_flow"],
        ],
    }
)

st.subheader(
    "📈 Corridor Flow Impact"
)

st.bar_chart(
    impact_df.set_index(
        "Scenario"
    ),
    width="stretch",
)


# ============================================================
# AI INSIGHT
# ============================================================

st.header(
    "🧠 AI Corridor Insight"
)

if recommended_direction != current_signal:

    st.success(
        f"TraFlow detected that "
        f"{recommended_direction} has the "
        f"highest corridor priority. "
        f"The system recommends switching "
        f"the green phase from "
        f"{current_signal} to "
        f"{recommended_direction}."
    )

else:

    st.success(
        f"TraFlow recommends maintaining "
        f"{current_signal} GREEN because "
        f"it currently has the highest "
        f"priority."
    )


# ============================================================
# DECISION SUPPORT
# ============================================================

st.header(
    "🎯 Decision Support"
)

if risk >= 75:

    st.error(
        "🔴 HIGH propagation risk. "
        "Immediate adaptive signal intervention "
        "is recommended."
    )

elif risk >= 50:

    st.warning(
        "🟠 MODERATE propagation risk. "
        "Monitor the corridor and prepare "
        "adaptive signal intervention."
    )

elif risk >= 30:

    st.info(
        "🟡 LOW propagation risk. "
        "Traffic conditions should continue "
        "to be monitored."
    )

else:

    st.success(
        "🟢 MINIMAL propagation risk. "
        "No immediate corridor congestion threat."
    )


# ============================================================
# ARCHITECTURE
# ============================================================

st.header(
    "⚙️ How TraFlow Works"
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
        "LSTM predicts downstream traffic state",
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


architecture_columns = st.columns(3)

for index, step in enumerate(steps):

    with architecture_columns[
        index % 3
    ]:

        st.subheader(
            f"{step[0]} {step[1]}"
        )

        st.caption(
            step[2]
        )


# ============================================================
# SYSTEM OUTPUT
# ============================================================

st.header(
    "📋 TraFlow System Output"
)

st.write(
    f"**Current signal:** "
    f"{current_signal} GREEN"
)

st.write(
    f"**AI recommendation:** "
    f"{recommended_direction} GREEN"
)

st.write(
    f"**Green time:** "
    f"30s → {recommended_green}s"
)

st.write(
    f"**Propagation risk:** "
    f"{risk:.2f}/100 ({risk_level})"
)

st.write(
    f"**Traffic trend:** "
    f"{trend}"
)

st.write(
    f"**Typical propagation lag:** "
    f"{average_lag:.1f} min"
)


# ============================================================
# FOOTER
# ============================================================

timestamp = result.get(
    "current_timestamp",
    "Available dataset observation"
)

st.divider()

st.caption(
    "TraFlow AI • "
    "Corridor-Level Traffic Intelligence "
    "& Adaptive Signal Simulation"
)

st.caption(
    f"Latest dataset observation: {timestamp}"
)

st.caption(
    "Prototype decision-support system. "
    "Signal recommendations are simulated and "
    "are not connected to real traffic infrastructure."
)