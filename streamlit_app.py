"""
Beanconqueror Reference Graph Builder (Streamlit)

Run:
    streamlit run streamlit_app.py
"""
from __future__ import annotations

import json
from datetime import date
from typing import Any, Dict, List, Optional, Tuple

import altair as alt
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components  # noqa: F401

from bc_ref_builder import (
    BrewSection,
    PTSection,
    build_reference_json,
    compute_pt_summaries,
    compute_section_summaries,
)

FLOW_COLOR = "#0071a5"
WEIGHT_COLOR = "#b08d2a"
PRESSURE_COLOR = "#00c853"
TEMP_COLOR = "#ff3d00"


def _chart_render(chart: alt.Chart) -> None:
    try:
        st.altair_chart(chart, width="stretch")
    except TypeError:
        st.altair_chart(chart, use_container_width=True)


def _download_button(label: str, data: bytes, file_name: str, mime: str, key: str) -> None:
    try:
        st.download_button(label, data=data, file_name=file_name, mime=mime, key=key, width="stretch")
    except TypeError:
        st.download_button(label, data=data, file_name=file_name, mime=mime, key=key, use_container_width=True)


def _stretch_button(container, label: str, key: str) -> bool:
    try:
        return container.button(label, key=key, use_container_width=True)
    except TypeError:
        return container.button(label, key=key)


def _stretch_main_button(label: str, key: str) -> bool:
    try:
        return st.button(label, key=key, use_container_width=True)
    except TypeError:
        return st.button(label, key=key)


def _inject_viewport_blocker() -> None:
    st.markdown(
        """
<style>
  #bc_overlay_block {
    display: none;
    position: fixed;
    inset: 0;
    z-index: 2147483647;
    background: rgba(0,0,0,0.78);
    backdrop-filter: blur(2px);
    -webkit-backdrop-filter: blur(2px);
    align-items: center;
    justify-content: center;
    padding: 24px;
    text-align: center;
  }
  #bc_overlay_block .bc_box{
    max-width: 520px;
    width: 100%;
    border-radius: 16px;
    padding: 18px;
    border: 1px solid rgba(255,193,7,0.35);
    background: rgba(255,193,7,0.12);
    color: white;
    font-size: 16px;
    line-height: 1.35;
    font-family: system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif;
  }
  #bc_overlay_block .bc_title{
    font-weight: 700;
    margin-bottom: 8px;
    font-size: 17px;
  }
  @media (max-width: 699px) and (max-aspect-ratio: 1/1) {
    #bc_overlay_block { display: flex; }
  }
</style>

<div id="bc_overlay_block" role="dialog" aria-modal="true">
  <div class="bc_box">
    <div class="bc_title">📱 Editor disabled on small portrait screens</div>
    Rotate to <b>landscape</b> or open on <b>desktop</b>.
  </div>
</div>
        """,
        unsafe_allow_html=True,
    )


WEIGHT_MODE_LABELS = {
    "flow": "Flow",
    "weight_delta": "Δ Weight",
    "weight_target": "Target Weight",
    "constant": "Constant",
}
WEIGHT_LABEL_TO_MODE = {v: k for k, v in WEIGHT_MODE_LABELS.items()}
WEIGHT_MODE_TO_LABEL = {k: v for k, v in WEIGHT_MODE_LABELS.items()}

PT_MODE_LABELS = {
    "constant": "Constant",
    "delta": "Δ Value",
    "target": "Target",
}
PT_LABEL_TO_MODE = {v: k for k, v in PT_MODE_LABELS.items()}
PT_MODE_TO_LABEL = {k: v for k, v in PT_MODE_LABELS.items()}


def _weight_sections_to_compact(sections: List[BrewSection]) -> List[Dict[str, Any]]:
    return [{"d": float(s.duration_s), "m": str(s.mode), "v": float(s.value), "l": str(s.label or "")} for s in sections]


def _weight_sections_from_compact(payload: Any) -> List[BrewSection]:
    if not isinstance(payload, list):
        return []
    out: List[BrewSection] = []
    for it in payload:
        if not isinstance(it, dict):
            continue
        try:
            out.append(
                BrewSection(
                    duration_s=float(it.get("d", 0.0)),
                    mode=str(it.get("m", "constant")),  # type: ignore[arg-type]
                    value=float(it.get("v", 0.0)),
                    label=str(it.get("l", "")),
                )
            )
        except Exception:
            continue
    return out


def _pt_sections_to_compact(sections: List[PTSection]) -> List[Dict[str, Any]]:
    return [{"d": float(s.duration_s), "m": str(s.mode), "v": float(s.value), "l": str(s.label or "")} for s in sections]


def _pt_sections_from_compact(payload: Any) -> List[PTSection]:
    if not isinstance(payload, list):
        return []
    out: List[PTSection] = []
    for it in payload:
        if not isinstance(it, dict):
            continue
        try:
            out.append(
                PTSection(
                    duration_s=float(it.get("d", 0.0)),
                    mode=str(it.get("m", "constant")),  # type: ignore[arg-type]
                    value=float(it.get("v", 0.0)),
                    label=str(it.get("l", "")),
                )
            )
        except Exception:
            continue
    return out


def _restore_from_query_params() -> None:
    if st.session_state.get("_restored_from_url", False):
        return

    qp = st.query_params
    if "profile" in qp:
        try:
            raw = qp.get("profile")
            if isinstance(raw, list):
                raw = raw[0]
            payload = json.loads(raw) if raw else None

            if isinstance(payload, list):
                st.session_state.weight_sections = _weight_sections_from_compact(payload)
                st.session_state.enable_weight = True
                st.session_state.enable_pressure = False
                st.session_state.enable_temperature = False

            elif isinstance(payload, dict):
                en = payload.get("en", {})
                if isinstance(en, dict):
                    st.session_state.enable_weight = bool(en.get("w", True))
                    st.session_state.enable_pressure = bool(en.get("p", False))
                    st.session_state.enable_temperature = bool(en.get("t", False))

                st.session_state.weight_sections = _weight_sections_from_compact(payload.get("w", []))
                st.session_state.pressure_sections = _pt_sections_from_compact(payload.get("p", []))
                st.session_state.temperature_sections = _pt_sections_from_compact(payload.get("t", []))

        except Exception:
            pass

    st.session_state._restored_from_url = True


def _sync_url_if_needed() -> None:
    try:
        payload = {
            "en": {
                "w": bool(st.session_state.enable_weight),
                "p": bool(st.session_state.enable_pressure),
                "t": bool(st.session_state.enable_temperature),
            },
            "w": _weight_sections_to_compact(list(st.session_state.weight_sections)),
            "p": _pt_sections_to_compact(list(st.session_state.pressure_sections)),
            "t": _pt_sections_to_compact(list(st.session_state.temperature_sections)),
        }
        raw = json.dumps(payload, separators=(",", ":"))
        if st.session_state.get("_last_profile_payload") != raw:
            st.query_params["profile"] = raw
            st.session_state._last_profile_payload = raw
    except Exception:
        pass


def _new_id(counter_key: str) -> int:
    if counter_key not in st.session_state:
        st.session_state[counter_key] = 1
    sid = int(st.session_state[counter_key])
    st.session_state[counter_key] = sid + 1
    return sid


def _ensure_ids_match(sections_key: str, ids_key: str, counter_key: str) -> None:
    if ids_key not in st.session_state:
        st.session_state[ids_key] = []
    ids: List[int] = list(st.session_state[ids_key])
    n = len(st.session_state[sections_key])

    while len(ids) < n:
        ids.append(_new_id(counter_key))
    if len(ids) > n:
        ids = ids[:n]
    st.session_state[ids_key] = ids


def _move_pair(sections: List[Any], ids: List[int], i: int, direction: int) -> Tuple[List[Any], List[int]]:
    j = i + direction
    if j < 0 or j >= len(sections):
        return sections, ids
    new_s = sections[:]
    new_i = ids[:]
    new_s[i], new_s[j] = new_s[j], new_s[i]
    new_i[i], new_i[j] = new_i[j], new_i[i]
    return new_s, new_i


def _wkey(prefix: str, section_id: int, field: str) -> str:
    return f"{prefix}_sec_{section_id}_{field}"


def _end_key_weight(section_id: int) -> str:
    v = int(st.session_state.get("weight_target_end_key_version", 0))
    return f"{_wkey('w', section_id, 'end')}_v{v}"


def _end_key_pressure(section_id: int) -> str:
    v = int(st.session_state.get("pressure_target_end_key_version", 0))
    return f"{_wkey('p', section_id, 'end')}_v{v}"


def _end_key_temperature(section_id: int) -> str:
    v = int(st.session_state.get("temperature_target_end_key_version", 0))
    return f"{_wkey('t', section_id, 'end')}_v{v}"


def _default_filename_base() -> str:
    return f"Beanconqueror_Ref_Graph_{date.today():%y%m%d}"


def _sanitize_filename_base(name: str) -> str:
    bad = '<>:"/\\|?*\n\r\t'
    s = (name or "").strip()
    for ch in bad:
        s = s.replace(ch, "_")
    return s.strip(" .")


def _fmt(x: float) -> str:
    return f"{x:.2f}"


def _fmt_time_mmss(t_s: float) -> str:
    t = int(round(float(t_s)))
    m, s = divmod(t, 60)
    return f"{m}:{s:02d}"


def _init_state() -> None:
    if "weight_sections" not in st.session_state:
        st.session_state.weight_sections = []
    if "pressure_sections" not in st.session_state:
        st.session_state.pressure_sections = []
    if "temperature_sections" not in st.session_state:
        st.session_state.temperature_sections = []

    if "enable_weight" not in st.session_state:
        st.session_state.enable_weight = True
    if "enable_pressure" not in st.session_state:
        st.session_state.enable_pressure = False
    if "enable_temperature" not in st.session_state:
        st.session_state.enable_temperature = False

    _restore_from_query_params()

    if "weight_ids" not in st.session_state:
        st.session_state.weight_ids = []
    if "pressure_ids" not in st.session_state:
        st.session_state.pressure_ids = []
    if "temperature_ids" not in st.session_state:
        st.session_state.temperature_ids = []

    _ensure_ids_match("weight_sections", "weight_ids", "weight_next_id")
    _ensure_ids_match("pressure_sections", "pressure_ids", "pressure_next_id")
    _ensure_ids_match("temperature_sections", "temperature_ids", "temperature_next_id")

    if "weight_target_end_key_version" not in st.session_state:
        st.session_state.weight_target_end_key_version = 0
    if "pressure_target_end_key_version" not in st.session_state:
        st.session_state.pressure_target_end_key_version = 0
    if "temperature_target_end_key_version" not in st.session_state:
        st.session_state.temperature_target_end_key_version = 0
    if "load_example" not in st.session_state:
        st.session_state.load_example = False

    for k, v in {
        "w_new_label": "",
        "w_new_mode": WEIGHT_MODE_TO_LABEL["constant"],
        "w_new_duration": 0.0,
        "w_new_flow": 0.0,
        "w_new_delta": 0.0,
        "w_new_end": 0.0,
        "w_reset_add": False,
        "p_new_label": "",
        "p_new_mode": PT_MODE_TO_LABEL["constant"],
        "p_new_duration": 0.0,
        "p_new_delta": 0.0,
        "p_new_target": 0.0,
        "p_reset_add": False,
        "t_new_label": "",
        "t_new_mode": PT_MODE_TO_LABEL["constant"],
        "t_new_duration": 0.0,
        "t_new_delta": 0.0,
        "t_new_target": 0.0,
        "t_reset_add": False,
    }.items():
        if k not in st.session_state:
            st.session_state[k] = v

    if "export_filename" not in st.session_state:
        st.session_state.export_filename = _default_filename_base()

    if st.session_state.w_reset_add:
        st.session_state.w_new_label = ""
        st.session_state.w_new_mode = WEIGHT_MODE_TO_LABEL["constant"]
        st.session_state.w_new_duration = 0.0
        st.session_state.w_new_flow = 0.0
        st.session_state.w_new_delta = 0.0
        st.session_state.w_new_end = 0.0
        st.session_state.w_reset_add = False

    if st.session_state.p_reset_add:
        st.session_state.p_new_label = ""
        st.session_state.p_new_mode = PT_MODE_TO_LABEL["constant"]
        st.session_state.p_new_duration = 0.0
        st.session_state.p_new_delta = 0.0
        st.session_state.p_new_target = 0.0
        st.session_state.p_reset_add = False

    if st.session_state.t_reset_add:
        st.session_state.t_new_label = ""
        st.session_state.t_new_mode = PT_MODE_TO_LABEL["constant"]
        st.session_state.t_new_duration = 0.0
        st.session_state.t_new_delta = 0.0
        st.session_state.t_new_target = 0.0
        st.session_state.t_reset_add = False


def _sync_weight_from_widgets(sections: List[BrewSection], ids: List[int], initial_weight_g: float) -> List[BrewSection]:
    if not sections:
        return []

    summaries = compute_section_summaries(sections, initial_weight_g=float(initial_weight_g))
    new: List[BrewSection] = []

    for sec, sid, summ in zip(sections, ids, summaries):
        label = str(st.session_state.get(_wkey("w", sid, "label"), sec.label))
        mode_label = st.session_state.get(_wkey("w", sid, "mode"), WEIGHT_MODE_TO_LABEL[sec.mode])
        mode = WEIGHT_LABEL_TO_MODE.get(mode_label, sec.mode)

        duration = float(st.session_state.get(_wkey("w", sid, "duration"), sec.duration_s))
        duration = max(duration, 0.1)

        if mode == "constant":
            value = 0.0
        elif mode == "flow":
            value = float(st.session_state.get(_wkey("w", sid, "flow"), summ["flow_gps"]))
        elif mode == "weight_delta":
            value = float(st.session_state.get(_wkey("w", sid, "delta"), summ["delta_weight_g"]))
        else:
            start_w = float(summ["start_weight_g"])
            proposed = float(st.session_state.get(_end_key_weight(sid), summ["end_weight_g"]))
            value = max(proposed, start_w)

        new.append(BrewSection(duration_s=duration, mode=mode, value=float(value), label=label))

    return new


def _sync_pt_from_widgets(prefix: str, sections: List[PTSection], ids: List[int], initial_value: float, end_key_fn) -> List[PTSection]:
    if not sections:
        return []

    summaries = compute_pt_summaries(sections, initial_value=float(initial_value))
    new: List[PTSection] = []

    for sec, sid, summ in zip(sections, ids, summaries):
        label = str(st.session_state.get(_wkey(prefix, sid, "label"), sec.label))
        mode_label = st.session_state.get(_wkey(prefix, sid, "mode"), PT_MODE_TO_LABEL[sec.mode])
        mode = PT_LABEL_TO_MODE.get(mode_label, sec.mode)

        duration = float(st.session_state.get(_wkey(prefix, sid, "duration"), sec.duration_s))
        duration = max(duration, 0.1)

        if mode == "constant":
            value = 0.0
        elif mode == "delta":
            value = float(st.session_state.get(_wkey(prefix, sid, "delta"), summ["delta_value"]))
        else:
            value = float(st.session_state.get(end_key_fn(sid), summ["end_value"]))

        new.append(PTSection(duration_s=duration, mode=mode, value=float(value), label=label))

    return new


def _preview_json(dt_s: float, initial_weight_g: float, initial_pressure: float, initial_temperature: float) -> Dict[str, Any]:
    return build_reference_json(
        dt_s=float(dt_s),
        start_ms=0,
        realtime_timestampdelta_ms=int(round(float(dt_s) * 1000)),
        enable_weight=bool(st.session_state.enable_weight),
        enable_pressure=bool(st.session_state.enable_pressure),
        enable_temperature=bool(st.session_state.enable_temperature),
        weight_sections=list(st.session_state.weight_sections),
        pressure_sections=list(st.session_state.pressure_sections),
        temperature_sections=list(st.session_state.temperature_sections),
        initial_weight_g=float(initial_weight_g),
        initial_pressure=float(initial_pressure),
        initial_temperature=float(initial_temperature),
    )


def _preview_df(out: Dict[str, Any]) -> pd.DataFrame:
    t_vals: List[float] = []

    def add_times(arr: List[Dict[str, Any]]) -> None:
        for it in arr:
            try:
                t_vals.append(float(it["brew_time"]))
            except Exception:
                pass

    add_times(out.get("weight", []))
    add_times(out.get("waterFlow", []))
    add_times(out.get("pressureFlow", []))
    add_times(out.get("temperatureFlow", []))

    if not t_vals:
        return pd.DataFrame({"t_s": []})

    ts = sorted({round(float(x), 10) for x in t_vals})

    w_map = {float(w["brew_time"]): float(w["actual_weight"]) for w in out.get("weight", [])}
    f_map = {float(f["brew_time"]): float(f["value"]) for f in out.get("waterFlow", [])}
    p_map = {float(p["brew_time"]): float(p["actual_pressure"]) for p in out.get("pressureFlow", [])}
    t_map = {float(t["brew_time"]): float(t["actual_temperature"]) for t in out.get("temperatureFlow", [])}

    def lookup(m: Dict[float, float], x: float) -> Optional[float]:
        return m.get(x, None)

    return pd.DataFrame(
        {
            "t_s": ts,
            "weight_g": [lookup(w_map, x) for x in ts],
            "flow_gps": [lookup(f_map, x) for x in ts],
            "pressure": [lookup(p_map, x) for x in ts],
            "temperature": [lookup(t_map, x) for x in ts],
        }
    )


def _chart(df: pd.DataFrame) -> alt.Chart:
    base = alt.Chart(df).encode(
        x=alt.X(
            "t_s:Q",
            title="Time (m:ss)",
            axis=alt.Axis(
                format=".0f",
                labelExpr="floor(datum.value/60) + ':' + (datum.value%60 < 10 ? '0' : '') + (datum.value%60)",
                labelColor="white",
                titleColor="white",
                tickColor="white",
                domainColor="white",
            ),
        )
    )

    layers: List[alt.Chart] = []

    LEFT_OFFSET_0 = 0
    LEFT_OFFSET_1 = 55
    RIGHT_OFFSET_0 = 0
    RIGHT_OFFSET_1 = 55

    OPACITY = 1.0

    WEIGHT_DASH = []           # solid
    FLOW_DASH = [8, 4]         # dashed
    TEMP_DASH = [2, 4]         # dotted
    PRESSURE_DASH = [5, 3, 2, 3]  # dash-dot

    if st.session_state.enable_weight:
        layers.append(
            base.transform_filter(alt.datum.weight_g != None)
            .mark_line(color=WEIGHT_COLOR, strokeWidth=2, opacity=OPACITY, strokeDash=WEIGHT_DASH)
            .encode(
                y=alt.Y(
                    "weight_g:Q",
                    title="Weight",
                    axis=alt.Axis(
                        orient="left",
                        offset=LEFT_OFFSET_0,
                        labelColor=WEIGHT_COLOR,
                        titleColor=WEIGHT_COLOR,
                        tickColor=WEIGHT_COLOR,
                        domainColor=WEIGHT_COLOR,
                    ),
                )
            )
        )

        layers.append(
            base.transform_filter(alt.datum.flow_gps != None)
            .mark_line(color=FLOW_COLOR, strokeWidth=2, opacity=OPACITY, strokeDash=FLOW_DASH)
            .encode(
                y=alt.Y(
                    "flow_gps:Q",
                    title="Flow",
                    axis=alt.Axis(
                        orient="right",
                        offset=RIGHT_OFFSET_0,
                        labelColor=FLOW_COLOR,
                        titleColor=FLOW_COLOR,
                        tickColor=FLOW_COLOR,
                        domainColor=FLOW_COLOR,
                    ),
                )
            )
        )

    if st.session_state.enable_temperature:
        layers.append(
            base.transform_filter(alt.datum.temperature != None)
            .mark_line(color=TEMP_COLOR, strokeWidth=2, opacity=OPACITY, strokeDash=TEMP_DASH)
            .encode(
                y=alt.Y(
                    "temperature:Q",
                    title="Temperature",
                    axis=alt.Axis(
                        orient="left",
                        offset=LEFT_OFFSET_1,
                        labelColor=TEMP_COLOR,
                        titleColor=TEMP_COLOR,
                        tickColor=TEMP_COLOR,
                        domainColor=TEMP_COLOR,
                    ),
                )
            )
        )

    if st.session_state.enable_pressure:
        layers.append(
            base.transform_filter(alt.datum.pressure != None)
            .mark_line(color=PRESSURE_COLOR, strokeWidth=2, opacity=OPACITY, strokeDash=PRESSURE_DASH)
            .encode(
                y=alt.Y(
                    "pressure:Q",
                    title="Pressure",
                    axis=alt.Axis(
                        orient="right",
                        offset=RIGHT_OFFSET_1,
                        labelColor=PRESSURE_COLOR,
                        titleColor=PRESSURE_COLOR,
                        tickColor=PRESSURE_COLOR,
                        domainColor=PRESSURE_COLOR,
                    ),
                )
            )
        )

    if not layers:
        return base.mark_line().encode(y=alt.value(0)).properties(height=280)

    return (
        alt.layer(*layers)
        .resolve_scale(y="independent")
        .properties(height=280, padding={"left": 90, "right": 90, "top": 10, "bottom": 10})
        .configure_view(stroke=None)
        .configure_axis(grid=False)
    )


def _ordered_keys_for_tree(data: Dict[str, Any]) -> List[str]:
    filled = [k for k, v in data.items() if isinstance(v, list) and len(v) > 0]
    empty = [k for k, v in data.items() if isinstance(v, list) and len(v) == 0]
    other = [k for k, v in data.items() if not isinstance(v, list)]
    filled.sort(key=str.lower)
    empty.sort(key=str.lower)
    other.sort(key=str.lower)
    return filled + empty + other


def _json_tree(data: Dict[str, Any]) -> str:
    keys = _ordered_keys_for_tree(data)
    lines: List[str] = ["root"]
    for idx, k in enumerate(keys):
        v = data[k]
        last = idx == len(keys) - 1
        branch = "└─ " if last else "├─ "
        cont = "   " if last else "│  "
        if isinstance(v, list):
            suffix = " (empty)" if len(v) == 0 else ""
            lines.append(f"{branch}{k}{suffix}")
            if len(v) > 0 and isinstance(v[0], dict):
                for ck in sorted(v[0].keys()):
                    lines.append(f"{cont}├─ {ck}")
        elif isinstance(v, dict):
            lines.append(f"{branch}{k}")
            for ck in sorted(v.keys()):
                lines.append(f"{cont}├─ {ck}")
        else:
            lines.append(f"{branch}{k}")
    return "\n".join(lines)


def _example_weight_sections() -> List[BrewSection]:
    return [
        BrewSection(duration_s=10, mode="flow", value=5.0, label="Add bloom water"),
        BrewSection(duration_s=35, mode="constant", value=0.0, label="Bloom"),
        BrewSection(duration_s=15, mode="weight_delta", value=50.0, label="Pour 50g"),
        BrewSection(duration_s=10, mode="constant", value=0.0, label="Wait"),
        BrewSection(duration_s=10, mode="weight_delta", value=50.0, label="Pour 50g"),
        BrewSection(duration_s=10, mode="constant", value=0.0, label="Wait"),
        BrewSection(duration_s=10, mode="weight_delta", value=50.0, label="Pour 50g"),
        BrewSection(duration_s=10, mode="constant", value=0.0, label="Wait"),
        BrewSection(duration_s=10, mode="weight_delta", value=50.0, label="Pour 50g"),
        BrewSection(duration_s=60, mode="constant", value=0.0, label="Wait for draw-down"),
    ]


def _example_pressure_sections(total_s: float) -> List[PTSection]:
    total_s = max(float(total_s), 0.1)

    # simple espresso-like profile
    return [
        PTSection(duration_s=0.20 * total_s, mode="target", value=3.0, label="Preinfusion"),
        PTSection(duration_s=0.15 * total_s, mode="target", value=9.0, label="Ramp"),
        PTSection(duration_s=0.45 * total_s, mode="constant", value=0.0, label="Hold"),
        PTSection(duration_s=0.20 * total_s, mode="target", value=6.0, label="Decline"),
    ]


def _example_temperature_sections(total_s: float) -> List[PTSection]:
    total_s = max(float(total_s), 0.1)

    # simple slight decline
    return [
        PTSection(duration_s=0.35 * total_s, mode="target", value=94.0, label="Start"),
        PTSection(duration_s=0.65 * total_s, mode="target", value=92.0, label="Decline"),
    ]


def main() -> None:
    st.set_page_config(page_title="Beanconqueror Reference Graph Builder", page_icon="☕", layout="wide")
    _inject_viewport_blocker()
    _init_state()

    st.title("Beanconqueror Reference Graph Builder ☕")

    st.markdown(
        """
    A lightweight tool to generate **reference brew graphs** for the coffee logging app **Beanconqueror**.

    Beanconqueror allows importing graphs as JSON time series (e.g., weight, flow, pressure, temperature) which can be used as a reference in the background while tracking a new brew in real-time.  
    This app provides an interactive editor to build such reference profiles section-by-section and export them in the JSON format expected by Beanconqueror.

    - Supports a **Weight / Flow profile editor**
    - Section-based logic: **Constant**, **Δ Weight**, **Target Weight**, **Flow**
    - Supports a **Pressure profile editor**
    - Section-based logic: **Constant**, **Δ Value**, **Target**
    - Supports a **Temperature profile editor**
    - Section-based logic: **Constant**, **Δ Value**, **Target**
    - Enable/disable each profile independently (weight/flow, pressure, temperature)
    - Automatically computes the corresponding time series
    - Visualizes all enabled profiles on the same graph before export
    - Exports JSON compatible with Beanconqueror reference graphs
        """
    )

    st.markdown("---")
    st.header("Graph Overview")

    if st.session_state.get("load_example", False):
        st.session_state.weight_sections = _example_weight_sections()
        total_s = sum(float(s.duration_s) for s in st.session_state.weight_sections)
        st.session_state.pressure_sections = _example_pressure_sections(total_s)
        st.session_state.temperature_sections = _example_temperature_sections(total_s)

        st.session_state.weight_ids = [_new_id("weight_next_id") for _ in range(len(st.session_state.weight_sections))]
        st.session_state.pressure_ids = [_new_id("pressure_next_id") for _ in range(len(st.session_state.pressure_sections))]
        st.session_state.temperature_ids = [_new_id("temperature_next_id") for _ in range(len(st.session_state.temperature_sections))]

        st.session_state.enable_weight = True
        st.session_state.enable_pressure = True
        st.session_state.enable_temperature = True

        st.session_state.load_example = False
        _sync_url_if_needed()
        st.rerun()

    with st.sidebar:
        st.header("Settings")

        dt_s = st.number_input("Sample period dT (s)", min_value=0.01, max_value=2.0, value=0.1, step=0.01)

        st.markdown("### Enable profiles")
        st.checkbox("Weight / Flow", key="enable_weight")
        st.checkbox("Pressure", key="enable_pressure")
        st.checkbox("Temperature", key="enable_temperature")

        st.markdown("---")
        initial_weight_g = st.number_input("Initial weight (g)", min_value=0.0, max_value=10_000.0, value=0.0, step=1.0)
        initial_pressure = st.number_input("Initial pressure", value=0.0, step=0.1)
        initial_temperature = st.number_input("Initial temperature", value=0.0, step=0.1)

        st.markdown("---")

        if st.button("🧹 Clear ALL profiles"):
            st.session_state.weight_sections = []
            st.session_state.pressure_sections = []
            st.session_state.temperature_sections = []
            st.session_state.weight_ids = []
            st.session_state.pressure_ids = []
            st.session_state.temperature_ids = []
            _sync_url_if_needed()
            st.rerun()

        if st.button("✨ Import example"):
            st.session_state.load_example = True
            st.rerun()

    out = _preview_json(float(dt_s), float(initial_weight_g), float(initial_pressure), float(initial_temperature))
    df = _preview_df(out)

    if len(df) > 0:
        _chart_render(_chart(df))
        st.caption("Weight / Temperature (left) and Flow / Pressure (right) appear when enabled.")
    else:
        st.info("Nothing to plot yet. Enable a profile and add sections.")

    st.markdown("---")

    if st.session_state.enable_weight:
        st.header("Weight/ Flow Profile Editor")

        _ensure_ids_match("weight_sections", "weight_ids", "weight_next_id")
        sections: List[BrewSection] = list(st.session_state.weight_sections)
        ids: List[int] = list(st.session_state.weight_ids)

        if sections:
            summaries = compute_section_summaries(sections, initial_weight_g=float(initial_weight_g))

            header_cols = st.columns([2.0, 2.2, 1.4, 1.1, 1.1, 1.3, 1.3, 1.2, 1.2, 0.5, 0.5, 0.5])
            header_cols[0].markdown("**Label**")
            header_cols[1].markdown("**Mode**")
            header_cols[2].markdown("**Duration (s)**")
            header_cols[3].markdown("**Start Time**")
            header_cols[4].markdown("**End Time**")
            header_cols[5].markdown("**Flow (g/s)**")
            header_cols[6].markdown("**ΔWeight (g)**")
            header_cols[7].markdown("**Start Weight**")
            header_cols[8].markdown("**End Weight**")
            header_cols[9].markdown("**↑**")
            header_cols[10].markdown("**↓**")
            header_cols[11].markdown("**Del**")

            for i, (sec, sid, row) in enumerate(zip(sections, ids, summaries)):
                cols = st.columns([2.0, 2.2, 1.4, 1.1, 1.1, 1.3, 1.3, 1.2, 1.2, 0.5, 0.5, 0.5])

                cols[0].text_input("Label", value=sec.label, key=_wkey("w", sid, "label"), label_visibility="collapsed")

                cols[1].selectbox(
                    "Mode",
                    options=list(WEIGHT_LABEL_TO_MODE.keys()),
                    index=list(WEIGHT_LABEL_TO_MODE.keys()).index(WEIGHT_MODE_TO_LABEL[sec.mode]),
                    key=_wkey("w", sid, "mode"),
                    label_visibility="collapsed",
                )

                cols[2].number_input(
                    "Duration (s)",
                    min_value=0.1,
                    max_value=3600.0,
                    value=float(sec.duration_s),
                    step=0.5,
                    key=_wkey("w", sid, "duration"),
                    label_visibility="collapsed",
                )

                cols[3].markdown(_fmt_time_mmss(row["t_start_s"]))
                cols[4].markdown(_fmt_time_mmss(row["t_end_s"]))

                mode_label = st.session_state.get(_wkey("w", sid, "mode"), WEIGHT_MODE_TO_LABEL[sec.mode])
                mode = WEIGHT_LABEL_TO_MODE[mode_label]

                if mode == "constant":
                    cols[5].markdown("—")
                    cols[6].markdown("—")
                elif mode == "flow":
                    cols[5].number_input(
                        "Flow (g/s)",
                        min_value=0.0,
                        max_value=1000.0,
                        value=float(row["flow_gps"]),
                        step=0.1,
                        key=_wkey("w", sid, "flow"),
                        label_visibility="collapsed",
                    )
                    cols[6].markdown(_fmt(float(row["delta_weight_g"])))
                elif mode == "weight_delta":
                    cols[5].markdown(_fmt(float(row["flow_gps"])))
                    cols[6].number_input(
                        "ΔWeight (g)",
                        min_value=0.0,
                        max_value=10_000.0,
                        value=float(row["delta_weight_g"]),
                        step=1.0,
                        key=_wkey("w", sid, "delta"),
                        label_visibility="collapsed",
                    )
                else:
                    cols[5].markdown(_fmt(float(row["flow_gps"])))
                    cols[6].markdown(_fmt(float(row["delta_weight_g"])))

                cols[7].markdown(_fmt(float(row["start_weight_g"])))

                if mode == "weight_target":
                    start_w = float(row["start_weight_g"])
                    cols[8].number_input(
                        "End Weight (g)",
                        min_value=start_w,
                        max_value=10_000.0,
                        value=float(max(float(row["end_weight_g"]), start_w)),
                        step=1.0,
                        key=_end_key_weight(sid),
                        label_visibility="collapsed",
                    )
                else:
                    cols[8].markdown(_fmt(float(row["end_weight_g"])))

                if _stretch_button(cols[9], "↑", key=f"w_up_{sid}"):
                    new_secs, new_ids = _move_pair(sections, ids, i, -1)
                    st.session_state.weight_sections = new_secs
                    st.session_state.weight_ids = new_ids
                    st.session_state.weight_target_end_key_version = int(st.session_state.weight_target_end_key_version) + 1
                    _sync_url_if_needed()
                    st.rerun()

                if _stretch_button(cols[10], "↓", key=f"w_down_{sid}"):
                    new_secs, new_ids = _move_pair(sections, ids, i, +1)
                    st.session_state.weight_sections = new_secs
                    st.session_state.weight_ids = new_ids
                    st.session_state.weight_target_end_key_version = int(st.session_state.weight_target_end_key_version) + 1
                    _sync_url_if_needed()
                    st.rerun()

                if _stretch_button(cols[11], "🗑️", key=f"w_del_{sid}"):
                    st.session_state.weight_sections = sections[:i] + sections[i + 1 :]
                    st.session_state.weight_ids = ids[:i] + ids[i + 1 :]
                    _sync_url_if_needed()
                    st.rerun()

            new_sections = _sync_weight_from_widgets(
                list(st.session_state.weight_sections),
                list(st.session_state.weight_ids),
                float(initial_weight_g),
            )
            if new_sections != list(st.session_state.weight_sections):
                st.session_state.weight_sections = new_sections
                _sync_url_if_needed()
                st.rerun()

        st.markdown("---")
        # st.subheader("+ Add Weight Section")

        add_cols = st.columns([2.0, 2.2, 1.4, 1.3, 1.3])
        add_cols[0].text_input("Label", key="w_new_label")
        add_cols[1].selectbox("Mode", options=list(WEIGHT_LABEL_TO_MODE.keys()), key="w_new_mode")
        add_cols[2].number_input("Duration (s)", min_value=0.0, max_value=3600.0, step=0.5, key="w_new_duration")

        new_mode = WEIGHT_LABEL_TO_MODE[st.session_state.w_new_mode]

        if new_mode == "constant":
            add_cols[3].markdown("Flow: —")
            add_cols[4].markdown("ΔWeight: —")
        elif new_mode == "flow":
            add_cols[3].number_input("Flow (g/s)", min_value=0.0, max_value=1000.0, step=0.1, key="w_new_flow")
            add_cols[4].markdown("ΔWeight: auto")
        elif new_mode == "weight_delta":
            add_cols[3].markdown("Flow: auto")
            add_cols[4].number_input("ΔWeight (g)", min_value=0.0, max_value=10_000.0, step=1.0, key="w_new_delta")
        else:
            add_cols[3].markdown("Flow: auto")
            add_cols[4].number_input("End Weight (g)", min_value=0.0, max_value=10_000.0, step=1.0, key="w_new_end")

        if _stretch_main_button("Add weight section", key="w_add_btn"):
            dur = float(st.session_state.w_new_duration)
            if dur <= 0:
                st.error("Duration must be > 0")
            else:
                if new_mode == "constant":
                    sec = BrewSection(duration_s=dur, mode="constant", value=0.0, label=str(st.session_state.w_new_label))
                elif new_mode == "flow":
                    sec = BrewSection(duration_s=dur, mode="flow", value=float(st.session_state.w_new_flow), label=str(st.session_state.w_new_label))
                elif new_mode == "weight_delta":
                    sec = BrewSection(duration_s=dur, mode="weight_delta", value=float(st.session_state.w_new_delta), label=str(st.session_state.w_new_label))
                else:
                    sec = BrewSection(duration_s=dur, mode="weight_target", value=float(st.session_state.w_new_end), label=str(st.session_state.w_new_label))

                st.session_state.weight_sections = list(st.session_state.weight_sections) + [sec]
                st.session_state.weight_ids = list(st.session_state.weight_ids) + [_new_id("weight_next_id")]
                st.session_state.w_reset_add = True
                _sync_url_if_needed()
                st.rerun()
    else:
        st.header("Weight Profile Editor")
        st.info("Weight profile disabled.")

    st.markdown("---")
    st.header("Pressure Profile Editor")
    if not st.session_state.enable_pressure:
        st.info("Pressure profile disabled.")
    else:
        _ensure_ids_match("pressure_sections", "pressure_ids", "pressure_next_id")
        p_secs: List[PTSection] = list(st.session_state.pressure_sections)
        p_ids: List[int] = list(st.session_state.pressure_ids)

        if p_secs:
            p_sum = compute_pt_summaries(p_secs, initial_value=float(initial_pressure))

            header_cols = st.columns([2.0, 2.0, 1.4, 1.1, 1.1, 1.3, 1.2, 1.2, 0.5, 0.5, 0.5])
            header_cols[0].markdown("**Label**")
            header_cols[1].markdown("**Mode**")
            header_cols[2].markdown("**Duration (s)**")
            header_cols[3].markdown("**Start Time**")
            header_cols[4].markdown("**End Time**")
            header_cols[5].markdown("**ΔPressure**")
            header_cols[6].markdown("**Start**")
            header_cols[7].markdown("**End**")
            header_cols[8].markdown("**↑**")
            header_cols[9].markdown("**↓**")
            header_cols[10].markdown("**Del**")

            for i, (sec, sid, row) in enumerate(zip(p_secs, p_ids, p_sum)):
                cols = st.columns([2.0, 2.0, 1.4, 1.1, 1.1, 1.3, 1.2, 1.2, 0.5, 0.5, 0.5])

                cols[0].text_input("Label", value=sec.label, key=_wkey("p", sid, "label"), label_visibility="collapsed")

                cols[1].selectbox(
                    "Mode",
                    options=list(PT_LABEL_TO_MODE.keys()),
                    index=list(PT_LABEL_TO_MODE.keys()).index(PT_MODE_TO_LABEL[sec.mode]),
                    key=_wkey("p", sid, "mode"),
                    label_visibility="collapsed",
                )

                cols[2].number_input(
                    "Duration (s)",
                    min_value=0.1,
                    max_value=3600.0,
                    value=float(sec.duration_s),
                    step=0.5,
                    key=_wkey("p", sid, "duration"),
                    label_visibility="collapsed",
                )

                cols[3].markdown(_fmt_time_mmss(row["t_start_s"]))
                cols[4].markdown(_fmt_time_mmss(row["t_end_s"]))

                mode_label = st.session_state.get(_wkey("p", sid, "mode"), PT_MODE_TO_LABEL[sec.mode])
                mode = PT_LABEL_TO_MODE.get(mode_label, sec.mode)

                if mode == "delta":
                    cols[5].number_input(
                        "ΔPressure",
                        value=float(row["delta_value"]),
                        step=0.1,
                        key=_wkey("p", sid, "delta"),
                        label_visibility="collapsed",
                    )
                else:
                    cols[5].markdown(_fmt(float(row["delta_value"])))

                cols[6].markdown(_fmt(float(row["start_value"])))

                if mode == "target":
                    cols[7].number_input(
                        "End",
                        value=float(row["end_value"]),
                        step=0.1,
                        key=_end_key_pressure(sid),
                        label_visibility="collapsed",
                    )
                else:
                    cols[7].markdown(_fmt(float(row["end_value"])))

                if _stretch_button(cols[8], "↑", key=f"p_up_{sid}"):
                    new_secs, new_ids = _move_pair(p_secs, p_ids, i, -1)
                    st.session_state.pressure_sections = new_secs
                    st.session_state.pressure_ids = new_ids
                    st.session_state.pressure_target_end_key_version = int(st.session_state.pressure_target_end_key_version) + 1
                    _sync_url_if_needed()
                    st.rerun()

                if _stretch_button(cols[9], "↓", key=f"p_down_{sid}"):
                    new_secs, new_ids = _move_pair(p_secs, p_ids, i, +1)
                    st.session_state.pressure_sections = new_secs
                    st.session_state.pressure_ids = new_ids
                    st.session_state.pressure_target_end_key_version = int(st.session_state.pressure_target_end_key_version) + 1
                    _sync_url_if_needed()
                    st.rerun()

                if _stretch_button(cols[10], "🗑️", key=f"p_del_{sid}"):
                    st.session_state.pressure_sections = p_secs[:i] + p_secs[i + 1 :]
                    st.session_state.pressure_ids = p_ids[:i] + p_ids[i + 1 :]
                    _sync_url_if_needed()
                    st.rerun()

            new_p = _sync_pt_from_widgets(
                "p",
                list(st.session_state.pressure_sections),
                list(st.session_state.pressure_ids),
                float(initial_pressure),
                _end_key_pressure,
            )
            if new_p != list(st.session_state.pressure_sections):
                st.session_state.pressure_sections = new_p
                _sync_url_if_needed()
                st.rerun()

        st.markdown("---")
        # st.subheader("+ Add Pressure Section")
        add_cols = st.columns([2.0, 2.0, 1.4, 1.6])
        add_cols[0].text_input("Label", key="p_new_label")
        add_cols[1].selectbox("Mode", options=list(PT_LABEL_TO_MODE.keys()), key="p_new_mode")
        add_cols[2].number_input("Duration (s)", min_value=0.0, max_value=3600.0, step=0.5, key="p_new_duration")

        p_mode = PT_LABEL_TO_MODE[st.session_state.p_new_mode]
        if p_mode == "constant":
            add_cols[3].markdown("Constant: hold")
        elif p_mode == "delta":
            add_cols[3].number_input("Δ", step=0.1, key="p_new_delta")
        else:
            add_cols[3].number_input("Target", step=0.1, key="p_new_target")

        if _stretch_main_button("Add pressure section", key="p_add_btn"):
            dur = float(st.session_state.p_new_duration)
            if dur <= 0:
                st.error("Duration must be > 0")
            else:
                if p_mode == "constant":
                    sec = PTSection(duration_s=dur, mode="constant", value=0.0, label=str(st.session_state.p_new_label))
                elif p_mode == "delta":
                    sec = PTSection(duration_s=dur, mode="delta", value=float(st.session_state.p_new_delta), label=str(st.session_state.p_new_label))
                else:
                    sec = PTSection(duration_s=dur, mode="target", value=float(st.session_state.p_new_target), label=str(st.session_state.p_new_label))

                st.session_state.pressure_sections = list(st.session_state.pressure_sections) + [sec]
                st.session_state.pressure_ids = list(st.session_state.pressure_ids) + [_new_id("pressure_next_id")]
                st.session_state.p_reset_add = True
                _sync_url_if_needed()
                st.rerun()

    st.markdown("---")
    st.header("Temperature Profile Editor")
    if not st.session_state.enable_temperature:
        st.info("Temperature profile disabled.")
    else:
        _ensure_ids_match("temperature_sections", "temperature_ids", "temperature_next_id")
        t_secs: List[PTSection] = list(st.session_state.temperature_sections)
        t_ids: List[int] = list(st.session_state.temperature_ids)

        if t_secs:
            t_sum = compute_pt_summaries(t_secs, initial_value=float(initial_temperature))

            header_cols = st.columns([2.0, 2.0, 1.4, 1.1, 1.1, 1.3, 1.2, 1.2, 0.5, 0.5, 0.5])
            header_cols[0].markdown("**Label**")
            header_cols[1].markdown("**Mode**")
            header_cols[2].markdown("**Duration (s)**")
            header_cols[3].markdown("**Start Time**")
            header_cols[4].markdown("**End Time**")
            header_cols[5].markdown("**ΔTemp**")
            header_cols[6].markdown("**Start**")
            header_cols[7].markdown("**End**")
            header_cols[8].markdown("**↑**")
            header_cols[9].markdown("**↓**")
            header_cols[10].markdown("**Del**")

            for i, (sec, sid, row) in enumerate(zip(t_secs, t_ids, t_sum)):
                cols = st.columns([2.0, 2.0, 1.4, 1.1, 1.1, 1.3, 1.2, 1.2, 0.5, 0.5, 0.5])

                cols[0].text_input("Label", value=sec.label, key=_wkey("t", sid, "label"), label_visibility="collapsed")

                cols[1].selectbox(
                    "Mode",
                    options=list(PT_LABEL_TO_MODE.keys()),
                    index=list(PT_LABEL_TO_MODE.keys()).index(PT_MODE_TO_LABEL[sec.mode]),
                    key=_wkey("t", sid, "mode"),
                    label_visibility="collapsed",
                )

                cols[2].number_input(
                    "Duration (s)",
                    min_value=0.1,
                    max_value=3600.0,
                    value=float(sec.duration_s),
                    step=0.5,
                    key=_wkey("t", sid, "duration"),
                    label_visibility="collapsed",
                )

                cols[3].markdown(_fmt_time_mmss(row["t_start_s"]))
                cols[4].markdown(_fmt_time_mmss(row["t_end_s"]))

                mode_label = st.session_state.get(_wkey("t", sid, "mode"), PT_MODE_TO_LABEL[sec.mode])
                mode = PT_LABEL_TO_MODE.get(mode_label, sec.mode)

                if mode == "delta":
                    cols[5].number_input(
                        "ΔTemp",
                        value=float(row["delta_value"]),
                        step=0.1,
                        key=_wkey("t", sid, "delta"),
                        label_visibility="collapsed",
                    )
                else:
                    cols[5].markdown(_fmt(float(row["delta_value"])))

                cols[6].markdown(_fmt(float(row["start_value"])))

                if mode == "target":
                    cols[7].number_input(
                        "End",
                        value=float(row["end_value"]),
                        step=0.1,
                        key=_end_key_temperature(sid),
                        label_visibility="collapsed",
                    )
                else:
                    cols[7].markdown(_fmt(float(row["end_value"])))

                if _stretch_button(cols[8], "↑", key=f"t_up_{sid}"):
                    new_secs, new_ids = _move_pair(t_secs, t_ids, i, -1)
                    st.session_state.temperature_sections = new_secs
                    st.session_state.temperature_ids = new_ids
                    st.session_state.temperature_target_end_key_version = int(st.session_state.temperature_target_end_key_version) + 1
                    _sync_url_if_needed()
                    st.rerun()

                if _stretch_button(cols[9], "↓", key=f"t_down_{sid}"):
                    new_secs, new_ids = _move_pair(t_secs, t_ids, i, +1)
                    st.session_state.temperature_sections = new_secs
                    st.session_state.temperature_ids = new_ids
                    st.session_state.temperature_target_end_key_version = int(st.session_state.temperature_target_end_key_version) + 1
                    _sync_url_if_needed()
                    st.rerun()

                if _stretch_button(cols[10], "🗑️", key=f"t_del_{sid}"):
                    st.session_state.temperature_sections = t_secs[:i] + t_secs[i + 1 :]
                    st.session_state.temperature_ids = t_ids[:i] + t_ids[i + 1 :]
                    _sync_url_if_needed()
                    st.rerun()

            new_t = _sync_pt_from_widgets(
                "t",
                list(st.session_state.temperature_sections),
                list(st.session_state.temperature_ids),
                float(initial_temperature),
                _end_key_temperature,
            )
            if new_t != list(st.session_state.temperature_sections):
                st.session_state.temperature_sections = new_t
                _sync_url_if_needed()
                st.rerun()

        st.markdown("---")
        # st.subheader("+ Add Temperature Section")
        add_cols = st.columns([2.0, 2.0, 1.4, 1.6])
        add_cols[0].text_input("Label", key="t_new_label")
        add_cols[1].selectbox("Mode", options=list(PT_LABEL_TO_MODE.keys()), key="t_new_mode")
        add_cols[2].number_input("Duration (s)", min_value=0.0, max_value=3600.0, step=0.5, key="t_new_duration")

        t_mode = PT_LABEL_TO_MODE[st.session_state.t_new_mode]
        if t_mode == "constant":
            add_cols[3].markdown("Constant: hold")
        elif t_mode == "delta":
            add_cols[3].number_input("Δ", step=0.1, key="t_new_delta")
        else:
            add_cols[3].number_input("Target", step=0.1, key="t_new_target")

        if _stretch_main_button("Add temperature section", key="t_add_btn"):
            dur = float(st.session_state.t_new_duration)
            if dur <= 0:
                st.error("Duration must be > 0")
            else:
                if t_mode == "constant":
                    sec = PTSection(duration_s=dur, mode="constant", value=0.0, label=str(st.session_state.t_new_label))
                elif t_mode == "delta":
                    sec = PTSection(duration_s=dur, mode="delta", value=float(st.session_state.t_new_delta), label=str(st.session_state.t_new_label))
                else:
                    sec = PTSection(duration_s=dur, mode="target", value=float(st.session_state.t_new_target), label=str(st.session_state.t_new_label))

                st.session_state.temperature_sections = list(st.session_state.temperature_sections) + [sec]
                st.session_state.temperature_ids = list(st.session_state.temperature_ids) + [_new_id("temperature_next_id")]
                st.session_state.t_reset_add = True
                _sync_url_if_needed()
                st.rerun()

    st.markdown("---")
    st.header("Export JSON")

    st.text_input(
        "File name",
        key="export_filename",
        help='Name without extension. ".json" is added automatically.',
    )

    out_bytes = json.dumps(out, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    fn_base = _sanitize_filename_base(str(st.session_state.export_filename)) or _default_filename_base()
    fn = fn_base if fn_base.lower().endswith(".json") else (fn_base + ".json")

    _download_button("📥 Download JSON", data=out_bytes, file_name=fn, mime="application/json", key="download_json")

    with st.expander("JSON structure"):
        st.code(_json_tree(out), language="text")

    _sync_url_if_needed()


if __name__ == "__main__":
    main()