# streamlit_app.py
"""
Beanconqueror Reference Graph Builder (Streamlit)

Run:
    streamlit run streamlit_app.py
"""
from __future__ import annotations

import json
from datetime import date
from typing import Any, Dict, List, Tuple

import altair as alt
import pandas as pd
import streamlit as st

from bc_ref_builder import BrewSection, build_reference_json, compute_section_summaries

FLOW_COLOR = "#0071a5"    # blue
WEIGHT_COLOR = "#b08d2a"  # coffee yellow-brown

MODE_LABELS = {
    "flow": "Flow",
    "weight_delta": "Δ Weight",
    "weight_target": "Target Weight",
    "wait": "Wait",
}
LABEL_TO_MODE = {v: k for k, v in MODE_LABELS.items()}
MODE_TO_LABEL = {k: v for k, v in MODE_LABELS.items()}


# ----------------------------
# Persistence via URL query param
# ----------------------------
def _sections_to_compact(sections: List[BrewSection]) -> List[Dict[str, Any]]:
    return [{"d": float(s.duration_s), "m": str(s.mode), "v": float(s.value), "l": str(s.label or "")} for s in sections]


def _sections_from_compact(payload: Any) -> List[BrewSection]:
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
                    mode=str(it.get("m", "wait")),  # type: ignore[arg-type]
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
            secs = _sections_from_compact(payload)
            if secs:
                st.session_state.sections = secs
        except Exception:
            pass

    st.session_state._restored_from_url = True


def _sync_url_if_needed() -> None:
    """Update ?profile=... only when it actually changed (prevents annoying extra reruns online)."""
    try:
        payload = json.dumps(_sections_to_compact(st.session_state.sections), separators=(",", ":"))
        if st.session_state.get("_last_profile_payload") != payload:
            st.query_params["profile"] = payload
            st.session_state._last_profile_payload = payload
    except Exception:
        pass


# ----------------------------
# Stable IDs for rows (fixes reorder-online)
# ----------------------------
def _new_section_id() -> int:
    if "next_section_id" not in st.session_state:
        st.session_state.next_section_id = 1
    sid = int(st.session_state.next_section_id)
    st.session_state.next_section_id = sid + 1
    return sid


def _ensure_ids_match_sections() -> None:
    """Make sure section_ids exists and matches len(sections)."""
    if "section_ids" not in st.session_state:
        st.session_state.section_ids = []

    ids: List[int] = list(st.session_state.section_ids)
    n = len(st.session_state.sections)

    # extend
    while len(ids) < n:
        ids.append(_new_section_id())

    # trim
    if len(ids) > n:
        ids = ids[:n]

    st.session_state.section_ids = ids


def _reset_ids_for_sections() -> None:
    """Assign fresh ids (useful after importing/restoring)."""
    st.session_state.section_ids = [_new_section_id() for _ in range(len(st.session_state.sections))]


def _move_section_pair(sections: List[BrewSection], ids: List[int], i: int, direction: int) -> Tuple[List[BrewSection], List[int]]:
    j = i + direction
    if j < 0 or j >= len(sections):
        return sections, ids
    new_secs = sections[:]
    new_ids = ids[:]
    new_secs[i], new_secs[j] = new_secs[j], new_secs[i]
    new_ids[i], new_ids[j] = new_ids[j], new_ids[i]
    return new_secs, new_ids


# ----------------------------
# UI helpers
# ----------------------------
def _default_filename_base() -> str:
    return f"Beanconqueror_Ref_Graph_{date.today():%y%m%d}"


def _init_state() -> None:
    if "sections" not in st.session_state:
        st.session_state.sections = []

    _restore_from_query_params()

    # init ids after restore (or first run)
    if "section_ids" not in st.session_state:
        st.session_state.section_ids = []
    if "next_section_id" not in st.session_state:
        st.session_state.next_section_id = 1

    # If we restored sections from URL and ids are empty, create ids.
    _ensure_ids_match_sections()
    if len(st.session_state.section_ids) != len(st.session_state.sections):
        _reset_ids_for_sections()

    # Add-row widget keys
    for k, v in {
        "new_label_input": "",
        "new_mode_input": MODE_TO_LABEL["wait"],
        "new_duration_input": 0.0,
        "new_flow_input": 0.0,
        "new_delta_input": 0.0,
        "new_end_input": 0.0,
        "reset_add_row": False,
    }.items():
        if k not in st.session_state:
            st.session_state[k] = v

    if "export_filename" not in st.session_state:
        st.session_state.export_filename = _default_filename_base()

    if st.session_state.reset_add_row:
        st.session_state["new_label_input"] = ""
        st.session_state["new_mode_input"] = MODE_TO_LABEL["wait"]
        st.session_state["new_duration_input"] = 0.0
        st.session_state["new_flow_input"] = 0.0
        st.session_state["new_delta_input"] = 0.0
        st.session_state["new_end_input"] = 0.0
        st.session_state.reset_add_row = False


def _fmt(x: float) -> str:
    return f"{x:.2f}"


def _fmt_time_mmss(t_s: float) -> str:
    t = int(round(float(t_s)))
    m, s = divmod(t, 60)
    return f"{m}:{s:02d}"


def _wkey(section_id: int, field: str) -> str:
    return f"sec_{section_id}_{field}"


def _example_sections_from_script() -> List[BrewSection]:
    return [
        BrewSection(duration_s=10, mode="flow", value=5.0, label="Add bloom water"),
        BrewSection(duration_s=35, mode="wait", value=0.0, label="Bloom"),
        BrewSection(duration_s=15, mode="weight_delta", value=50.0, label="Pour 50g"),
        BrewSection(duration_s=10, mode="wait", value=0.0, label="Wait"),
        BrewSection(duration_s=10, mode="weight_delta", value=50.0, label="Pour 50g"),
        BrewSection(duration_s=10, mode="wait", value=0.0, label="Wait"),
        BrewSection(duration_s=10, mode="weight_delta", value=50.0, label="Pour 50g"),
        BrewSection(duration_s=10, mode="wait", value=0.0, label="Wait"),
        BrewSection(duration_s=10, mode="weight_delta", value=50.0, label="Pour 50g"),
        BrewSection(duration_s=60, mode="wait", value=0.0, label="Wait for drip-out"),
    ]


def _chart(df: pd.DataFrame) -> alt.Chart:
    base = alt.Chart(df).encode(
        x=alt.X(
            "t_s:Q",
            title="Time (m:ss)",
            axis=alt.Axis(
                format=".0f",
                labelExpr=(
                    "floor(datum.value/60) + ':' + "
                    "(datum.value%60 < 10 ? '0' : '') + "
                    "(datum.value%60)"
                ),
            ),
        )
    )
    c_weight = base.mark_line(color=WEIGHT_COLOR, strokeWidth=2).encode(y=alt.Y("weight_g:Q", title="Weight (g)"))
    c_flow = base.mark_line(color=FLOW_COLOR, strokeWidth=2).encode(
        y=alt.Y("flow_gps:Q", title="Flow (g/s)", axis=alt.Axis(orient="right"))
    )
    return alt.layer(c_weight, c_flow).resolve_scale(y="independent").properties(height=280)


def _build_preview_df(sections: List[BrewSection], *, dt_s: float, initial_weight_g: float) -> pd.DataFrame:
    data = build_reference_json(sections, dt_s=dt_s, initial_weight_g=initial_weight_g)
    weight = data["weight"]
    flow = data["waterFlow"]
    return pd.DataFrame(
        {
            "t_s": [float(w["brew_time"]) for w in weight],
            "weight_g": [float(w["actual_weight"]) for w in weight],
            "flow_gps": [float(f["value"]) for f in flow],
        }
    )


def _sync_sections_from_widgets(sections: List[BrewSection], ids: List[int], *, initial_weight_g: float) -> List[BrewSection]:
    if not sections:
        return []
    summaries = compute_section_summaries(sections, initial_weight_g=initial_weight_g)
    new_sections: List[BrewSection] = []

    for sec, sid, summ in zip(sections, ids, summaries):
        label = st.session_state.get(_wkey(sid, "label"), sec.label)
        mode_label = st.session_state.get(_wkey(sid, "mode"), MODE_TO_LABEL[sec.mode])
        mode = LABEL_TO_MODE.get(mode_label, sec.mode)

        duration = float(st.session_state.get(_wkey(sid, "duration"), sec.duration_s))
        duration = max(duration, 0.1)

        if mode == "wait":
            value = 0.0
        elif mode == "flow":
            value = float(st.session_state.get(_wkey(sid, "flow"), summ["flow_gps"]))
        elif mode == "weight_delta":
            value = float(st.session_state.get(_wkey(sid, "delta"), summ["delta_weight_g"]))
        elif mode == "weight_target":
            value = float(st.session_state.get(_wkey(sid, "end"), summ["end_weight_g"]))
        else:
            value = float(sec.value)

        new_sections.append(BrewSection(duration_s=duration, mode=mode, value=value, label=str(label)))

    return new_sections


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


def _sanitize_filename_base(name: str) -> str:
    bad = '<>:"/\\|?*\n\r\t'
    s = (name or "").strip()
    for ch in bad:
        s = s.replace(ch, "_")
    return s.strip(" .")


def main() -> None:
    st.set_page_config(page_title="Beanconqueror Reference Graph Builder", page_icon="☕", layout="wide")
    _init_state()

    # --- ONLY ONE "warning" mechanism: a real blocking overlay (699px, portrait-ish) ---
    st.markdown(
        """
        <style>
          #bc_overlay {
            display: none;
            position: fixed;
            inset: 0;
            z-index: 999999;
            background: rgba(0,0,0,0.75);
            backdrop-filter: blur(2px);
            -webkit-backdrop-filter: blur(2px);
            align-items: center;
            justify-content: center;
            padding: 24px;
            text-align: center;
          }
          #bc_overlay .box {
            max-width: 520px;
            width: 100%;
            border-radius: 16px;
            padding: 18px 18px;
            border: 1px solid rgba(255,193,7,0.35);
            background: rgba(255,193,7,0.12);
            color: white;
            font-size: 16px;
            line-height: 1.35;
          }
          #bc_overlay .box b { display:block; margin-bottom: 8px; font-size: 17px; }

          @media (max-width: 699px) and (max-aspect-ratio: 1/1) {
            #bc_overlay { display: flex; }
          }
        </style>

        <div id="bc_overlay">
          <div class="box">
            <b>📱 Editor disabled on small portrait screens</b>
            Rotate to <b>landscape</b> or open on <b>desktop</b>.
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.title("Beanconqueror Reference Graph Builder ☕")

    with st.sidebar:
        st.header("Settings")
        dt_s = st.number_input("Sample period dT (s)", min_value=0.01, max_value=2.0, value=0.1, step=0.01)
        initial_weight_g = st.number_input("Initial weight (g)", min_value=0.0, max_value=10_000.0, value=0.0, step=1.0)

        st.markdown("---")

        if st.button("🧹 Clear sections"):
            st.session_state.sections = []
            st.session_state.section_ids = []
            _sync_url_if_needed()
            st.rerun()

        if st.button("✨ Import example"):
            st.session_state.sections = _example_sections_from_script()
            _reset_ids_for_sections()
            _sync_url_if_needed()
            st.rerun()

    sections: List[BrewSection] = st.session_state.sections
    ids: List[int] = list(st.session_state.section_ids)
    _ensure_ids_match_sections()
    sections = st.session_state.sections
    ids = list(st.session_state.section_ids)

    # ---- Plot ----
    if sections:
        try:
            df_preview = _build_preview_df(sections, dt_s=float(dt_s), initial_weight_g=float(initial_weight_g))
            st.altair_chart(_chart(df_preview), use_container_width=True)
            st.caption("Left axis: weight (g). Right axis: flow (g/s).")
        except Exception as e:
            st.error(f"Could not build preview: {e}")
    else:
        st.info("No sections yet. Add one below.")

    st.markdown("---")
    st.header("Profile editor")

    # ---- Existing rows ----
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

            cols[0].text_input("Label", value=sec.label, key=_wkey(sid, "label"), label_visibility="collapsed")

            cols[1].selectbox(
                "Mode",
                options=list(LABEL_TO_MODE.keys()),
                index=list(LABEL_TO_MODE.keys()).index(MODE_TO_LABEL[sec.mode]),
                key=_wkey(sid, "mode"),
                label_visibility="collapsed",
            )

            cols[2].number_input(
                "Duration (s)",
                min_value=0.1,
                max_value=3600.0,
                value=float(sec.duration_s),
                step=0.5,
                key=_wkey(sid, "duration"),
                label_visibility="collapsed",
            )

            cols[3].markdown(_fmt_time_mmss(row["t_start_s"]))
            cols[4].markdown(_fmt_time_mmss(row["t_end_s"]))

            mode_label = st.session_state.get(_wkey(sid, "mode"), MODE_TO_LABEL[sec.mode])
            mode = LABEL_TO_MODE[mode_label]

            if mode == "wait":
                cols[5].markdown("—")
                cols[6].markdown("—")
            elif mode == "flow":
                cols[5].number_input(
                    "Flow (g/s)",
                    min_value=0.0,
                    max_value=1000.0,
                    value=float(row["flow_gps"]),
                    step=0.1,
                    key=_wkey(sid, "flow"),
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
                    key=_wkey(sid, "delta"),
                    label_visibility="collapsed",
                )
            else:  # weight_target
                cols[5].markdown(_fmt(float(row["flow_gps"])))
                cols[6].markdown(_fmt(float(row["delta_weight_g"])))

            cols[7].markdown(_fmt(float(row["start_weight_g"])))

            if mode == "weight_target":
                cols[8].number_input(
                    "End Weight (g)",
                    min_value=0.0,
                    max_value=10_000.0,
                    value=float(row["end_weight_g"]),
                    step=1.0,
                    key=_wkey(sid, "end"),
                    label_visibility="collapsed",
                )
            else:
                cols[8].markdown(_fmt(float(row["end_weight_g"])))

            if cols[9].button("↑", key=f"up_{sid}", width="stretch"):
                new_secs, new_ids = _move_section_pair(sections, ids, i, -1)
                st.session_state.sections = new_secs
                st.session_state.section_ids = new_ids
                _sync_url_if_needed()
                st.rerun()

            if cols[10].button("↓", key=f"down_{sid}", width="stretch"):
                new_secs, new_ids = _move_section_pair(sections, ids, i, +1)
                st.session_state.sections = new_secs
                st.session_state.section_ids = new_ids
                _sync_url_if_needed()
                st.rerun()

            if cols[11].button("🗑️", key=f"del_{sid}", width="stretch"):
                st.session_state.sections = sections[:i] + sections[i + 1 :]
                st.session_state.section_ids = ids[:i] + ids[i + 1 :]
                _sync_url_if_needed()
                st.rerun()

        # Apply edits (won't break ordering now because keys are stable by id)
        new_sections = _sync_sections_from_widgets(sections, ids, initial_weight_g=float(initial_weight_g))
        if new_sections != sections:
            st.session_state.sections = new_sections
            _sync_url_if_needed()
            st.rerun()

    # ---- Add row ----
    st.markdown("---")
    st.subheader("+ Add section")

    add_cols = st.columns([2.0, 2.2, 1.4, 1.3, 1.3])
    add_cols[0].text_input("Label", key="new_label_input")
    add_cols[1].selectbox("Mode", options=list(LABEL_TO_MODE.keys()), key="new_mode_input")
    add_cols[2].number_input("Duration (s)", min_value=0.0, max_value=3600.0, step=0.5, key="new_duration_input")

    new_mode = LABEL_TO_MODE[st.session_state.new_mode_input]

    if new_mode == "wait":
        add_cols[3].markdown("Flow: —")
        add_cols[4].markdown("ΔWeight: —")
    elif new_mode == "flow":
        add_cols[3].number_input("Flow (g/s)", min_value=0.0, max_value=1000.0, step=0.1, key="new_flow_input")
        add_cols[4].markdown("ΔWeight: auto")
    elif new_mode == "weight_delta":
        add_cols[3].markdown("Flow: auto")
        add_cols[4].number_input("ΔWeight (g)", min_value=0.0, max_value=10_000.0, step=1.0, key="new_delta_input")
    else:  # weight_target
        add_cols[3].markdown("Flow: auto")
        add_cols[4].number_input("End Weight (g)", min_value=0.0, max_value=10_000.0, step=1.0, key="new_end_input")

    if st.button("Add section", width="stretch"):
        dur = float(st.session_state.new_duration_input)
        if dur <= 0:
            st.error("Duration must be > 0")
        else:
            if new_mode == "wait":
                sec = BrewSection(duration_s=dur, mode="wait", value=0.0, label=str(st.session_state.new_label_input))
            elif new_mode == "flow":
                sec = BrewSection(duration_s=dur, mode="flow", value=float(st.session_state.new_flow_input), label=str(st.session_state.new_label_input))
            elif new_mode == "weight_delta":
                sec = BrewSection(duration_s=dur, mode="weight_delta", value=float(st.session_state.new_delta_input), label=str(st.session_state.new_label_input))
            else:
                sec = BrewSection(duration_s=dur, mode="weight_target", value=float(st.session_state.new_end_input), label=str(st.session_state.new_label_input))

            st.session_state.sections = st.session_state.sections + [sec]
            st.session_state.section_ids = list(st.session_state.section_ids) + [_new_section_id()]
            st.session_state.reset_add_row = True
            _sync_url_if_needed()
            st.rerun()

    # ---- Export ----
    st.markdown("---")
    st.header("Export JSON")

    if not st.session_state.sections:
        st.info("Add at least one section to export.")
        return

    st.text_input(
        "File name",
        value=str(st.session_state.export_filename),
        key="export_filename",
        help='Name without extension. ".json" is added automatically.',
    )

    out = build_reference_json(
        st.session_state.sections,
        dt_s=float(dt_s),
        start_ms=0,
        realtime_timestampdelta_ms=int(round(float(dt_s) * 1000)),
        initial_weight_g=float(initial_weight_g),
    )
    out_bytes = json.dumps(out, ensure_ascii=False, separators=(",", ":")).encode("utf-8")

    fn_base = _sanitize_filename_base(str(st.session_state.export_filename)) or _default_filename_base()
    fn = fn_base if fn_base.lower().endswith(".json") else (fn_base + ".json")

    st.download_button("📥 Download JSON", data=out_bytes, file_name=fn, mime="application/json", width="stretch")

    with st.expander("JSON structure"):
        st.code(_json_tree(out), language="text")

    # Keep URL updated (refresh-safe) without spamming changes
    _sync_url_if_needed()


if __name__ == "__main__":
    main()