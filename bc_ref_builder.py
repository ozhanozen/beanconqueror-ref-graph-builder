#!/usr/bin/env python3
"""
Beanconqueror reference graph generator.

Generates weight/flow related timeseries sections for Beanconqueror reference graphs:
- weight
- waterFlow
- realtimeFlow

Other arrays are left empty.

Behavior:
- "Weight target" sections are clamped: if target < start_weight, target becomes start_weight (hold).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Sequence


def _r(x: float, nd: int = 3) -> float:
    return float(round(float(x), nd))


SectionMode = Literal["flow", "weight_delta", "weight_target", "wait"]


@dataclass(frozen=True)
class BrewSection:
    duration_s: float
    mode: SectionMode
    value: float
    label: str = ""

    def validate(self, *, current_weight: float) -> None:
        if self.duration_s <= 0:
            raise ValueError("Section duration must be > 0")

        if self.mode == "flow":
            if self.value < 0:
                raise ValueError("Flow must be >= 0")

        elif self.mode == "wait":
            return

        elif self.mode == "weight_delta":
            if self.value < 0:
                raise ValueError("Added weight must be >= 0")

        elif self.mode == "weight_target":
            # target will be clamped in computation
            return

        else:
            raise ValueError(f"Unknown mode: {self.mode}")


def _format_timestamp_ms(ms_total: int) -> str:
    if ms_total < 0:
        ms_total = 0
    s_total, ms = divmod(ms_total, 1000)
    h, rem = divmod(s_total, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


def _brew_time_fixed(t: float, decimals: int = 1) -> str:
    return f"{t:.{decimals}f}"


def _brew_time_compact(t: float, decimals: int = 1) -> str:
    s = f"{t:.{decimals}f}"
    if "." in s:
        s = s.rstrip("0").rstrip(".")
    return s


def section_to_flow_gps(section: BrewSection, *, start_weight: float) -> float:
    section.validate(current_weight=start_weight)

    if section.mode == "wait":
        return 0.0

    if section.mode == "flow":
        return float(section.value)

    if section.mode == "weight_delta":
        if section.duration_s <= 0:
            return 0.0
        return float(section.value) / float(section.duration_s)

    if section.mode == "weight_target":
        # Clamp target to start weight
        target = max(float(section.value), float(start_weight))
        if section.duration_s <= 0:
            return 0.0
        return (target - float(start_weight)) / float(section.duration_s)

    raise ValueError(f"Unknown mode: {section.mode}")


def integrate(times: Sequence[float], values: Sequence[float], *, dt_s: float, y0: float = 0.0) -> List[float]:
    if len(times) != len(values):
        raise ValueError("times and values must have same length")
    if not times:
        return []

    ys: List[float] = [float(y0)]
    for k in range(1, len(times)):
        ys.append(ys[-1] + float(values[k - 1]) * float(dt_s))
    return ys


def compute_section_summaries(
    sections: Sequence[BrewSection],
    *,
    initial_weight_g: float = 0.0,
) -> List[Dict[str, Any]]:
    """
    Compute per-section derived values for UI:
    - start_weight_g
    - end_weight_g
    - flow_gps
    - delta_weight_g
    """
    out: List[Dict[str, Any]] = []
    t0 = 0.0
    w = float(initial_weight_g)

    for idx, s in enumerate(sections, start=1):
        flow = section_to_flow_gps(s, start_weight=w)
        w_end = w + flow * s.duration_s
        delta = w_end - w

        out.append(
            {
                "idx": idx,
                "label": s.label,
                "mode": s.mode,
                "duration_s": float(s.duration_s),
                "flow_gps": float(flow),
                "delta_weight_g": float(delta),
                "start_weight_g": float(w),
                "end_weight_g": float(w_end),
                "t_start_s": float(t0),
                "t_end_s": float(t0 + s.duration_s),
            }
        )

        t0 += s.duration_s
        w = w_end

    return out


def build_reference_json(
    sections: Sequence[BrewSection],
    *,
    dt_s: float = 0.1,
    start_ms: int = 0,
    realtime_timestampdelta_ms: Optional[int] = None,
    initial_weight_g: float = 0.0,
) -> Dict[str, Any]:
    if dt_s <= 0:
        raise ValueError("dt_s must be > 0")
    if realtime_timestampdelta_ms is None:
        realtime_timestampdelta_ms = int(round(dt_s * 1000))

    times: List[float] = []
    flows: List[float] = []

    t = 0.0
    w_start = float(initial_weight_g)

    for sec in sections:
        flow = section_to_flow_gps(sec, start_weight=w_start)
        n = int(round(sec.duration_s / dt_s))
        end_t = t + sec.duration_s

        for i in range(n):
            ti = t + i * dt_s
            if ti >= end_t - 1e-12:
                break
            times.append(round(ti, 10))
            flows.append(float(flow))

        w_start = w_start + flow * sec.duration_s
        t = end_t

    total_dur = sum(s.duration_s for s in sections)
    if not times or abs(times[-1] - total_dur) > 1e-9:
        times.append(round(total_dur, 10))
        flows.append(0.0)

    weights = integrate(times, flows, dt_s=dt_s, y0=initial_weight_g)

    weight_series: List[Dict[str, Any]] = []
    waterflow_series: List[Dict[str, Any]] = []
    realtimeflow_series: List[Dict[str, Any]] = []

    prev_w = weights[0] if weights else float(initial_weight_g)

    for i, (t, f, w) in enumerate(zip(times, flows, weights)):
        ms = start_ms + int(round(t * 1000))
        ts = _format_timestamp_ms(ms)

        actual_weight = _r(float(w))
        old_weight = _r(float(prev_w)) if i > 0 else float(actual_weight)

        bt_fixed = _brew_time_fixed(t, 1)

        weight_series.append(
            {
                "timestamp": ts,
                "brew_time": bt_fixed,
                "actual_weight": actual_weight,
                "old_weight": old_weight,
                "actual_smoothed_weight": actual_weight,
                "old_smoothed_weight": old_weight,
                "not_mutated_weight": actual_weight,
            }
        )

        waterflow_series.append(
            {
                "brew_time": _brew_time_compact(t, 1),
                "timestamp": ts,
                "value": _r(float(f)),
            }
        )

        realtimeflow_series.append(
            {
                "brew_time": bt_fixed,
                "timestamp": ts,
                "smoothed_weight": actual_weight,
                "timestampdelta": 0 if i == 0 else int(realtime_timestampdelta_ms),
                "flow_value": _r(float(f)),
            }
        )

        prev_w = actual_weight

    return {
        "weight": weight_series,
        "weightSecond": [],
        "waterFlow": waterflow_series,
        "realtimeFlow": realtimeflow_series,
        "realtimeFlowSecond": [],
        "pressureFlow": [],
        "temperatureFlow": [],
        "brewbyweight": [],
    }