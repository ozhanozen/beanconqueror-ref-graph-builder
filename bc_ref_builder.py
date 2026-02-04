#!/usr/bin/env python3
"""
Beanconqueror reference graph generator.

Generates timeseries arrays for Beanconqueror reference graphs:
- weight
- waterFlow
- realtimeFlow
- pressureFlow
- temperatureFlow

Behavior:
- Weight target sections are clamped: if target < start_weight, target becomes start_weight (hold).
- Pressure/Temperature targets are NOT clamped (they may decrease).
- old_* fields are set equal to actual_*.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple


def _r(x: float, nd: int = 3) -> float:
    return float(round(float(x), nd))


# ----------------------------
# Time formatting helpers
# ----------------------------
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


def integrate(times: Sequence[float], values: Sequence[float], *, dt_s: float, y0: float = 0.0) -> List[float]:
    if len(times) != len(values):
        raise ValueError("times and values must have same length")
    if not times:
        return []

    ys: List[float] = [float(y0)]
    for k in range(1, len(times)):
        ys.append(ys[-1] + float(values[k - 1]) * float(dt_s))
    return ys


# ----------------------------
# Weight / Flow sections
# ----------------------------
WeightMode = Literal["flow", "weight_delta", "weight_target", "constant"]


@dataclass(frozen=True)
class BrewSection:
    duration_s: float
    mode: WeightMode
    value: float
    label: str = ""

    def validate(self, *, current_weight: float) -> None:
        if self.duration_s <= 0:
            raise ValueError("Section duration must be > 0")

        if self.mode == "flow":
            if self.value < 0:
                raise ValueError("Flow must be >= 0")

        elif self.mode == "constant":
            return

        elif self.mode == "weight_delta":
            if self.value < 0:
                raise ValueError("Added weight must be >= 0")

        elif self.mode == "weight_target":
            return

        else:
            raise ValueError(f"Unknown mode: {self.mode}")


def section_to_flow_gps(section: BrewSection, *, start_weight: float) -> float:
    section.validate(current_weight=start_weight)

    if section.mode == "constant":
        return 0.0

    if section.mode == "flow":
        return float(section.value)

    if section.mode == "weight_delta":
        return 0.0 if section.duration_s <= 0 else float(section.value) / float(section.duration_s)

    if section.mode == "weight_target":
        target = max(float(section.value), float(start_weight))  # clamp
        return 0.0 if section.duration_s <= 0 else (target - float(start_weight)) / float(section.duration_s)

    raise ValueError(f"Unknown mode: {section.mode}")


def compute_section_summaries(
    sections: Sequence[BrewSection],
    *,
    initial_weight_g: float = 0.0,
) -> List[Dict[str, Any]]:
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


def build_weight_series(
    sections: Sequence[BrewSection],
    *,
    dt_s: float,
    start_ms: int,
    realtime_timestampdelta_ms: int,
    initial_weight_g: float,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], List[float]]:
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

    for i, (ti, f, w) in enumerate(zip(times, flows, weights)):
        ms = start_ms + int(round(ti * 1000))
        ts = _format_timestamp_ms(ms)

        actual_weight = _r(float(w))
        old_weight = _r(float(prev_w)) if i > 0 else float(actual_weight)

        bt_fixed = _brew_time_fixed(ti, 1)

        # IMPORTANT: calculated_real_flow intentionally NOT included
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
                "brew_time": _brew_time_compact(ti, 1),
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

    return weight_series, waterflow_series, realtimeflow_series, times


# ----------------------------
# Pressure / Temperature profiles
# ----------------------------
PTMode = Literal["constant", "delta", "target"]


@dataclass(frozen=True)
class PTSection:
    duration_s: float
    mode: PTMode
    value: float
    label: str = ""

    def validate(self) -> None:
        if self.duration_s <= 0:
            raise ValueError("Section duration must be > 0")


def _pt_rate(section: PTSection, *, start_value: float) -> float:
    section.validate()

    if section.mode == "constant":
        return 0.0

    if section.mode == "delta":
        return float(section.value) / float(section.duration_s) if section.duration_s > 0 else 0.0

    if section.mode == "target":
        target = float(section.value)
        return (target - float(start_value)) / float(section.duration_s) if section.duration_s > 0 else 0.0

    raise ValueError(f"Unknown PT mode: {section.mode}")


def compute_pt_summaries(
    sections: Sequence[PTSection],
    *,
    initial_value: float = 0.0,
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    t0 = 0.0
    v = float(initial_value)

    for idx, s in enumerate(sections, start=1):
        rate = _pt_rate(s, start_value=v)
        v_end = v + rate * s.duration_s
        delta = v_end - v

        out.append(
            {
                "idx": idx,
                "label": s.label,
                "mode": s.mode,
                "duration_s": float(s.duration_s),
                "rate_per_s": float(rate),
                "delta_value": float(delta),
                "start_value": float(v),
                "end_value": float(v_end),
                "t_start_s": float(t0),
                "t_end_s": float(t0 + s.duration_s),
            }
        )
        t0 += s.duration_s
        v = v_end

    return out


def build_pt_series(
    sections: Sequence[PTSection],
    *,
    dt_s: float,
    start_ms: int,
    initial_value: float,
    kind: Literal["pressure", "temperature"],
) -> Tuple[List[Dict[str, Any]], List[float], List[float]]:
    times: List[float] = []
    rates: List[float] = []

    t = 0.0
    v_start = float(initial_value)

    for sec in sections:
        rate = _pt_rate(sec, start_value=v_start)
        n = int(round(sec.duration_s / dt_s))
        end_t = t + sec.duration_s

        for i in range(n):
            ti = t + i * dt_s
            if ti >= end_t - 1e-12:
                break
            times.append(round(ti, 10))
            rates.append(float(rate))

        v_start = v_start + rate * sec.duration_s
        t = end_t

    total_dur = sum(s.duration_s for s in sections)
    if not times or abs(times[-1] - total_dur) > 1e-9:
        times.append(round(total_dur, 10))
        rates.append(0.0)

    values = integrate(times, rates, dt_s=dt_s, y0=initial_value)

    out: List[Dict[str, Any]] = []
    prev_val = values[0] if values else float(initial_value)

    for i, (ti, val) in enumerate(zip(times, values)):
        ms = start_ms + int(round(ti * 1000))
        ts = _format_timestamp_ms(ms)
        bt = _brew_time_fixed(ti, 1)

        actual = _r(float(val))
        old = _r(float(prev_val)) if i > 0 else float(actual)

        if kind == "pressure":
            out.append({"actual_pressure": actual, "old_pressure": old, "brew_time": bt, "timestamp": ts})
        else:
            out.append({"actual_temperature": actual, "old_temperature": old, "brew_time": bt, "timestamp": ts})

        prev_val = actual

    return out, times, values


# ----------------------------
# Main builder
# ----------------------------
def build_reference_json(
    *,
    dt_s: float = 0.1,
    start_ms: int = 0,
    realtime_timestampdelta_ms: Optional[int] = None,
    enable_weight: bool = True,
    enable_pressure: bool = False,
    enable_temperature: bool = False,
    weight_sections: Sequence[BrewSection] = (),
    pressure_sections: Sequence[PTSection] = (),
    temperature_sections: Sequence[PTSection] = (),
    initial_weight_g: float = 0.0,
    initial_pressure: float = 0.0,
    initial_temperature: float = 0.0,
) -> Dict[str, Any]:
    if dt_s <= 0:
        raise ValueError("dt_s must be > 0")

    if realtime_timestampdelta_ms is None:
        realtime_timestampdelta_ms = int(round(dt_s * 1000))

    weight_arr: List[Dict[str, Any]] = []
    waterflow_arr: List[Dict[str, Any]] = []
    realtimeflow_arr: List[Dict[str, Any]] = []
    pressure_arr: List[Dict[str, Any]] = []
    temp_arr: List[Dict[str, Any]] = []

    if enable_weight and weight_sections:
        weight_arr, waterflow_arr, realtimeflow_arr, _ = build_weight_series(
            weight_sections,
            dt_s=dt_s,
            start_ms=start_ms,
            realtime_timestampdelta_ms=int(realtime_timestampdelta_ms),
            initial_weight_g=float(initial_weight_g),
        )

    if enable_pressure and pressure_sections:
        pressure_arr, _, _ = build_pt_series(
            pressure_sections,
            dt_s=dt_s,
            start_ms=start_ms,
            initial_value=float(initial_pressure),
            kind="pressure",
        )

    if enable_temperature and temperature_sections:
        temp_arr, _, _ = build_pt_series(
            temperature_sections,
            dt_s=dt_s,
            start_ms=start_ms,
            initial_value=float(initial_temperature),
            kind="temperature",
        )

    return {
        "weight": weight_arr,
        "weightSecond": [],
        "waterFlow": waterflow_arr,
        "realtimeFlow": realtimeflow_arr,
        "realtimeFlowSecond": [],
        "pressureFlow": pressure_arr,
        "temperatureFlow": temp_arr,
        "brewbyweight": [],
    }