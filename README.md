# Beanconqueror Reference Graph Builder ☕ [![Streamlit app](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://beanconqueror-ref-graph-builder.streamlit.app/)

A lightweight tool to generate **reference brew graphs** for the coffee logging app **Beanconqueror**.

Beanconqueror allows importing graphs as JSON time series (e.g., weight, flow, pressure, temperature), which can be used as a reference in the background while tracking a new brew in real-time. This application provides an interactive editor to build such reference profiles section-by-section and export them in the JSON format expected by Beanconqueror.

It supports:
- **Weight / Flow profile editor**
  - Section-based logic: **Constant**, **Δ Weight**, **Target Weight**, **Flow**
- **Pressure profile editor**
  - Section-based logic: **Constant**, **Δ Value**, **Target**
- **Temperature profile editor**
  - Section-based logic: **Constant**, **Δ Value**, **Target**
- Enable/disable each profile independently (weight/flow, pressure, temperature)
- Automatic computation of the corresponding time series
- Visualization of all enabled profiles on the same graph before export
- Export of JSON compatible with Beanconqueror reference graphs

---

## How to Run

### Option 1: Running online

Go to the Streamlit app:
👉 https://beanconqueror-ref-graph-builder.streamlit.app/

Then:
1. Enable the profiles you want (Weight/Flow, Pressure, Temperature)
2. Add/edit sections
3. Download the exported `.json`
4. Import it into Beanconqueror as a reference graph

### Option 2: Running locally

Clone this repository and set up the environment:

```bash
git clone https://github.com/ozhanozen/beanconqueror-ref-graph-builder
cd beanconqueror-ref-graph-builder
pip install -r requirements.txt