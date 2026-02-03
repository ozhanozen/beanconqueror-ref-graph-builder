# Beanconqueror Reference Graph Builder ☕ [![Streamlit app](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://beanconqueror-ref-graph-builder.streamlit.app/)

A lightweight tool to generate **reference brew graphs** for the coffee logging app **Beanconqueror**.

Beanconqueror allows importing graphs as JSON time series (e.g., weight and flow) which can be used as a reference in the background while tracking new brew in real-time. This application provides an interactive editor to build such reference profiles section-by-section and export them in the JSON format expected by Beanconqueror.
* It supports section-based brewing logic (flow, weight delta, weight target, wait).
* It automatically computes the corresponding flow/weight time series.
* It visualizes the resulting weight and flow curves before export.
* It exports JSON compatible with Beanconqueror reference graphs.

---

## How to Run

**Option 1: Running online:**

Go to the [Streamlit app link](https://beanconqueror-ref-graph-builder.streamlit.app/)
 and follow the instructions.

**Option 2: Running locally:**

Clone this repository and set up the environment:
```bash
git clone https://github.com/ozhanozen/beanconqueror-ref-graph-builder
cd beanconqueror-ref-graph-builder
pip install -r requirements.txt