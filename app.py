import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import joblib
import os

# ... your existing setup / load_artifacts / FEATURES / sidebar code ...

# --- keep prediction in session_state so we can render safely ---
if "pred" not in st.session_state:
    st.session_state.pred = None

col1, col2 = st.columns(2)

with col1:
    if st.button("🔮 Predict"):
        # build row from user_vals in the same feature order
        row = pd.DataFrame([[user_vals[f] for f in FEATURES]], columns=FEATURES)
        X_scaled = scaler.transform(row)
        pred = float(model.predict(X_scaled)[0])
        st.session_state.pred = pred  # save for later rendering
        st.success(f"Estimated Power Generated: **{pred:,.0f}** units")

with col2:
    if st.button("✨ Reset to Defaults"):
        st.session_state.pred = None
        st.rerun()  # replaces deprecated st.experimental_rerun()

# ------- render visualization ONLY if a prediction exists -------
if st.session_state.pred is not None:
    pred = st.session_state.pred

    st.subheader("🔆 Power Generation Visualization")

    # 1) Bar chart for predicted power
    fig, ax = plt.subplots(figsize=(6, 1.8))
    ax.barh(["Predicted Power"], [pred])         # simple + theme-friendly
    ax.set_xlabel("Power (Units)")
    ax.set_xlim(0, max(pred * 1.5, 5000))        # auto-scale a bit above pred
    try:
        ax.bar_label(ax.containers[0], fmt='%d', label_type='center')
    except Exception:
        pass
    st.pyplot(fig)

    # 2) Tiny trend line (just a simple visual cue)
    st.markdown("##### 📈 Power Comparison Trend")
    trend_values = np.array([pred * 0.8, pred * 0.9, pred])
    st.line_chart(trend_values)

    # 3) Quick interpretation
    if pred < 2000:
        st.info("⚡ Prediction indicates **low to moderate** power generation for current conditions.")
    elif pred < 4000:
        st.success("🌞 Prediction indicates **good** power generation — favorable sunlight conditions.")
    else:
        st.balloons()
        st.success("🚀 **Excellent** solar output predicted — near-ideal conditions!")
