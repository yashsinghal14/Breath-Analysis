# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import io
import sys, types
from datetime import datetime

st.set_page_config(page_title="BreathScan — Cancer Risk Prediction", layout="wide", initial_sidebar_state="expanded")

# --------------------------
# Utilities & compatibility
# --------------------------
def _patch_numpy_core_for_joblib():
    """
    Safely patch numpy module map so joblib can unpickle models
    saved on environments where numpy introduced numpy._core.
    This is an in-memory alias, safe across environments.
    """
    import numpy as _np, sys as _sys, types as _types
    if not hasattr(_np, "_core"):
        # Create module alias pointing to numpy.core (in-memory)
        _np._core = _types.SimpleNamespace()
        _sys.modules['numpy._core'] = _np.core

def load_pickle_model(path):
    """
    Load joblib model, applying compatibility fixes where necessary.
    """
    try:
        return joblib.load(path)
    except ModuleNotFoundError as ex:
        # Try patching and reloading (common numpy._core mismatch)
        _patch_numpy_core_for_joblib()
        return joblib.load(path)

@st.cache_resource(show_spinner=False)
def load_model_and_cv(model_path="best_model.pkl", cv_path="cv_summary.csv"):
    model = load_pickle_model(model_path)
    cv = None
    try:
        cv = pd.read_csv(cv_path)
    except Exception:
        cv = None
    return model, cv

def safe_get_feature_names(preprocessor, fallback_cols=None):
    """
    Try to derive transformed feature names. If not possible, return fallback_cols.
    """
    try:
        # scikit-learn ColumnTransformer.get_feature_names_out exists in modern versions
        return list(preprocessor.get_feature_names_out())
    except Exception:
        # fallback: return the original columns (if provided)
        return list(fallback_cols) if fallback_cols is not None else []

def model_feature_importances(model, original_columns=None):
    """
    Extract feature importances for display. Supports:
    - coef_ (linear models)
    - feature_importances_ (tree-based)
    Returns a pd.Series sorted descending (index = feature names).
    """
    try:
        estimator = model.named_steps['clf']
        pre = model.named_steps.get('pre', None)
        # try to build feature names
        if pre is not None and original_columns is not None:
            feat_names = safe_get_feature_names(pre, fallback_cols=original_columns)
        else:
            feat_names = original_columns if original_columns is not None else []

        # If selector used, reduce names
        if 'sel' in model.named_steps and model.named_steps['sel'] != 'passthrough' and feat_names:
            try:
                mask = model.named_steps['sel'].get_support()
                feat_names = np.array(feat_names)[mask].tolist()
            except Exception:
                pass

        if hasattr(estimator, "coef_"):
            coeff = estimator.coef_
            # multiclass: average absolute across classes
            if coeff.ndim > 1:
                vals = np.mean(np.abs(coeff), axis=0)
            else:
                vals = np.abs(coeff)
            n = len(vals)
            feat_names = feat_names[:n] if feat_names else [f"f_{i}" for i in range(n)]
            return pd.Series(vals, index=feat_names).sort_values(ascending=False)
        elif hasattr(estimator, "feature_importances_"):
            vals = estimator.feature_importances_
            n = len(vals)
            feat_names = feat_names[:n] if feat_names else [f"f_{i}" for i in range(n)]
            return pd.Series(vals, index=feat_names).sort_values(ascending=False)
        else:
            return None
    except Exception:
        return None

def get_prediction_and_probs(model, X):
    """
    Predict classes and class probabilities (if available).
    Returns DataFrame with Predicted_Label and columns Prob_<label>.
    """
    preds = model.predict(X)
    out = pd.DataFrame({"Predicted_Label": preds})
    try:
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X)
            classes = model.named_steps['clf'].classes_
            for i, c in enumerate(classes):
                out[f"Prob_{c}"] = probs[:, i]
    except Exception:
        pass
    return out

def per_sample_feature_contributions(model, X_row):
    """
    For linear models, compute per-feature contribution = coef * value.
    For multiclass, use the class of interest (Cancer) if present and available.
    Returns a sorted pandas Series of top positive/negative contributors.
    """
    try:
        estimator = model.named_steps['clf']
        pre = model.named_steps.get('pre', None)
        if not hasattr(estimator, "coef_") or pre is None:
            return None
        # Extract feature names used by the model
        original_cols = getattr(pre, "feature_names_in_", None)
        feat_names = safe_get_feature_names(pre, fallback_cols=original_cols)
        if 'sel' in model.named_steps and model.named_steps['sel'] != 'passthrough':
            mask = model.named_steps['sel'].get_support()
            feat_names = np.array(feat_names)[mask]
        # Transform the row through preprocessor to get numeric vector
        X_trans = pre.transform(X_row)
        coefs = estimator.coef_
        # If multiclass, find index of 'Cancer' if exists, else average
        if coefs.ndim > 1:
            classes = estimator.classes_
            if "Cancer" in classes:
                idx = list(classes).index("Cancer")
                coef_vec = coefs[idx]
            else:
                coef_vec = np.mean(coefs, axis=0)
        else:
            coef_vec = coefs
        contributions = np.array(X_trans).ravel() * np.array(coef_vec).ravel()
        # Align to feature names (trim/pad)
        n = len(contributions)
        feat_names = feat_names[:n] if feat_names else [f"f_{i}" for i in range(n)]
        series = pd.Series(contributions, index=feat_names)
        # sort absolute impact
        series_sorted = series.sort_values(key=lambda s: np.abs(s), ascending=False)
        return series_sorted
    except Exception:
        return None

# --------------------------
# App: layout and logic
# --------------------------
# Header
st.title("BreathScan — Breath Analysis Cancer Risk (Prototype)")
st.caption("Clinical-use prototype. Not a medical device. For research & decision support only.")

# Sidebar: model load + metadata + threshold
with st.sidebar:
    st.header("Model & Settings")
    model_path = st.text_input("Model file path", value="best_model.pkl")
    cv_path = st.text_input("CV summary CSV", value="cv_summary.csv")
    model, cv_summary = load_model_and_cv(model_path, cv_path)
    st.write("Loaded model type:", type(model).__name__)
    if cv_summary is not None:
        st.markdown("**Cross-validation summary**")
        st.dataframe(cv_summary)
    st.markdown("---")
    st.write("Prediction settings")
    threshold = st.slider("Cancer probability threshold (for flagging risk)", min_value=0.0, max_value=1.0, value=0.50, step=0.01)
    batch_display_count = st.slider("Show first N predictions in table", min_value=5, max_value=200, value=25, step=5)
    st.markdown("---")
    st.markdown("Upload notes:")
    st.markdown("- Upload CSV with the same feature columns used during model training.")
    st.markdown("- The model pipeline handles missing values; however, column names must match.")
    st.markdown("---")
    st.markdown("Privacy & clinical use")
    st.write("We do not store uploads beyond this session. Do not upload identifiable patient info without proper consent.")
    st.write("If this will be used clinically, perform local validation on your site and consult regulatory guidance.")

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["Predict", "Feature Importance", "Explain a Prediction", "About & Deploy"])

# ----- TAB: Predict -----
with tab1:
    st.header("Make Predictions")
    st.write("Upload a CSV file containing samples (each row = one sample). The pipeline expects the same feature column names used for training.")
    uploaded_file = st.file_uploader("Upload CSV for prediction", type=["csv"], accept_multiple_files=False)
    st.write("Or drag & drop a single-row CSV with patient features for one-person prediction.")
    sample_mode = st.checkbox("Single-sample quick entry (paste CSV content)", value=False)

    input_df = None
    if uploaded_file is not None:
        try:
            input_df = pd.read_csv(uploaded_file)
            st.success(f"Uploaded dataset with shape {input_df.shape}")
        except Exception as e:
            st.error(f"Failed to read uploaded CSV: {e}")
    elif sample_mode:
        st.info("Paste a single-row CSV (headers + one data row) into the box below.")
        sample_text = st.text_area("Paste CSV here", height=160)
        if st.button("Load pasted CSV"):
            try:
                input_df = pd.read_csv(io.StringIO(sample_text))
                st.success(f"Pasted single sample loaded (shape={input_df.shape})")
            except Exception as e:
                st.error(f"Could not parse CSV text: {e}")

    if input_df is not None:
        st.subheader("Preview uploaded data (first rows)")
        st.dataframe(input_df.head(batch_display_count))

        # Try running predictions
        try:
            pred_df = get_prediction_and_probs(model, input_df)
            # Ensure we use 'Cancer' probability as the risk score if multiclass
            risk_col = None
            if any(col.startswith("Prob_") for col in pred_df.columns):
                prob_cols = [c for c in pred_df.columns if c.startswith("Prob_")]
                # find a 'Prob_Cancer' if present else choose max probability
                if "Prob_Cancer" in pred_df.columns:
                    pred_df["Cancer_Risk"] = pred_df["Prob_Cancer"]
                else:
                    # For multiclass where 'Cancer' not labeled, find class name containing 'Cancer' or fallback to max prob
                    cancer_cols = [c for c in prob_cols if "cancer" in c.lower()]
                    if cancer_cols:
                        pred_df["Cancer_Risk"] = pred_df[cancer_cols[0]]
                    else:
                        pred_df["Cancer_Risk"] = pred_df[prob_cols].max(axis=1)
                risk_col = "Cancer_Risk"
            else:
                pred_df["Cancer_Risk"] = np.nan

            # Determine binary flag using threshold
            pred_df["Cancer_Flag"] = pred_df["Cancer_Risk"].apply(lambda p: "High Risk" if pd.notna(p) and p >= threshold else "Low Risk/No Flag")

            # Combine with input for download
            results = pd.concat([input_df.reset_index(drop=True), pred_df.reset_index(drop=True)], axis=1)

            # Display
            st.subheader("Prediction Results")
            st.dataframe(results.head(batch_display_count))

            # Summary metrics
            n_high = (pred_df["Cancer_Flag"] == "High Risk").sum()
            st.markdown(f"**Samples flagged as High Risk (≥ {threshold:.2f}):** {n_high} / {len(pred_df)}")

            # Visual: probability distribution
            if "Cancer_Risk" in pred_df.columns and pred_df["Cancer_Risk"].notna().any():
                fig, ax = plt.subplots(figsize=(8, 3))
                sns.histplot(pred_df["Cancer_Risk"].dropna(), bins=25, ax=ax, kde=True)
                ax.axvline(threshold, color="red", linestyle="--", label=f"Threshold = {threshold:.2f}")
                ax.set_title("Cancer Risk Distribution (per sample)")
                ax.set_xlabel("Risk (probability)")
                ax.legend()
                st.pyplot(fig)

            # Download results
            towrite = io.BytesIO()
            results.to_csv(towrite, index=False)
            towrite.seek(0)
            st.download_button("Download results (CSV)", data=towrite, file_name="predictions_results.csv", mime="text/csv")

        except Exception as e:
            st.error(f"Prediction failed: {e}")
            st.exception(e)

# ----- TAB: Feature Importance -----
with tab2:
    st.header("Model Feature Importance")
    st.write("Feature importance is derived from the trained model. For linear models (e.g., LogisticRegression) we show absolute coefficients. For tree models, we show feature importances.")
    try:
        # Attempt to extract original feature columns if available
        pre = model.named_steps.get('pre', None)
        original_cols = None
        if pre is not None and hasattr(pre, "feature_names_in_"):
            original_cols = list(pre.feature_names_in_)
        fi = model_feature_importances(model, original_columns=original_cols)
        if fi is not None and not fi.empty:
            st.subheader("Top 30 Features")
            st.dataframe(pd.DataFrame({"Feature": fi.index, "Importance": fi.values}).head(30))
            # Plot top 20
            top = fi.head(20)
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.barh(top.index[::-1], top.values[::-1])
            ax.set_xlabel("Importance (abs coef or importance)")
            ax.set_title("Top 20 model features")
            st.pyplot(fig)
        else:
            st.warning("Feature importance not available for this model.")
    except Exception as e:
        st.error("Could not compute feature importances.")
        st.exception(e)

# ----- TAB: Explain a single prediction -----
with tab3:
    st.header("Explain a single sample prediction (feature contributions)")
    st.write("Paste a single-row CSV (headers + one row) with the same feature columns and run 'Explain'. This shows top positive and negative contributors if the model exposes coefficients.")
    explain_text = st.text_area("Paste single-row CSV here", height=140)
    if st.button("Explain pasted sample"):
        if not explain_text.strip():
            st.warning("Please paste CSV content for one sample (headers + one row).")
        else:
            try:
                sample_df = pd.read_csv(io.StringIO(explain_text))
                contributions = per_sample_feature_contributions(model, sample_df)
                if contributions is None:
                    st.warning("Per-feature contributions are not available for this model (not a linear model or missing preprocessor).")
                else:
                    st.subheader("Top contributing features (absolute impact)")
                    display_df = pd.DataFrame({
                        "Feature": contributions.index,
                        "Contribution": contributions.values
                    })
                    st.dataframe(display_df.head(30))
                    # Show positive vs negative contributions
                    pos = contributions[contributions > 0].head(10)
                    neg = contributions[contributions < 0].head(10)
                    if not pos.empty:
                        st.markdown("**Top positive contributions (increase Cancer score):**")
                        st.dataframe(pos.head(10).to_frame("Contribution"))
                    if not neg.empty:
                        st.markdown("**Top negative contributions (decrease Cancer score):**")
                        st.dataframe(neg.head(10).to_frame("Contribution"))
            except Exception as e:
                st.error(f"Could not parse or explain sample: {e}")

# ----- TAB: About & Deploy -----
with tab4:
    st.header("About this App")
    st.markdown("""
**BreathScan** is a prototype web interface for a breath-analysis model that estimates probability of *Cancer* from breath features.
- **Model:** loaded from `best_model.pkl` (a scikit-learn pipeline)
- **Date:** {}
- **Intended use:** research, triage support, and exploratory analysis only — **not** a diagnostic device.
    """.format(datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")))

    st.subheader("How it works (brief)")
    st.markdown("""
1. Upload a CSV that contains the breath features (same column names used during training).
2. The pipeline handles missing values and appropriate preprocessing.
3. The app shows class probabilities and flags samples above the configurable threshold.
4. For linear models the app shows feature contributions per sample to help interpret why a prediction was made.
    """)

    st.subheader("Deployment & security recommendations")
    st.markdown("""
- Run this app **on-premises** or in a secure cloud VPC if you will upload patient data.
- Use HTTPS and authentication (Streamlit Cloud / SSO / custom auth) if used by clinicians.
- Maintain model versioning and dataset provenance; keep an audit trail of predictions if used in regulated workflows.
    """)

    st.subheader("Next steps & validation")
    st.markdown("""
- **Local validation:** validate the model on local cohorts before any clinical use.
- **Calibration:** check model calibration (reliability) and consider Platt scaling or isotonic calibration if needed.
- **Explainability:** consider SHAP for non-linear models to show per-sample explanations more accurately.
    """)
    st.markdown("**Clinical disclaimer:** This tool is for research only and should not be used as a sole basis for clinical decisions. Consult clinical experts and regulatory guidelines before deployment.")

# Footer
st.markdown("---")
st.caption("Built with care — Machine Learning & Healthcare best practices applied. ⚕️")

