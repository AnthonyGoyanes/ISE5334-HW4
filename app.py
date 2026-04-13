"""
app.py
Streamlit application for Manufacturing Feasibility Classification.
Plug-and-play pipeline runner across 60 configurable ML pipelines.
"""
import os
import sys

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Ensure the project root is on the path
sys.path.insert(0, os.path.dirname(__file__))

from pipeline.catalog import (
    ALL_GROUPS,
    PIPELINE_CATALOG,
    PIPELINE_CATALOG_DICT,
    PIPELINE_GROUPS,
)

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Feasibility Classifier",
    page_icon="🔧",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
for _key in ("data", "results", "last_params"):
    if _key not in st.session_state:
        st.session_state[_key] = None


# ---------------------------------------------------------------------------
# Sidebar — Configuration
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("🔧 Configuration")

    # ── Data ─────────────────────────────────────────────────────────────────
    st.subheader("1 · Upload Point Clouds")

    feasible_files = st.file_uploader(
        "Feasible parts (.ply)",
        type=["ply"],
        accept_multiple_files=True,
        help="Upload all feasible part .ply files (label = 1)"
    )

    infeasible_files = st.file_uploader(
        "Infeasible parts (.ply)",
        type=["ply"],
        accept_multiple_files=True,
        help="Upload all infeasible part .ply files (label = 0)"
    )

    load_btn = st.button(
        "📂  Load Dataset",
        type="primary",
        use_container_width=True,
        disabled=(not feasible_files or not infeasible_files)
    )

    st.divider()

    # ── Pipeline selection ───────────────────────────────────────────────────
    st.subheader("2 · Pipeline Selection")
    run_mode = st.radio(
        "Run mode",
        ["All 60 Pipelines", "By Group", "Custom"],
        index=0,
    )

    if run_mode == "By Group":
        sel_group = st.selectbox("Group", ALL_GROUPS)
        pipeline_ids = PIPELINE_GROUPS[sel_group]
    elif run_mode == "Custom":
        all_ids = [p.id for p in PIPELINE_CATALOG]
        pipeline_ids = st.multiselect(
            "Select pipeline IDs",
            options=all_ids,
            default=["P-001", "P-003", "P-009", "P-020"],
            help="Select one or more pipelines to run.",
        )
    else:
        pipeline_ids = [p.id for p in PIPELINE_CATALOG]

    st.caption(f"**{len(pipeline_ids)}** pipeline(s) selected.")

    st.divider()

    # ── Advanced parameters ──────────────────────────────────────────────────
    st.subheader("3 · Parameters")
    with st.expander("Preprocessing"):
        ransac_dist = st.slider(
            "RANSAC distance threshold", 0.005, 0.10, 0.02, 0.005,
            help="Points within this distance of the fitted plane are considered inliers.",
        )
        ransac_iter = st.slider("RANSAC iterations", 100, 2000, 1000, 100)
        voxel_size  = st.slider("Voxel downsample size", 0.01, 0.20, 0.05, 0.01)
        dbscan_eps  = st.slider("DBSCAN ε (component linking)", 0.02, 0.30, 0.10, 0.01)

    with st.expander("Autoencoder (AE pipelines)"):
        ae_n_pts  = st.slider("Points sampled per cloud", 64, 512, 256, 64)
        ae_latent = st.slider("Latent dimension", 16, 128, 64, 16)
        ae_epochs = st.slider("Training epochs", 10, 200, 50, 10)

    with st.expander("Experiment"):
        test_size = st.slider("Test set fraction", 0.15, 0.40, 0.30, 0.05)
        rand_seed = st.number_input("Random seed", value=42, step=1)

    params = {
        "ransac_distance_threshold": ransac_dist,
        "ransac_iterations":         ransac_iter,
        "voxel_size":                voxel_size,
        "dbscan_eps":                dbscan_eps,
        "ae_n_points":               ae_n_pts,
        "ae_latent_dim":             ae_latent,
        "ae_epochs":                 ae_epochs,
        "test_size":                 test_size,
        "random_seed":               rand_seed,
    }

    st.divider()
    run_btn = st.button(
        "▶  Run Pipelines",
        type="primary",
        use_container_width=True,
        disabled=(st.session_state.data is None or len(pipeline_ids) == 0),
    )

# ---------------------------------------------------------------------------
# Load dataset
# ---------------------------------------------------------------------------
if load_btn and feasible_files and infeasible_files:
    import tempfile, os, trimesh

    def load_uploaded_files(uploaded_files, label):
        clouds, labels, filenames = [], [], []
        for f in uploaded_files:
            try:
                # Write to a temp file so trimesh can read it
                suffix = os.path.splitext(f.name)[1]
                with tempfile.NamedTemporaryFile(
                    delete=False, suffix=suffix
                ) as tmp:
                    tmp.write(f.read())
                    tmp_path = tmp.name
                obj = trimesh.load(tmp_path, process=False)
                os.unlink(tmp_path)  # clean up immediately
                if hasattr(obj, "vertices"):
                    pts = np.array(obj.vertices, dtype=np.float32)
                else:
                    pts = np.vstack(
                        [g.vertices for g in obj.geometry.values()]
                    ).astype(np.float32)
                clouds.append(pts)
                labels.append(label)
                filenames.append(f.name)
            except Exception as exc:
                st.warning(f"Could not load {f.name}: {exc}")
        return clouds, labels, filenames

    with st.spinner("Loading uploaded files…"):
        c1, l1, f1 = load_uploaded_files(feasible_files,   label=1)
        c2, l2, f2 = load_uploaded_files(infeasible_files, label=0)
        all_clouds    = c1 + c2
        all_labels    = np.array(l1 + l2, dtype=np.int32)
        all_filenames = f1 + f2
        st.session_state.data = (all_clouds, all_labels, all_filenames)

    st.sidebar.success(
        f"✅ Loaded **{len(all_clouds)}** clouds  "
        f"({len(c1)} feasible / {len(c2)} infeasible)"
    )
# ---------------------------------------------------------------------------
# Main area — tabs
# ---------------------------------------------------------------------------
tab_data, tab_results, tab_catalog = st.tabs(
    ["📊  Data Overview", "🚀  Pipeline Results", "📋  Pipeline Catalog"]
)

# ── Tab 1: Data overview ──────────────────────────────────────────────────
with tab_data:
    st.header("Dataset Overview")

    if st.session_state.data is None:
        st.info(
            "👈  Enter your feasible / infeasible directories in the sidebar "
            "and click **Load Dataset** to begin."
        )
    else:
        clouds, labels, filenames = st.session_state.data
        from pipeline.ingestion import dataset_summary
        s = dataset_summary(clouds, labels, filenames)

        # KPI row
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Total clouds",   s["n_total"])
        c2.metric("Feasible",       s["n_feasible"])
        c3.metric("Infeasible",     s["n_infeasible"])
        c4.metric("Avg pts/cloud",  f"{s['avg_points']:.0f}")
        c5.metric("Imbalance ratio",f"{s['n_feasible'] / max(s['n_infeasible'],1):.2f}:1")

        col_a, col_b = st.columns(2)

        with col_a:
            fig = px.bar(
                x=["Feasible", "Infeasible"],
                y=[s["n_feasible"], s["n_infeasible"]],
                color=["Feasible", "Infeasible"],
                color_discrete_map={"Feasible": "#4FC3F7", "Infeasible": "#F96167"},
                labels={"x": "Class", "y": "Count"},
                title="Class Distribution",
            )
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        with col_b:
            point_counts = [len(c) for c in clouds]
            feasible_counts   = [len(c) for c, l in zip(clouds, labels) if l == 1]
            infeasible_counts = [len(c) for c, l in zip(clouds, labels) if l == 0]
            fig2 = go.Figure()
            fig2.add_trace(go.Histogram(x=feasible_counts,   name="Feasible",   marker_color="#4FC3F7", opacity=0.75))
            fig2.add_trace(go.Histogram(x=infeasible_counts, name="Infeasible", marker_color="#F96167", opacity=0.75))
            fig2.update_layout(
                barmode="overlay",
                title="Point Count Distribution",
                xaxis_title="Points per cloud",
                yaxis_title="Count",
            )
            st.plotly_chart(fig2, use_container_width=True)

        st.subheader("Point Cloud Statistics")
        st.dataframe(
            pd.DataFrame({
                "Statistic": ["Min", "Max", "Mean", "Std Dev"],
                "Points per cloud": [
                    s["min_points"], s["max_points"],
                    f"{s['avg_points']:.1f}", f"{s['std_points']:.1f}",
                ],
            }),
            use_container_width=False,
            hide_index=True,
        )


# ── Tab 2: Pipeline results ───────────────────────────────────────────────
with tab_results:
    st.header("Pipeline Results")

    # Run pipelines
    if run_btn and st.session_state.data is not None and pipeline_ids:
        clouds, labels, _ = st.session_state.data

        progress_bar = st.progress(0.0, text="Initialising…")
        status_text  = st.empty()

        _counters = {"pipeline": 0}

        def _callback(stage: str, detail: str, current: int, total: int):
            if stage == "preprocess_start":
                status_text.markdown(f"⚙️  **Preprocessing** with `{detail}`…")
            elif stage == "ae_train":
                status_text.markdown("🤖  **Training autoencoder** (unsupervised)…")
                progress_bar.progress(0.05, text="Training autoencoder…")
            elif stage == "features":
                status_text.markdown(f"🔢  **Extracting features** `{detail}`…")
            elif stage == "pipeline":
                _counters["pipeline"] += 1
                frac = max(0.1, _counters["pipeline"] / len(pipeline_ids))
                progress_bar.progress(frac, text=f"Running {detail}…")
            elif stage == "done":
                progress_bar.progress(1.0, text="Complete!")
                status_text.markdown("✅  All pipelines complete.")

        from pipeline.runner import run_pipelines

        with st.spinner("Running pipelines — this may take several minutes…"):
            results_df = run_pipelines(
                clouds, labels,
                pipeline_ids=pipeline_ids,
                params=params,
                progress_callback=_callback,
            )

        st.session_state.results = results_df
        st.session_state.last_params = params

    # Display results
    if st.session_state.results is not None:
        df = st.session_state.results.copy()
        ok_df = df[df["status"] == "ok"].copy()

        if ok_df.empty:
            st.error("All selected pipelines failed. Check the terminal for details.")
        else:
            failed = df[df["status"] != "ok"]
            if not failed.empty:
                st.warning(f"⚠️  {len(failed)} pipeline(s) failed: {', '.join(failed['id'].tolist())}")

            # Controls
            col1, col2, col3 = st.columns([2, 2, 1])
            with col1:
                sort_metric = st.selectbox(
                    "Sort by",
                    ["f1_infeasible", "recall_infeasible", "roc_auc", "f1_macro", "accuracy"],
                    key="sort_metric",
                )
            with col2:
                n_top = st.slider("Show top N in chart", 5, min(30, len(ok_df)), min(15, len(ok_df)))
            with col3:
                csv_bytes = df.to_csv(index=False).encode()
                st.download_button(
                    "⬇️  Export CSV",
                    data=csv_bytes,
                    file_name="pipeline_results.csv",
                    mime="text/csv",
                )

            sorted_df = ok_df.sort_values(sort_metric, ascending=False).reset_index(drop=True)

            # Metric summary
            best = sorted_df.iloc[0]
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Best pipeline",   best["id"])
            c2.metric("F1 (infeasible)", f"{best.get('f1_infeasible', float('nan')):.3f}")
            c3.metric("Recall (infeas.)", f"{best.get('recall_infeasible', float('nan')):.3f}")
            c4.metric("ROC-AUC",         f"{best.get('roc_auc', float('nan')):.3f}")

            # Results table
            display_cols = [
                "id", "name", "group", "pp", "fe", "ib", "fs", "cl",
                "accuracy", "f1_macro", "f1_infeasible",
                "recall_infeasible", "roc_auc", "runtime_s",
            ]
            display_cols = [c for c in display_cols if c in sorted_df.columns]
            st.dataframe(
                sorted_df[display_cols].style.background_gradient(
                    subset=["f1_infeasible", "recall_infeasible", "roc_auc"],
                    cmap="YlGn",
                ),
                use_container_width=True,
                height=420,
            )

            # Bar chart — top N
            top_df = sorted_df.head(n_top)
            fig_bar = px.bar(
                top_df,
                x="id",
                y=sort_metric,
                color="group",
                hover_data=["name", "pp", "fe", "ib", "fs", "cl"],
                title=f"Top {n_top} Pipelines by {sort_metric}",
                labels={"id": "Pipeline ID", sort_metric: sort_metric},
                height=450,
            )
            fig_bar.update_xaxes(tickangle=-45)
            st.plotly_chart(fig_bar, use_container_width=True)

            # Scatter: F1 infeasible vs ROC-AUC
            if "roc_auc" in sorted_df.columns and "f1_infeasible" in sorted_df.columns:
                fig_sc = px.scatter(
                    sorted_df,
                    x="roc_auc",
                    y="f1_infeasible",
                    color="group",
                    hover_data=["id", "name"],
                    size="accuracy",
                    title="F1 (Infeasible) vs ROC-AUC — all pipelines",
                    labels={"roc_auc": "ROC-AUC", "f1_infeasible": "F1 Infeasible"},
                    height=450,
                )
                st.plotly_chart(fig_sc, use_container_width=True)

            # Confusion matrix of best pipeline
            st.subheader(f"Confusion Matrix — Best: {best['id']}  ({best['name']})")
            if all(k in best for k in ["tn", "fp", "fn", "tp"]):
                cm_data = np.array([[best["tn"], best["fp"]],
                                    [best["fn"], best["tp"]]])
                fig_cm = px.imshow(
                    cm_data,
                    labels=dict(x="Predicted", y="Actual", color="Count"),
                    x=["Infeasible", "Feasible"],
                    y=["Infeasible", "Feasible"],
                    text_auto=True,
                    color_continuous_scale="Blues",
                    title=f"Confusion Matrix — {best['id']}",
                )
                st.plotly_chart(fig_cm, use_container_width=False)

    else:
        st.info(
            "Select pipelines in the sidebar and click **▶ Run Pipelines** "
            "to see results here."
        )


# ── Tab 3: Pipeline Catalog ───────────────────────────────────────────────
with tab_catalog:
    st.header("Pipeline Catalog — 60 Configurations")

    filter_group = st.selectbox(
        "Filter by group", ["All"] + ALL_GROUPS, key="catalog_filter"
    )

    rows = []
    for p in PIPELINE_CATALOG:
        if filter_group != "All" and p.group != filter_group:
            continue
        rows.append({
            "ID": p.id, "Name": p.name, "Group": p.group,
            "PP": p.pp, "FE": p.fe, "IB": p.ib, "FS": p.fs, "CL": p.cl,
            "Plane Model": "✓" if p.requires_plane_model else "",
            "AE": "✓" if p.requires_ae else "",
        })

    st.dataframe(
        pd.DataFrame(rows),
        use_container_width=True,
        height=600,
        hide_index=True,
    )

    st.caption(
        "**PP** Preprocessing · **FE** Feature Engineering · "
        "**IB** Imbalance · **FS** Feature Selection · **CL** Classifier  "
        "| Plane Model = requires PP-C\\* or PP-E\\* · AE = requires autoencoder pre-training"
    )
