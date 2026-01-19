"""
MaxShapley Annotation Tool - Streamlit UI

A browser-based annotation tool for labeling source relevance in multi-hop QA datasets.
"""

import streamlit as st
from datetime import datetime
import sys
import os
import pandas as pd
import extra_streamlit_components as stx

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from annotation_backend import (
    get_next_sample, save_annotation, get_progress_stats,
    get_annotation_for_sample, delete_annotation, load_annotations,
    get_annotations_summary, get_inter_annotator_stats,
    get_all_annotations_for_sample, get_completion_stats, export_annotations
)
from sample_loader import load_sample, format_sample_for_ui
from config import DATASETS, DATASET_CONFIGS, RELEVANCE_SCALE, EXISTING_SAMPLES, TARGET_SAMPLES_PER_DATASET


# Page configuration
st.set_page_config(
    page_title="MaxShapley Annotation Tool",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    /* Make question/answer area visually distinct */
    .fixed-header {
        background-color: #f8f9fa;
        padding: 15px;
        border-radius: 5px;
        margin-bottom: 20px;
        border-left: 4px solid #1f77b4;
    }

    /* Disagreement highlight */
    .disagreement-high {
        background-color: #ffcccc;
        padding: 5px;
        border-radius: 3px;
    }

    .disagreement-low {
        background-color: #ffffcc;
        padding: 5px;
        border-radius: 3px;
    }
    </style>
""", unsafe_allow_html=True)


def initialize_session_state():
    """Initialize Streamlit session state variables."""
    if "annotator" not in st.session_state:
        st.session_state.annotator = ""
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False
    if "current_index" not in st.session_state:
        st.session_state.current_index = None
    if "annotations_temp" not in st.session_state:
        st.session_state.annotations_temp = {}
    if "comments_temp" not in st.session_state:
        st.session_state.comments_temp = {}
    if "dataset" not in st.session_state:
        st.session_state.dataset = DATASETS[0]
    if "page" not in st.session_state:
        st.session_state.page = "Annotate"
    if "goto_sample" not in st.session_state:
        st.session_state.goto_sample = None
    if "url_initialized" not in st.session_state:
        st.session_state.url_initialized = False


def sync_from_url():
    """Read URL query parameters and update session state."""
    params = st.query_params

    # Only sync from URL once per session (on initial load)
    if st.session_state.url_initialized:
        return

    st.session_state.url_initialized = True

    # Sync page
    if "page" in params:
        page = params["page"]
        if page in ["Annotate", "Overview"]:
            st.session_state.page = page

    # Sync dataset
    if "dataset" in params:
        dataset = params["dataset"]
        if dataset in DATASETS:
            st.session_state.dataset = dataset

    # Sync sample index
    if "sample" in params:
        try:
            sample_idx = int(params["sample"])
            if 0 <= sample_idx < TARGET_SAMPLES_PER_DATASET:
                st.session_state.current_index = sample_idx
        except ValueError:
            pass


def sync_to_url():
    """Update URL query parameters to reflect current session state."""
    params = {}

    # Always include page and dataset
    params["page"] = st.session_state.page
    params["dataset"] = st.session_state.dataset

    # Include sample index if on Annotate page and we have a current index
    if st.session_state.page == "Annotate" and st.session_state.current_index is not None:
        params["sample"] = str(st.session_state.current_index)

    # Update URL without triggering a rerun
    st.query_params.update(params)


# Password for authentication
AUTH_PASSWORD = "shapley123888"
COOKIE_NAME = "maxshapley_auth"


def check_cookie_auth(cookie_manager):
    """Check if user is authenticated via cookie."""
    auth_cookie = cookie_manager.get(COOKIE_NAME)
    if auth_cookie and not st.session_state.logged_in:
        st.session_state.annotator = auth_cookie
        st.session_state.logged_in = True
        return True
    return False


def set_auth_cookie(cookie_manager, username):
    """Set authentication cookie."""
    cookie_manager.set(COOKIE_NAME, username, expires_at=datetime(2030, 1, 1))


def clear_auth_cookie(cookie_manager):
    """Clear authentication cookie."""
    cookie_manager.delete(COOKIE_NAME)


def render_login_page():
    """Render the login page."""
    st.title("MaxShapley Annotation Tool")
    st.markdown("---")

    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        st.subheader("Login")

        username = st.text_input("Username:", placeholder="Enter your name")
        password = st.text_input("Password:", type="password", placeholder="Enter password")

        if st.button("Login", type="primary", use_container_width=True):
            if not username:
                st.error("Please enter your username")
            elif password != AUTH_PASSWORD:
                st.error("Incorrect password")
            else:
                st.session_state.annotator = username
                st.session_state.logged_in = True
                if "_cookie_manager" in st.session_state:
                    set_auth_cookie(st.session_state._cookie_manager, username)
                st.success(f"Welcome, {username}!")
                st.rerun()

        st.markdown("---")
        st.caption("Contact the project coordinator if you need access.")


def render_sidebar():
    """Render the sidebar with navigation and progress stats."""
    with st.sidebar:
        st.title("MaxShapley Annotation")

        # Show logged in user
        annotator = st.session_state.annotator
        st.success(f"Logged in as: **{annotator}**")

        if st.button("Logout", use_container_width=True):
            st.session_state.logged_in = False
            st.session_state.annotator = ""
            st.session_state.current_index = None
            if "_cookie_manager" in st.session_state:
                clear_auth_cookie(st.session_state._cookie_manager)
            st.rerun()

        st.divider()

        # Page navigation
        st.subheader("Navigation")
        page = st.radio(
            "Select page:",
            options=["Annotate", "Overview"],
            index=0 if st.session_state.page == "Annotate" else 1,
            horizontal=True
        )
        st.session_state.page = page

        st.divider()

        # Dataset selection
        st.subheader("Dataset")
        dataset_options = {
            dataset: DATASET_CONFIGS[dataset]["name"]
            for dataset in DATASETS
        }

        dataset = st.radio(
            "Select dataset:",
            options=DATASETS,
            format_func=lambda x: dataset_options[x],
            index=DATASETS.index(st.session_state.dataset)
        )

        # Reset current index if dataset changed
        if dataset != st.session_state.dataset:
            st.session_state.dataset = dataset
            st.session_state.current_index = None

        st.divider()

        # Progress statistics
        st.subheader("Progress")

        # Individual progress
        personal_stats = get_progress_stats(dataset, annotator)
        progress_pct = (personal_stats['completed'] / personal_stats['total']) * 100

        st.metric(
            "Your progress",
            f"{personal_stats['completed']} / {personal_stats['total']} samples",
            delta=f"{progress_pct:.0f}%"
        )

        # Show remaining
        remaining = personal_stats['total'] - personal_stats['completed']
        if remaining > 0:
            st.caption(f"{remaining} samples remaining")
        else:
            st.success("All done!")

        st.divider()

        # Overall progress (unique samples annotated by anyone)
        overall_stats = get_progress_stats(dataset)
        st.caption("**Team Progress (new samples):**")
        if overall_stats['by_annotator']:
            for ann, count in sorted(overall_stats['by_annotator'].items()):
                st.text(f"  {ann}: {count}/70")

        st.divider()

        # Info about original samples
        st.caption("Samples 0-29 have existing annotations from mingxun & sara.")

        st.divider()

        # Help and instructions
        with st.expander("Help"):
            st.markdown("""
            **Relevance scale:**
            - **3**: Highly relevant - key information
            - **2**: Moderately relevant - supporting info
            - **1**: Weakly relevant - tangential
            - **0**: Not relevant

            See the full annotation guide for detailed instructions.
            """)


def render_question_section(sample):
    """Render the question and answer section."""
    st.markdown('<div class="fixed-header">', unsafe_allow_html=True)
    st.subheader("Question")
    st.info(sample["question"])

    if sample["answer"]:
        st.success(f"**Answer:** {sample['answer']}")

    st.markdown('</div>', unsafe_allow_html=True)


def render_source_annotation(sample):
    """Render the sources with annotation controls."""
    st.subheader("Sources")
    st.write("Rate each source's relevance to answering the question (0-3):")

    # Initialize temp annotations for this sample if needed
    if sample["sample_id"] not in st.session_state.annotations_temp:
        st.session_state.annotations_temp[sample["sample_id"]] = {}

    annotations = st.session_state.annotations_temp[sample["sample_id"]]

    # Use Streamlit container for sources
    sources_container = st.container(height=500)

    with sources_container:
        for i, source in enumerate(sample["sources"]):
            with st.expander(f"**Source {i+1}: {source['id']}**", expanded=True):
                st.markdown(source["text"])
                st.divider()

                cols = st.columns(4)
                current_score = annotations.get(source["id"])

                for score in range(4):
                    label = f"{score} - {RELEVANCE_SCALE[score].split()[0]}"
                    button_type = "primary" if current_score == score else "secondary"

                    if cols[score].button(
                        label,
                        key=f"{sample['sample_id']}_{source['id']}_{score}",
                        type=button_type,
                        use_container_width=True
                    ):
                        annotations[source["id"]] = score
                        st.rerun()

                if source["id"] in annotations:
                    st.success(f"Selected: {annotations[source['id']]} - {RELEVANCE_SCALE[annotations[source['id']]]}")
                else:
                    st.warning("Not yet annotated")


def render_comment_section(sample, existing_annotation=None):
    """Render the comment section for the sample."""
    st.divider()
    st.subheader("Comments")
    st.caption("Optional: Leave any notes about this sample (e.g., issues, ambiguities, or interesting observations)")

    # Initialize comment in session state if needed
    if sample["sample_id"] not in st.session_state.comments_temp:
        # Load existing comment if available
        if existing_annotation and "comment" in existing_annotation:
            st.session_state.comments_temp[sample["sample_id"]] = existing_annotation["comment"]
        else:
            st.session_state.comments_temp[sample["sample_id"]] = ""

    comment = st.text_area(
        "Comment:",
        value=st.session_state.comments_temp[sample["sample_id"]],
        key=f"comment_{sample['sample_id']}",
        height=100,
        placeholder="Enter any comments about this sample..."
    )

    # Update session state when user types
    st.session_state.comments_temp[sample["sample_id"]] = comment


def render_navigation(sample, dataset, annotator, current_idx):
    """Render navigation controls."""
    st.divider()

    col1, col2, col3, col4 = st.columns([1, 1, 1, 2])

    with col1:
        # Disable Previous if at first sample
        prev_disabled = current_idx <= 0
        if st.button("Previous", use_container_width=True, disabled=prev_disabled):
            st.session_state.current_index = current_idx - 1
            st.rerun()

    with col2:
        if st.button("Skip", use_container_width=True):
            st.session_state.current_index = None
            st.rerun()

    with col3:
        existing = get_annotation_for_sample(dataset, sample["sample_id"], annotator)
        if existing:
            if st.button("Delete", use_container_width=True, type="secondary"):
                if delete_annotation(dataset, sample["sample_id"], annotator):
                    st.success("Annotation deleted!")
                    st.session_state.current_index = None
                    if sample["sample_id"] in st.session_state.annotations_temp:
                        del st.session_state.annotations_temp[sample["sample_id"]]
                    st.rerun()

    with col4:
        if st.button("Save & Next", use_container_width=True, type="primary"):
            annotations = st.session_state.annotations_temp.get(sample["sample_id"], {})

            if len(annotations) != len(sample["sources"]):
                st.error(f"Please annotate all {len(sample['sources'])} sources before saving")
            else:
                # Get comment if any
                comment = st.session_state.comments_temp.get(sample["sample_id"], "").strip()

                annotation_data = {
                    "sample_id": sample["sample_id"],
                    "dataset": dataset,
                    "annotator": annotator,
                    "timestamp": datetime.now().isoformat(),
                    "annotations": annotations
                }

                # Only include comment if not empty
                if comment:
                    annotation_data["comment"] = comment

                try:
                    save_annotation(annotation_data, dataset)
                    del st.session_state.annotations_temp[sample["sample_id"]]
                    if sample["sample_id"] in st.session_state.comments_temp:
                        del st.session_state.comments_temp[sample["sample_id"]]
                    # Go to next sample (current + 1), or wrap to beginning if at end
                    next_idx = current_idx + 1
                    if next_idx >= TARGET_SAMPLES_PER_DATASET:
                        next_idx = 0
                    st.session_state.current_index = next_idx
                    st.success("Saved successfully!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error saving annotation: {e}")


def render_annotate_page():
    """Render the annotation page."""
    annotator = st.session_state.annotator
    dataset = st.session_state.dataset

    # Show annotation status summary
    personal_stats = get_progress_stats(dataset, annotator)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Completed", f"{personal_stats['completed']} / 100")
    with col2:
        st.metric("Original (locked)", f"30 / 30", delta="Done", delta_color="off")
    with col3:
        st.metric("New samples", f"{personal_stats['new_completed']} / 70")

    st.divider()

    # Check if we should go to a specific sample
    if st.session_state.goto_sample is not None:
        st.session_state.current_index = st.session_state.goto_sample
        st.session_state.goto_sample = None

    # Get next sample to annotate
    if st.session_state.current_index is None:
        next_idx = get_next_sample(annotator, dataset)

        if next_idx is None:
            st.success("Congratulations! You've completed all samples for this dataset!")
            st.info("You can switch to another dataset using the sidebar, or go to Overview to review annotations.")
            st.stop()

        st.session_state.current_index = next_idx

    current_idx = st.session_state.current_index

    # Jump to specific sample
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        st.caption(f"Sample {current_idx} of {dataset.upper()} (samples 0-{TARGET_SAMPLES_PER_DATASET-1})")
    with col2:
        # Use a unique key based on current_idx to force widget refresh on navigation
        goto = st.number_input(
            "Go to sample:",
            min_value=0,
            max_value=TARGET_SAMPLES_PER_DATASET - 1,
            value=current_idx,
            key=f"goto_input_{current_idx}"
        )
        if goto != current_idx:
            st.session_state.current_index = goto
            st.rerun()
    with col3:
        # "Next Unannotated" button - searches from current position + 1
        next_unannotated = get_next_sample(annotator, dataset, start_from=current_idx + 1)
        if next_unannotated is not None:
            if st.button("Next Unannotated", type="primary", use_container_width=True):
                st.session_state.current_index = next_unannotated
                st.rerun()
        else:
            st.success("All done!")

    # Load and format sample
    try:
        raw_sample = load_sample(dataset, current_idx)
        sample = format_sample_for_ui(raw_sample, dataset, current_idx)
    except FileNotFoundError as e:
        st.error(f"""
        **Error:** Sample files not found.

        Please run the data preparation script first:
        ```bash
        python annotation_tool/data_preparation.py
        ```

        Error details: {e}
        """)
        st.stop()
    except Exception as e:
        st.error(f"Error loading sample: {e}")
        st.stop()

    # Check if this is a placeholder sample (copies of 0-29)
    if raw_sample.get("note") and "PLACEHOLDER" in raw_sample.get("note", ""):
        st.warning(f"""
        **Note:** This sample (index {current_idx}) is a PLACEHOLDER - it's a copy of sample {current_idx % EXISTING_SAMPLES}.

        To use real data, replace the placeholder samples in `data/samples/{dataset}_100.json` with new samples from the full dataset.
        """)

    # Check if this sample was already annotated
    existing = get_annotation_for_sample(dataset, sample["sample_id"], annotator)
    if existing:
        st.info("You have already annotated this sample. You can modify and re-save, or delete it.")
        if sample["sample_id"] not in st.session_state.annotations_temp:
            st.session_state.annotations_temp[sample["sample_id"]] = existing["annotations"].copy()

    # Render main content
    render_question_section(sample)
    render_source_annotation(sample)
    render_comment_section(sample, existing)
    render_navigation(sample, dataset, annotator, current_idx)


def render_overview_page():
    """Render the overview page with annotation statistics and disagreement detection."""
    dataset = st.session_state.dataset

    st.header(f"Annotation Overview - {DATASET_CONFIGS[dataset]['name']}")

    # Get statistics
    try:
        stats = get_inter_annotator_stats(dataset)
        summary = get_annotations_summary(dataset)
    except Exception as e:
        st.error(f"Error loading annotations: {e}")
        st.stop()

    # Display overall statistics
    st.subheader("Statistics")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Annotated Samples", stats["total_samples"])
    with col2:
        st.metric("Multi-Annotator Samples", stats["samples_with_multiple_annotators"])
    with col3:
        st.metric("Samples with Disagreement", stats["samples_with_disagreement"])
    with col4:
        st.metric("Avg. Max Disagreement", f"{stats['average_disagreement']:.2f}")

    st.divider()

    # Completion progress per annotator
    st.subheader("Completion Progress by Annotator")
    try:
        completion_stats = get_completion_stats(dataset)
        if completion_stats:
            # Build table data
            progress_rows = []
            for annotator, ann_stats in sorted(completion_stats.items()):
                progress_rows.append({
                    "Annotator": annotator,
                    "Complete": ann_stats["complete"],
                    "Partial": ann_stats["partial"],
                    "Not Started": ann_stats["not_started"],
                    "Total": ann_stats["total"],
                    "Progress": f"{ann_stats['complete'] / ann_stats['total'] * 100:.1f}%"
                })

            progress_df = pd.DataFrame(progress_rows)
            st.dataframe(progress_df, use_container_width=True, hide_index=True)

            # Visual progress bars
            st.caption("Visual Progress:")
            for annotator, ann_stats in sorted(completion_stats.items()):
                complete_pct = ann_stats["complete"] / ann_stats["total"]
                partial_pct = ann_stats["partial"] / ann_stats["total"]

                col1, col2 = st.columns([1, 4])
                with col1:
                    st.write(f"**{annotator}**")
                with col2:
                    # Stacked progress bar using markdown
                    complete_bar = int(complete_pct * 40)
                    partial_bar = int(partial_pct * 40)
                    empty_bar = 40 - complete_bar - partial_bar
                    bar = f"{'█' * complete_bar}{'▓' * partial_bar}{'░' * empty_bar}"
                    st.code(f"{bar} {ann_stats['complete']}✓ {ann_stats['partial']}◐ {ann_stats['not_started']}○")
        else:
            st.info("No annotations found yet.")
    except Exception as e:
        st.warning(f"Could not load completion stats: {e}")

    st.divider()

    # Tabs for different views
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["All Annotations", "Browse Samples", "Disagreements", "Re-annotate", "Export"])

    with tab1:
        st.subheader("All Annotations")

        if not summary:
            st.info("No annotations found for this dataset.")
        else:
            # Load all annotations to check for comments and per-annotator status
            all_annotations = load_annotations(dataset)
            comments_by_sample = {}
            for ann in all_annotations:
                sid = ann["sample_id"]
                if ann.get("comment"):
                    if sid not in comments_by_sample:
                        comments_by_sample[sid] = []
                    comments_by_sample[sid].append(ann["annotator"])

            # Build data for display
            rows = []
            for sample_id, data in sorted(summary.items(), key=lambda x: int(x[0].split("_")[1])):
                idx = int(sample_id.split("_")[1])
                sample_type = "Original (0-29)" if idx < EXISTING_SAMPLES else "New (30-99)"
                disagreement = data["max_disagreement"]
                has_comment = sample_id in comments_by_sample

                # Get per-annotator completion status
                try:
                    sample = load_sample(dataset, idx)
                    formatted = format_sample_for_ui(sample, dataset, idx)
                    num_sources = len(formatted["sources"])

                    annotator_statuses = []
                    for annotator in sorted(data["annotators"]):
                        ann_data = data["annotations"].get(annotator, {})
                        num_annotated = len(ann_data)
                        if num_annotated >= num_sources:
                            annotator_statuses.append(f"{annotator}: ✓")
                        elif num_annotated > 0:
                            annotator_statuses.append(f"{annotator}: ◐ ({num_annotated}/{num_sources})")
                        else:
                            annotator_statuses.append(f"{annotator}: ○")

                    status_str = ", ".join(annotator_statuses)
                except Exception:
                    status_str = ", ".join(sorted(data["annotators"]))

                rows.append({
                    "Index": idx,
                    "Sample ID": sample_id,
                    "Type": sample_type,
                    "Annotator Status": status_str,
                    "Max Disagreement": disagreement,
                    "Has Disagreement": "Yes" if data["has_disagreement"] else "No",
                    "Has Comment": "Yes" if has_comment else "No"
                })

            df = pd.DataFrame(rows)
            st.dataframe(df, use_container_width=True, hide_index=True)

    with tab2:
        st.subheader("Browse Individual Samples")
        st.caption("Select a sample to view its question, answer, and all annotations")

        # Sample selector
        col1, col2 = st.columns([1, 3])
        with col1:
            browse_idx = st.number_input(
                "Sample index:",
                min_value=0,
                max_value=TARGET_SAMPLES_PER_DATASET - 1,
                value=0,
                key="browse_idx"
            )

        browse_sample_id = f"{dataset}_{browse_idx}"

        # Load and display sample
        try:
            raw_sample = load_sample(dataset, browse_idx)
            sample = format_sample_for_ui(raw_sample, dataset, browse_idx)

            # Display question and answer
            st.markdown("---")
            st.markdown("**Question:**")
            st.info(sample["question"])

            if sample["answer"]:
                st.markdown("**Answer:**")
                st.success(sample["answer"])

            # Get all annotations for this sample
            sample_annotations = get_all_annotations_for_sample(dataset, browse_sample_id)

            st.markdown("---")

            if sample_annotations:
                st.success(f"**Annotated by {len(sample_annotations)} annotator(s):** {', '.join([a['annotator'] for a in sample_annotations])}")

                # Show comments if any
                comments = [(a["annotator"], a.get("comment", "")) for a in sample_annotations if a.get("comment")]
                if comments:
                    st.markdown("**Comments:**")
                    for annotator_name, comment in comments:
                        st.info(f"**{annotator_name}:** {comment}")

                # Build annotation lookup for display
                annotation_lookup = {}
                for ann in sample_annotations:
                    for source_id, score in ann["annotations"].items():
                        if source_id not in annotation_lookup:
                            annotation_lookup[source_id] = {}
                        annotation_lookup[source_id][ann["annotator"]] = score

                # Show each source with its annotations
                st.markdown("**Sources and Annotations:**")
                for i, source in enumerate(sample["sources"]):
                    source_id = source["id"]

                    # Build score summary for this source
                    if source_id in annotation_lookup:
                        scores_str = " | ".join([
                            f"{annotator}: {score} ({RELEVANCE_SCALE[score].split()[0]})"
                            for annotator, score in annotation_lookup[source_id].items()
                        ])
                        header = f"Source {i+1}: {source_id} — {scores_str}"
                    else:
                        header = f"Source {i+1}: {source_id} — No annotations"

                    with st.expander(header, expanded=False):
                        st.write(source["text"])
            else:
                st.warning(f"No annotations found for {browse_sample_id}")

                # Still show sources even if no annotations
                st.markdown("**Sources:**")
                for i, source in enumerate(sample["sources"]):
                    with st.expander(f"Source {i+1}: {source['id']}", expanded=False):
                        st.write(source["text"])

        except Exception as e:
            st.error(f"Error loading sample: {e}")

    with tab3:
        st.subheader("Samples with Disagreements")
        st.caption("Showing samples where annotators differ by 2+ points on any source")

        disagreement_samples = [
            (sid, data) for sid, data in summary.items()
            if data["has_disagreement"]
        ]

        if not disagreement_samples:
            st.success("No significant disagreements found!")
        else:
            for sample_id, data in sorted(disagreement_samples, key=lambda x: -x[1]["max_disagreement"]):
                idx = int(sample_id.split("_")[1])

                with st.expander(f"**{sample_id}** - Max disagreement: {data['max_disagreement']}", expanded=False):
                    # Show annotations by each annotator
                    all_sources = set()
                    for ann in data["annotations"].values():
                        all_sources.update(ann.keys())

                    # Build comparison table
                    comparison_rows = []
                    for source in sorted(all_sources):
                        row = {"Source": source}
                        scores = []
                        for annotator in sorted(data["annotators"]):
                            score = data["annotations"][annotator].get(source, "N/A")
                            row[annotator] = score
                            if isinstance(score, int):
                                scores.append(score)

                        # Calculate disagreement for this source
                        if len(scores) > 1:
                            row["Diff"] = max(scores) - min(scores)
                        else:
                            row["Diff"] = 0

                        comparison_rows.append(row)

                    comparison_df = pd.DataFrame(comparison_rows)
                    st.dataframe(comparison_df, use_container_width=True, hide_index=True)

                    # Button to go to this sample for re-annotation
                    if st.button(f"Re-annotate {sample_id}", key=f"goto_{sample_id}"):
                        st.session_state.goto_sample = idx
                        st.session_state.page = "Annotate"
                        st.rerun()

    with tab4:
        st.subheader("Re-annotate Specific Sample")
        st.caption("Enter a sample index to view and re-annotate")

        col1, col2 = st.columns([2, 1])

        with col1:
            sample_idx = st.number_input(
                "Sample index:",
                min_value=0,
                max_value=TARGET_SAMPLES_PER_DATASET - 1,
                value=0,
                help=f"Samples 0-{TARGET_SAMPLES_PER_DATASET-1}"
            )

        with col2:
            if st.button("Go to Sample", type="primary"):
                st.session_state.goto_sample = sample_idx
                st.session_state.page = "Annotate"
                st.rerun()

        # Show existing annotations for selected sample
        sample_id = f"{dataset}_{sample_idx}"
        existing_annotations = get_all_annotations_for_sample(dataset, sample_id)

        if existing_annotations:
            st.write(f"**Existing annotations for {sample_id}:**")

            for ann in existing_annotations:
                header = f"Annotator: {ann['annotator']} ({ann['timestamp'][:10]})"
                if ann.get("comment"):
                    header += " - Has comment"
                with st.expander(header):
                    ann_df = pd.DataFrame([
                        {"Source": k, "Score": v, "Label": RELEVANCE_SCALE[v]}
                        for k, v in ann["annotations"].items()
                    ])
                    st.dataframe(ann_df, use_container_width=True, hide_index=True)

                    # Show comment if present
                    if ann.get("comment"):
                        st.markdown("**Comment:**")
                        st.info(ann["comment"])
        else:
            st.info(f"No annotations found for {sample_id}")

    with tab5:
        st.subheader("Export Annotations")
        st.caption("Export annotated data in the same format as the original subset files")

        col1, col2 = st.columns(2)

        with col1:
            # Get list of annotators for this dataset
            all_anns = load_annotations(dataset)
            annotators = sorted(set(a["annotator"] for a in all_anns))

            aggregation_options = ["average", "first"] + annotators
            aggregation = st.selectbox(
                "Score aggregation method:",
                options=aggregation_options,
                help="How to combine scores from multiple annotators"
            )

            min_annotators = st.number_input(
                "Minimum annotators per sample:",
                min_value=1,
                max_value=max(len(annotators), 1),
                value=1,
                help="Only include samples with at least this many annotators"
            )

        with col2:
            only_complete = st.checkbox(
                "Only export complete annotations",
                value=True,
                help="Only include samples where all sources are annotated"
            )

            include_meta = st.checkbox(
                "Include annotation metadata",
                value=False,
                help="Include info about annotators and aggregation method"
            )

        st.divider()

        # Preview export
        if st.button("Generate Export Preview", type="secondary"):
            with st.spinner("Generating export..."):
                try:
                    exported = export_annotations(
                        dataset,
                        aggregation=aggregation,
                        min_annotators=min_annotators,
                        only_complete=only_complete
                    )

                    if not exported:
                        st.warning("No samples match the export criteria.")
                    else:
                        st.success(f"Found {len(exported)} samples to export")

                        # Store in session state for download
                        st.session_state.export_data = exported
                        st.session_state.export_include_meta = include_meta

                        # Show preview
                        st.write("**Preview (first 3 samples):**")
                        for i, sample in enumerate(exported[:3]):
                            with st.expander(f"Sample {i}: {sample.get('_id') or sample.get('query_id', 'unknown')}"):
                                st.write(f"**Question:** {sample['question'][:200]}...")
                                if sample.get('answer'):
                                    st.write(f"**Answer:** {sample['answer']}")
                                st.write(f"**Relevance Labels:** {sample.get('relevance_labels', [])}")
                                if include_meta and '_annotation_meta' in sample:
                                    st.write(f"**Meta:** {sample['_annotation_meta']}")

                except Exception as e:
                    st.error(f"Error generating export: {e}")

        # Download button
        if "export_data" in st.session_state and st.session_state.export_data:
            import json

            export_data = st.session_state.export_data
            include_meta = st.session_state.get("export_include_meta", False)

            # Remove meta if not requested
            if not include_meta:
                export_data = [
                    {k: v for k, v in sample.items() if k != "_annotation_meta"}
                    for sample in export_data
                ]

            json_str = json.dumps(export_data, indent=2)

            st.download_button(
                label=f"Download {dataset}_annotated_export.json",
                data=json_str,
                file_name=f"{dataset}_annotated_export.json",
                mime="application/json",
                type="primary"
            )

            st.caption(f"Export contains {len(export_data)} samples")


def main():
    """Main application logic."""
    initialize_session_state()

    # Initialize cookie manager (must be done early, creates a component)
    cookie_manager = stx.CookieManager(key="main_cookie_manager")

    # Check for cookie-based authentication
    check_cookie_auth(cookie_manager)

    # Sync state from URL on initial load
    sync_from_url()

    # Store cookie_manager in session state for use in other functions
    st.session_state._cookie_manager = cookie_manager

    # Show login page if not logged in
    if not st.session_state.logged_in:
        render_login_page()
        st.stop()

    # Show main app with sidebar
    render_sidebar()

    if st.session_state.page == "Annotate":
        render_annotate_page()
    else:
        render_overview_page()

    # Sync URL to reflect current state
    sync_to_url()


if __name__ == "__main__":
    main()
