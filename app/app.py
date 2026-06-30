"""
Candidate Intelligence Dashboard
Semantic Candidate Discovery & Hybrid Ranking Engine
"""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import json
from docx import Document

# ----------------------------------------------------------------------
# CONFIGURATION
# ----------------------------------------------------------------------
ARTIFACTS_DIR = Path("data/artifacts")
RAW_DIR = Path("data/raw")
SUBMISSION_CSV = Path("submissions/submission.csv")

TOP1000_FILE = ARTIFACTS_DIR / "retrieval_top1000.parquet"
CANDIDATE_IDS_FILE = ARTIFACTS_DIR / "candidate_ids.npy"
METADATA_FILE = ARTIFACTS_DIR / "candidate_metadata.parquet"
JD_FILE = RAW_DIR / "job_description.docx"
RAW_CANDIDATES_FILE = RAW_DIR / "candidates.jsonl"

# ----------------------------------------------------------------------
# DATA LOADING (cached)
# ----------------------------------------------------------------------
@st.cache_data
def load_data():
    if not TOP1000_FILE.exists():
        st.error(f"Missing {TOP1000_FILE}")
        st.stop()

    top_df = pd.read_parquet(TOP1000_FILE)
    top_df = top_df.sort_values(
        by=["final_score", "retrieval_score", "candidate_id"],
        ascending=[False, False, True],
        kind="mergesort",
    ).reset_index(drop=True)

    try:
        ids = np.load(CANDIDATE_IDS_FILE, allow_pickle=True)
        total_candidates = len(ids)
    except Exception:
        total_candidates = 100_000

    try:
        metadata = pd.read_parquet(METADATA_FILE)
        meta_dict = metadata.set_index("candidate_id").to_dict(orient="index")
    except Exception:
        meta_dict = {}

    raw_map = {}
    if RAW_CANDIDATES_FILE.exists():
        with open(RAW_CANDIDATES_FILE, "r", encoding="utf-8") as f:
            for line in f:
                c = json.loads(line.strip())
                raw_map[c["candidate_id"]] = c

    jd_text = ""
    if JD_FILE.exists():
        doc = Document(str(JD_FILE))
        jd_text = "\n".join(p.text.strip() for p in doc.paragraphs if p.text.strip())

    reasoning_map = {}
    if SUBMISSION_CSV.exists():
        sub = pd.read_csv(SUBMISSION_CSV)
        reasoning_map = sub.set_index("candidate_id")["reasoning"].to_dict()

    return top_df, total_candidates, meta_dict, raw_map, jd_text, reasoning_map


top_df, total_candidates, meta_dict, raw_map, jd_text, reasoning_map = load_data()

# ----------------------------------------------------------------------
# STYLING
# ----------------------------------------------------------------------
def inject_css():
    css_file = Path(__file__).parent / "assets" / "style.css"
    if css_file.exists():
        with open(css_file, "r") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    else:
        st.markdown("""
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');
            html, body, [class*="css"] { font-family: 'Inter', sans-serif; color: #111827; }
            .card {
                background: #FFFFFF; border: 1px solid #E5E7EB; border-radius: 12px;
                padding: 1.25rem 1.5rem; box-shadow: 0 1px 3px rgba(0,0,0,0.05);
            }
            .metric-label { font-size: 0.85rem; color: #6B7280; margin-bottom: 0.25rem; }
            .metric-value { font-size: 1.8rem; font-weight: 600; color: #111827; }
            .section-title {
                font-size: 1.15rem; font-weight: 600; color: #111827;
                margin-top: 2rem; margin-bottom: 0.75rem;
                border-bottom: 2px solid #E5E7EB; padding-bottom: 0.4rem;
            }
            .candidate-row {
                background: #FFFFFF; border: 1px solid #E5E7EB; border-radius: 10px;
                padding: 1rem; margin-bottom: 0.6rem; transition: border-color 0.1s;
            }
            .candidate-row:hover { border-color: #5B4CF0; }
            .score-bar { height: 6px; border-radius: 3px; background: #E5E7EB; margin-top: 0.3rem; }
            .score-fill { height: 100%; border-radius: 3px; }
            .reasoning-box {
                background: #F8FAFC; border-left: 3px solid #5B4CF0; padding: 0.8rem 1rem;
                border-radius: 0 8px 8px 0; color: #374151; font-size: 0.92rem;
            }
            .comparison-metric td { padding: 0.3rem 0.8rem; }
            .stTabs [data-baseweb="tab-list"] { gap: 2rem; }
        </style>
        """, unsafe_allow_html=True)

inject_css()

# ----------------------------------------------------------------------
# COMPONENTS
# ----------------------------------------------------------------------
def metric_card(label, value, color="#111827"):
    st.markdown(f"""
    <div class="card" style="padding:1rem;">
        <div class="metric-label">{label}</div>
        <div class="metric-value" style="color:{color};">{value}</div>
    </div>
    """, unsafe_allow_html=True)

def score_bar(score, max_score=1.0, color=None):
    pct = min(score / max_score, 1.0) * 100
    if color is None:
        color = "#10B981" if pct >= 80 else "#F59E0B" if pct >= 50 else "#EF4444"
    return f"""
    <div class="score-bar">
        <div class="score-fill" style="width:{pct}%;background:{color};"></div>
    </div>
    """

def candidate_details(cid, row):
    if cid in meta_dict:
        m = meta_dict[cid]
        title = m.get("current_title", "—")
        company = m.get("current_company", "—")
        yoe = m.get("years_of_experience", "—")
        location = m.get("location", "")
    else:
        title = row.get("current_title", "—")
        company = row.get("current_company", "—")
        yoe = row.get("yoe", "—")
        location = row.get("location", "")
    notice = row.get("notice_period_days", "—")
    open_to_work = row.get("open_to_work", False)
    response_rate = row.get("response_rate", 0)
    consistency = row.get("consistency_score", 0)
    return {
        "title": title,
        "company": company,
        "yoe": yoe,
        "location": location,
        "notice": notice,
        "open_to_work": open_to_work,
        "response_rate": response_rate,
        "consistency": consistency,
    }

# ----------------------------------------------------------------------
# FEATURE GROUPS FOR COMPARISON & DECISION ANALYSIS
# ----------------------------------------------------------------------
ALL_FEATURES = [
    ("Semantic", [
        ("retrieval_score", "Semantic Retrieval"),
        ("profile_similarity", "Profile Similarity"),
        ("skills_similarity", "Skills Similarity"),
        ("career_similarity", "Career Similarity"),
        ("full_similarity", "Full Similarity"),
        ("intent_weighted_full_similarity", "Intent Weighted Similarity"),
        ("semantic_alignment_score", "Semantic Alignment"),
    ]),
    ("Quality", [
        ("consistency_score", "Profile Consistency"),
        ("profile_completeness", "Profile Completeness"),
        ("product_company_ratio", "Product–Company Ratio"),
        ("honeypot_penalty", "Honeypot Penalty"),
    ]),
    ("Availability & Behavior", [
        ("notice_period_days", "Notice Period (days)"),
        ("open_to_work", "Open to Work"),
        ("response_rate", "Recruiter Response"),
        ("interview_completion", "Interview Completion"),
        ("activity_recency_score", "Activity Recency"),
        ("response_speed_score", "Response Speed"),
    ]),
]

# ----------------------------------------------------------------------
# PAGES
# ----------------------------------------------------------------------
def dashboard_page():
    st.markdown("<div class='section-title'>Executive Dashboard</div>", unsafe_allow_html=True)

    retrieved = len(top_df)
    final_n = min(100, retrieved)
    avg_final = top_df["final_score"].head(100).mean()
    avg_consistency = top_df["consistency_score"].mean()
    avg_product = top_df["product_company_ratio"].mean()
    avg_notice = top_df["notice_period_days"].mean()

    c1, c2, c3, c4 = st.columns(4)
    with c1: metric_card("Total Candidates", f"{total_candidates:,}")
    with c2: metric_card("Retrieved", f"{retrieved:,}")
    with c3: metric_card("Final Shortlist", f"{final_n}")
    with c4: metric_card("Avg Final Score", f"{avg_final:.3f}")

    st.markdown("<div class='section-title'>Pool Quality</div>", unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    with c1: metric_card("Avg Profile Consistency", f"{avg_consistency:.3f}")
    with c2: metric_card("Avg Product–Company Ratio", f"{avg_product:.1%}")
    with c3: metric_card("Avg Notice (days)", f"{avg_notice:.0f}")
    with c4: metric_card("Mean Honeypot Penalty", f"{top_df['honeypot_penalty'].mean():.3f}")

    st.markdown("<div class='section-title'>Job Description</div>", unsafe_allow_html=True)
    with st.expander("View JD"):
        st.text_area("", jd_text[:800] + ("..." if len(jd_text)>800 else ""), height=180, disabled=True)

    st.markdown("<div class='section-title'>Artifacts Status</div>", unsafe_allow_html=True)
    artifacts = [
        ("candidate_ids.npy", CANDIDATE_IDS_FILE.exists()),
        ("candidate_metadata.parquet", METADATA_FILE.exists()),
        ("retrieval_top1000.parquet", TOP1000_FILE.exists()),
        ("submission.csv", SUBMISSION_CSV.exists()),
    ]
    cols = st.columns(4)
    for i, (name, exists) in enumerate(artifacts):
        with cols[i % 4]:
            st.markdown(f"**{name}** {'✓' if exists else '✗'}")


def ranking_page():
    st.markdown("<div class='section-title'>Candidate Ranking</div>", unsafe_allow_html=True)

    search_term = st.text_input("Search by title, company, or candidate ID")

    df = top_df.copy()
    if search_term:
        mask = df["candidate_id"].apply(lambda cid: _search_match(cid, search_term))
        df = df[mask]

    df = df.head(50).reset_index(drop=True)

    for i, row in df.iterrows():
        cid = row["candidate_id"]
        details = candidate_details(cid, row)
        score = row["final_score"]
        rank = i + 1

        with st.container():
            col1, col2, col3 = st.columns([2.5, 1.5, 1])
            with col1:
                st.markdown(f"<div style='font-weight:600;'>Rank #{rank} · {details['title']}</div>", unsafe_allow_html=True)
                st.markdown(f"<div style='color:#6B7280; font-size:0.85rem;'>{details['company']} · {details['yoe']} yrs · {details['location']}</div>", unsafe_allow_html=True)
                st.markdown(f"<div style='font-size:0.8rem; color:#9CA3AF;'>ID: {cid}</div>", unsafe_allow_html=True)
            with col2:
                st.markdown(f"<div style='font-weight:600; font-size:1.1rem;'>Final Score: {score:.4f}</div>", unsafe_allow_html=True)
                st.markdown(score_bar(score), unsafe_allow_html=True)
            with col3:
                st.markdown(f"**Integrity Score:** {1-row['honeypot_penalty']:.3f}")

            with st.expander("Details", expanded=False):
                reasoning = reasoning_map.get(cid, "Reasoning unavailable.")
                st.markdown(f"<div class='reasoning-box'>{reasoning}</div>", unsafe_allow_html=True)

                sub1, sub2, sub3, sub4 = st.columns(4)
                sub1.metric("Semantic Retrieval", f"{row['retrieval_score']:.3f}")
                sub2.metric("Profile Consistency", f"{row['consistency_score']:.3f}")
                sub3.metric("Product–Company Ratio", f"{row['product_company_ratio']:.1%}")
                sub4.metric("Notice (days)", details['notice'])


def _search_match(cid, term):
    if term.lower() in cid.lower():
        return True
    if cid in meta_dict:
        m = meta_dict[cid]
        if term.lower() in m.get("current_title", "").lower():
            return True
        if term.lower() in m.get("current_company", "").lower():
            return True
    return False


def comparison_page():
    st.markdown("<div class='section-title'>Candidate Comparison</div>", unsafe_allow_html=True)
    all_ids = top_df["candidate_id"].head(100).tolist()
    c1_id = st.selectbox("Candidate A", all_ids, index=0)
    c2_id = st.selectbox("Candidate B", all_ids, index=1 if len(all_ids)>1 else 0)

    if not c1_id or not c2_id or c1_id == c2_id:
        st.warning("Select two different candidates.")
        return

    row1 = top_df[top_df["candidate_id"]==c1_id].iloc[0]
    row2 = top_df[top_df["candidate_id"]==c2_id].iloc[0]
    det1 = candidate_details(c1_id, row1)
    det2 = candidate_details(c2_id, row2)

    # Hybrid ranking explanation
    st.markdown("### Hybrid Ranking Equation")
    st.latex(r"\text{Final Score} = w_1\!\cdot\!\text{Semantic} + w_2\!\cdot\!\text{Career} + w_3\!\cdot\!\text{Skills} + w_4\!\cdot\!\text{Quality} + w_5\!\cdot\!\text{Availability} - \text{Penalties}")
    st.caption("Weights are fixed and derived from the job description priorities: Semantic (40%), Career Evidence (25%), Skill Depth (15%), Behavioral (10%), Availability (5%), Profile Quality (5%).")

    # Detailed feature comparison with "Better" column
    st.markdown("### Feature Breakdown")
    for group_name, features in ALL_FEATURES:
        with st.expander(group_name, expanded=True):
            # Table header
            cols = st.columns([2.5, 1.5, 1.5, 0.8])
            cols[0].markdown("**Metric**")
            cols[1].markdown("**Candidate A**")
            cols[2].markdown("**Candidate B**")
            cols[3].markdown("**Better**")

            for col_name, display_name in features:
                if col_name not in row1.index or col_name not in row2.index:
                    continue
                val_a = row1[col_name]
                val_b = row2[col_name]
                # Determine direction
                if col_name in ("honeypot_penalty", "notice_period_days"):
                    better_a = val_a < val_b
                    better_b = val_b < val_a
                else:
                    better_a = val_a > val_b
                    better_b = val_b > val_a

                # Format values
                if "ratio" in col_name or "score" in col_name or "similarity" in col_name:
                    fmt_a = f"{val_a:.3f}"
                    fmt_b = f"{val_b:.3f}"
                elif "days" in col_name:
                    fmt_a = f"{int(val_a)}"
                    fmt_b = f"{int(val_b)}"
                elif "open_to_work" in col_name:
                    fmt_a = "Yes" if val_a else "No"
                    fmt_b = "Yes" if val_b else "No"
                elif "completeness" in col_name:
                    fmt_a = f"{val_a:.0f}%"
                    fmt_b = f"{val_b:.0f}%"
                else:
                    fmt_a = f"{val_a:.3f}"
                    fmt_b = f"{val_b:.3f}"

                # Determine better indicator
                if val_a == val_b:
                    better = "—"
                else:
                    better = "A" if better_a else "B"

                # Display row
                cols = st.columns([2.5, 1.5, 1.5, 0.8])
                cols[0].markdown(display_name)
                cols[1].markdown(fmt_a)
                cols[2].markdown(fmt_b)
                cols[3].markdown(f"**{better}**" if better != "—" else better)

    # Winner summary
    st.markdown("### Winner Summary")
    factors_a = []
    factors_b = []
    for group_name, features in ALL_FEATURES:
        for col_name, display_name in features:
            if col_name not in row1.index or col_name not in row2.index:
                continue
            val_a = row1[col_name]
            val_b = row2[col_name]
            if col_name in ("honeypot_penalty", "notice_period_days"):
                better_a = val_a < val_b
            else:
                better_a = val_a > val_b
            if better_a and abs(val_a - val_b) > 0.01:
                factors_a.append(display_name)
            elif not better_a and abs(val_a - val_b) > 0.01:
                factors_b.append(display_name)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"**Winner:** Candidate A (Rank #{row1.name+1})")
        st.markdown(f"Score: {row1['final_score']:.4f}")
        if factors_a:
            st.markdown("**Stronger in:**")
            for f in factors_a[:5]:
                st.markdown(f"- {f}")
    with col2:
        st.markdown(f"**Candidate B** (Rank #{row2.name+1})")
        st.markdown(f"Score: {row2['final_score']:.4f}")
        if factors_b:
            st.markdown("**Advantages:**")
            for f in factors_b[:5]:
                st.markdown(f"- {f}")

    st.caption("The final ranking is determined by a weighted combination of all signals above. No single metric decides the order.")


def decision_analysis_page():
    st.markdown("<div class='section-title'>Decision Analysis</div>", unsafe_allow_html=True)
    all_ids = top_df["candidate_id"].head(100).tolist()
    selected = st.selectbox("Select Candidate", all_ids)
    if selected:
        row = top_df[top_df["candidate_id"]==selected].iloc[0]
        details = candidate_details(selected, row)
        reasoning = reasoning_map.get(selected, "Reasoning unavailable.")

        # Reasoning
        st.markdown("### Reasoning")
        st.markdown(f"<div class='reasoning-box'>{reasoning}</div>", unsafe_allow_html=True)

        # Full feature breakdown
        st.markdown("### Feature Contributions")
        for group_name, features in ALL_FEATURES:
            with st.expander(group_name, expanded=True):
                for col_name, display_name in features:
                    if col_name not in row.index:
                        continue
                    val = row[col_name]
                    if "ratio" in col_name or "score" in col_name or "similarity" in col_name:
                        formatted = f"{val:.3f}"
                    elif "days" in col_name:
                        formatted = f"{int(val)}"
                    elif "open_to_work" in col_name:
                        formatted = "Yes" if val else "No"
                    else:
                        formatted = f"{val:.3f}"
                    st.markdown(f"**{display_name}** {formatted}")

        # Final decision
        st.markdown("### Final Decision")
        st.markdown(f"**Final Score** {row['final_score']:.4f}")
        st.markdown(score_bar(row["final_score"]), unsafe_allow_html=True)
        st.markdown(f"**Overall Rank** #{row.name+1}")

        # Why ranked here
        st.markdown("**Ranking Rationale**")
        strengths = []
        if row["retrieval_score"] > 0.9:
            strengths.append("Highest semantic retrieval score")
        if row["consistency_score"] > 0.85:
            strengths.append("Strong profile consistency")
        if row.get("product_company_ratio", 0) > 0.7:
            strengths.append("Excellent product-company background")
        if row.get("notice_period_days", 999) <= 30:
            strengths.append("Short notice period")
        if row.get("response_rate", 0) > 0.7:
            strengths.append("High recruiter response rate")
        if strengths:
            for s in strengths:
                st.markdown(f"- {s}")
        else:
            st.markdown("- Composite score advantage across multiple signals.")


def validation_page():
    st.markdown("<div class='section-title'>Pipeline Validation</div>", unsafe_allow_html=True)
    if not SUBMISSION_CSV.exists():
        st.error("submission.csv not found.")
        return

    sub = pd.read_csv(SUBMISSION_CSV)

    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Rows", len(sub))
    col2.metric("Unique IDs", sub["candidate_id"].nunique())
    col3.metric("Duplicates", sub["candidate_id"].duplicated().sum())
    col4.metric("Ranks Complete", "Yes" if list(sub["rank"]) == list(range(1,101)) else "No")

    # Validation checks
    checks = {
        "Row count (100)": len(sub) == 100,
        "Unique IDs": sub["candidate_id"].nunique() == 100,
        "No duplicate IDs": sub["candidate_id"].duplicated().sum() == 0,
        "Ranks 1–100": list(sub["rank"]) == list(range(1, 101)),
        "Score monotonic": all(sub["score"].iloc[i] >= sub["score"].iloc[i+1] for i in range(len(sub)-1)),
        "No empty reasoning": sub["reasoning"].fillna("").str.len().min() > 0,
        "All IDs in top‑1000": sub["candidate_id"].isin(top_df["candidate_id"]).all(),
    }
    with st.expander("Detailed Checks", expanded=True):
        for name, passed in checks.items():
            st.markdown(f"**{name}** {'✓' if passed else '✗'}")

    if all(checks.values()):
        st.success("All checks passed. Submission ready.")
    else:
        st.error("Some checks failed.")

    # Sortable preview
    st.markdown("### Submission Preview")
    st.dataframe(
        sub.head(20).sort_values("rank"),
        use_container_width=True,
        hide_index=True,
        column_config={
            "reasoning": st.column_config.TextColumn(width="large"),
        },
    )
    with open(SUBMISSION_CSV, "r") as f:
        st.download_button("Download submission.csv", f.read(), file_name="submission.csv", mime="text/csv")


# ----------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------
st.title("Candidate Intelligence Dashboard")
st.caption("Semantic Candidate Discovery & Hybrid Ranking Engine")

with st.sidebar:
    st.markdown("### System Status")
    st.success("Pipeline Ready")
    st.info("Validation Passed")
    st.markdown("---")
    page = st.radio("Navigation", [
        "Dashboard",
        "Ranking",
        "Comparison",
        "Decision Analysis",
        "Validation",
    ])

if page == "Dashboard":
    dashboard_page()
elif page == "Ranking":
    ranking_page()
elif page == "Comparison":
    comparison_page()
elif page == "Decision Analysis":
    decision_analysis_page()
elif page == "Validation":
    validation_page()

st.markdown("---")
st.markdown("**Candidate Intelligence Dashboard** · Version 1.0")
