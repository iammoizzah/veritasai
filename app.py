import streamlit as st
from main import run_pipeline
from src.tools.vector_store import get_collection_stats

st.set_page_config(
    page_title="VeritasAI",
    page_icon="⚖️",
    layout="wide"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Fraunces:ital,wght@0,700;1,600&display=swap');
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0;}
html,body,[class*="css"]{font-family:'Inter',sans-serif;background:#F9F9F7!important;color:#1A1A1A!important;}

.hero{background:#0F0F0F;border-radius:16px;padding:2.5rem;margin-bottom:1.5rem;}
.hero h1{font-family:'Fraunces',serif;font-size:2.8rem;font-weight:700;color:#FFF;letter-spacing:-0.02em;}
.hero h1 em{color:#4ADE80;font-style:italic;}
.hero-sub{font-size:0.9rem;color:#666;margin-top:0.4rem;font-weight:300;}
.hero-stats{display:flex;gap:2rem;margin-top:1.25rem;}
.hero-stat{font-size:0.72rem;color:#555;}
.hero-stat span{color:#4ADE80;font-weight:600;font-size:0.85rem;}

.verdict-box{border-radius:14px;padding:1.5rem 2rem;margin:1rem 0;}
.verdict-TRUE{background:#F0FFF4;border:1.5px solid #86EFAC;}
.verdict-FALSE{background:#FFF1F1;border:1.5px solid #FCA5A5;}
.verdict-MISLEADING{background:#FFFBEB;border:1.5px solid #FDE68A;}
.verdict-PARTIALLY_TRUE{background:#EFF6FF;border:1.5px solid #93C5FD;}
.verdict-UNVERIFIABLE{background:#F9F9F9;border:1.5px solid #E0E0E0;}
.verdict-CACHED{background:#F5F3FF;border:1.5px solid #C4B5FD;}

.verdict-label{font-size:0.6rem;font-weight:700;letter-spacing:0.2em;text-transform:uppercase;margin-bottom:0.4rem;}
.verdict-TRUE .verdict-label{color:#16A34A;}
.verdict-FALSE .verdict-label{color:#DC2626;}
.verdict-MISLEADING .verdict-label{color:#D97706;}
.verdict-PARTIALLY_TRUE .verdict-label{color:#2563EB;}
.verdict-UNVERIFIABLE .verdict-label{color:#6B7280;}
.verdict-CACHED .verdict-label{color:#7C3AED;}

.verdict-text{font-family:'Fraunces',serif;font-size:2rem;font-weight:700;color:#1A1A1A;}
.confidence{font-size:0.82rem;color:#888;margin-top:0.2rem;}

.meta-grid{display:grid;grid-template-columns:1fr 1fr;gap:0.75rem;margin:1rem 0;}
.meta-card{background:#FFF;border:1px solid #EBEBEB;border-radius:10px;padding:1rem 1.1rem;}
.meta-label{font-size:0.6rem;font-weight:700;letter-spacing:0.16em;text-transform:uppercase;color:#AAA;margin-bottom:0.3rem;}
.meta-value{font-size:0.88rem;color:#1A1A1A;font-weight:500;}

.section-head{font-size:0.6rem;font-weight:700;letter-spacing:0.18em;text-transform:uppercase;color:#BBB;padding-bottom:0.5rem;border-bottom:1px solid #EBEBEB;margin:1.5rem 0 0.75rem;}

.fact-item{display:flex;gap:0.6rem;padding:0.45rem 0;border-bottom:1px solid #F5F5F5;}
.fact-dot{width:6px;height:6px;border-radius:50%;background:#4ADE80;flex-shrink:0;margin-top:0.35rem;}
.fact-text{font-size:0.85rem;color:#444;line-height:1.55;}

.source-chip{display:inline-block;font-size:0.7rem;background:#F5F5F5;border:1px solid #EBEBEB;border-radius:4px;padding:0.2rem 0.5rem;margin:0.2rem;color:#666;word-break:break-all;}

.agent-step{display:flex;align-items:center;gap:0.75rem;padding:0.5rem 0;border-bottom:1px solid #F5F5F5;}
.step-icon{font-size:1rem;width:28px;text-align:center;}
.step-text{font-size:0.82rem;color:#444;}
.step-badge{font-size:0.65rem;padding:0.15rem 0.5rem;border-radius:999px;font-weight:600;margin-left:auto;}
.badge-done{background:#F0FFF4;color:#16A34A;border:1px solid #86EFAC;}
.badge-cached{background:#F5F3FF;color:#7C3AED;border:1px solid #C4B5FD;}

.stButton>button{background:#0F0F0F!important;color:#4ADE80!important;border:none!important;border-radius:10px!important;padding:0.7rem 2rem!important;font-family:'Inter',sans-serif!important;font-weight:600!important;font-size:0.9rem!important;width:100%!important;}
.stButton>button:hover{background:#1A1A1A!important;}
div[data-testid="stTextArea"] textarea{background:#FFF!important;border:1.5px solid #EBEBEB!important;border-radius:10px!important;color:#1A1A1A!important;font-family:'Inter',sans-serif!important;font-size:0.92rem!important;}
div[data-testid="stTextArea"] textarea:focus{border-color:#4ADE80!important;box-shadow:0 0 0 3px rgba(74,222,128,0.12)!important;}
div[data-testid="stTextArea"] textarea::placeholder{color:#CCC!important;}
#MainMenu,footer,header{visibility:hidden;}
</style>
""", unsafe_allow_html=True)

# ── Hero ──────────────────────────────────────────────────────────────────────
stats = get_collection_stats()
st.markdown(f"""
<div class="hero">
    <h1>Veritas<em>AI</em></h1>
    <p class="hero-sub">Automated claim verification · Multi-agent pipeline · Real evidence · Bias detection</p>
    <div class="hero-stats">
        <div class="hero-stat">Claims verified: <span>{stats.get('total_claims', 0)}</span></div>
        <div class="hero-stat">Evidence chunks: <span>{stats.get('total_evidence_chunks', 0)}</span></div>
        <div class="hero-stat">Eval accuracy: <span>60%</span></div>
        <div class="hero-stat">RAGAS score: <span>0.604</span></div>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Input ─────────────────────────────────────────────────────────────────────
claim = st.text_area(
    "Claim",
    placeholder="Enter any claim to verify...\ne.g. Pakistan is the 5th most populous country in the world",
    height=120,
    label_visibility="collapsed",
    key="claim_input"
)

col1, col2 = st.columns([3, 1])
with col1:
    run_btn = st.button("Verify Claim →")
with col2:
    use_hf = st.checkbox("HuggingFace enrichment", value=True)

# ── Run Pipeline ──────────────────────────────────────────────────────────────
if run_btn:
    if not claim.strip():
        st.warning("Enter a claim to verify.")
        st.stop()

    # Live agent status
    st.markdown('<div class="section-head">// Pipeline Running</div>',
                unsafe_allow_html=True)
    steps_ph = st.empty()
    steps_ph.markdown("""
    <div>
        <div class="agent-step"><div class="step-icon">🔄</div><div class="step-text">Checking semantic cache...</div></div>
        <div class="agent-step"><div class="step-icon">⏳</div><div class="step-text">Decomposing claim</div></div>
        <div class="agent-step"><div class="step-icon">⏳</div><div class="step-text">Collecting evidence (web + news + Wikipedia + RAG)</div></div>
        <div class="agent-step"><div class="step-icon">⏳</div><div class="step-text">Bias detection + credibility scoring</div></div>
        <div class="agent-step"><div class="step-icon">⏳</div><div class="step-text">Synthesizing verdict</div></div>
    </div>
    """, unsafe_allow_html=True)

    with st.spinner("Verifying..."):
        result = run_pipeline(claim.strip(), verbose=False, use_hf=use_hf)

    cache_hit = result.get("cache_hit", False)
    steps_ph.markdown(f"""
    <div>
        <div class="agent-step">
            <div class="step-icon">{'🗄️' if cache_hit else '✅'}</div>
            <div class="step-text">Cache {'hit' if cache_hit else 'miss'} — {'returned cached result' if cache_hit else 'running full pipeline'}</div>
            <span class="step-badge {'badge-cached' if cache_hit else 'badge-done'}">{'CACHED' if cache_hit else 'DONE'}</span>
        </div>
        {'<div class="agent-step"><div class="step-icon">✅</div><div class="step-text">Claim decomposed</div><span class="step-badge badge-done">DONE</span></div>' if not cache_hit else ''}
        {'<div class="agent-step"><div class="step-icon">✅</div><div class="step-text">Evidence collected from web + news + Wikipedia + RAG</div><span class="step-badge badge-done">DONE</span></div>' if not cache_hit else ''}
        {'<div class="agent-step"><div class="step-icon">✅</div><div class="step-text">Bias detected · Source credibility scored · HF enriched</div><span class="step-badge badge-done">DONE</span></div>' if not cache_hit else ''}
        <div class="agent-step"><div class="step-icon">✅</div><div class="step-text">Verdict synthesized in {result.get('processing_time', 0)}s</div><span class="step-badge badge-done">DONE</span></div>
    </div>
    """, unsafe_allow_html=True)

    # ── Verdict ───────────────────────────────────────────────────────────────
    verdict = result.get("final_verdict", "UNVERIFIABLE")
    confidence = result.get("confidence_score", 0)
    verdict_class = "CACHED" if cache_hit else verdict

    verdict_icons = {
        "TRUE": "✅", "FALSE": "❌", "MISLEADING": "⚠️",
        "PARTIALLY_TRUE": "🔶", "UNVERIFIABLE": "❓", "CACHED": "🗄️"
    }

    st.markdown(f"""
    <div class="verdict-box verdict-{verdict_class}">
        <div class="verdict-label">{'Cached Result' if cache_hit else 'Verdict'}</div>
        <div class="verdict-text">{verdict_icons.get(verdict_class, '❓')} {verdict}</div>
        <div class="confidence">Confidence: {confidence}% · {result.get('processing_time', 0)}s</div>
    </div>
    """, unsafe_allow_html=True)

    # ── Summary ───────────────────────────────────────────────────────────────
    st.markdown('<div class="section-head">// Summary</div>',
                unsafe_allow_html=True)
    st.markdown(f"""
    <div style="background:#FFF;border:1px solid #EBEBEB;border-radius:12px;padding:1.25rem 1.5rem;">
        <p style="font-size:0.92rem;color:#333;line-height:1.75;">
            {result.get('executive_summary', '')}
        </p>
        <p style="font-size:0.85rem;color:#888;line-height:1.7;margin-top:0.75rem;">
            {result.get('detailed_explanation', '')}
        </p>
    </div>
    """, unsafe_allow_html=True)

    col_left, col_right = st.columns(2)

    with col_left:
        # Key facts
        facts = result.get("key_facts", [])
        if facts:
            st.markdown('<div class="section-head">// Key Facts</div>',
                        unsafe_allow_html=True)
            facts_html = "".join(
                f'<div class="fact-item"><div class="fact-dot"></div>'
                f'<div class="fact-text">{f}</div></div>'
                for f in facts
            )
            st.markdown(
                f'<div style="background:#FFF;border:1px solid #EBEBEB;border-radius:12px;padding:1rem 1.25rem;">{facts_html}</div>', unsafe_allow_html=True)

        # Bias + Credibility
        bias = result.get("bias_analysis", {})
        cred = result.get("credibility_analysis", {})
        if bias or cred:
            st.markdown(
                '<div class="section-head">// Bias & Credibility</div>', unsafe_allow_html=True)
            st.markdown(f"""
            <div class="meta-grid">
                <div class="meta-card">
                    <div class="meta-label">Bias Direction</div>
                    <div class="meta-value">{bias.get('overall_bias_direction', 'unknown').title()}</div>
                    <div style="font-size:0.72rem;color:#AAA;margin-top:0.15rem;">{bias.get('bias_severity', 'none')} severity</div>
                </div>
                <div class="meta-card">
                    <div class="meta-label">Source Credibility</div>
                    <div class="meta-value">{cred.get('average_credibility', 0)}/100</div>
                    <div style="font-size:0.72rem;color:#AAA;margin-top:0.15rem;">{cred.get('credibility_label', '')}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    with col_right:
        # HF enrichment
        hf = result.get("hf_enrichment", {})
        clf = hf.get("hf_classification", {})
        entities = hf.get("hf_entities", [])
        if clf.get("claim_category"):
            st.markdown(
                '<div class="section-head">// HuggingFace Analysis</div>', unsafe_allow_html=True)
            ents_html = "".join(
                f'<span class="source-chip">{e["word"]} ({e["entity_type"]})</span>'
                for e in entities[:6]
            )
            st.markdown(f"""
            <div class="meta-grid">
                <div class="meta-card">
                    <div class="meta-label">Claim Category</div>
                    <div class="meta-value">{clf.get('claim_category', '').title()}</div>
                    <div style="font-size:0.72rem;color:#AAA;margin-top:0.15rem;">{clf.get('category_confidence', 0)}% confidence</div>
                </div>
                <div class="meta-card">
                    <div class="meta-label">Entities Detected</div>
                    <div style="margin-top:0.2rem;">{ents_html if ents_html else '<span style="font-size:0.8rem;color:#AAA;">None detected</span>'}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        # Sources
        sources = result.get("sources", [])
        if sources:
            st.markdown(
                '<div class="section-head">// Sources Used</div>', unsafe_allow_html=True)
            chips = "".join(
                f'<a href="{s}" target="_blank" class="source-chip">{s[:55]}...</a>'
                if len(s) > 55 else f'<a href="{s}" target="_blank" class="source-chip">{s}</a>'
                for s in sources[:6] if s
            )
            st.markdown(
                f'<div style="background:#FFF;border:1px solid #EBEBEB;border-radius:12px;padding:1rem 1.25rem;">{chips}</div>', unsafe_allow_html=True)

        # Sub-claim breakdown
        sub_results = result.get("sub_claim_results", [])
        if sub_results:
            st.markdown(
                '<div class="section-head">// Sub-Claim Breakdown</div>', unsafe_allow_html=True)
            for sc in sub_results:
                v = sc.get("verdict", "UNVERIFIABLE")
                color_map = {"TRUE": "#16A34A", "FALSE": "#DC2626", "MISLEADING": "#D97706",
                             "PARTIALLY_TRUE": "#2563EB", "UNVERIFIABLE": "#6B7280"}
                color = color_map.get(v, "#6B7280")
                st.markdown(f"""
                <div style="background:#FFF;border:1px solid #EBEBEB;border-left:3px solid {color};
                     border-radius:8px;padding:0.6rem 0.9rem;margin:0.3rem 0;">
                    <div style="font-size:0.72rem;font-weight:700;color:{color};">{v}</div>
                    <div style="font-size:0.8rem;color:#444;margin-top:0.1rem;">{sc.get('sub_claim_text', '')[:100]}</div>
                    <div style="font-size:0.7rem;color:#AAA;margin-top:0.15rem;">
                        Confidence: {sc.get('confidence', 0)}%
                    </div>
                </div>
                """, unsafe_allow_html=True)

else:
    st.markdown("""
    <div style="text-align:center;padding:4rem 2rem;background:#FFF;border:1.5px dashed #EBEBEB;border-radius:16px;margin-top:0.5rem;">
        <div style="font-family:'Fraunces',serif;font-size:2.5rem;color:#EBEBEB;margin-bottom:0.75rem;">⚖️</div>
        <div style="font-size:1rem;font-weight:600;color:#1A1A1A;margin-bottom:0.3rem;">Enter a claim above to verify</div>
        <div style="font-size:0.82rem;color:#AAA;max-width:440px;margin:0 auto;line-height:1.6;">
            The pipeline will decompose your claim, search for evidence across web,
            news, and Wikipedia, detect bias, score source credibility, and synthesize a verdict.
        </div>
        <div style="margin-top:1.5rem;display:flex;justify-content:center;gap:0.75rem;flex-wrap:wrap;">
            <span style="font-size:0.75rem;color:#AAA;background:#F9F9F9;padding:0.3rem 0.75rem;border-radius:999px;border:1px solid #EBEBEB;">
                Pakistan is the 5th most populous country
            </span>
            <span style="font-size:0.75rem;color:#AAA;background:#F9F9F9;padding:0.3rem 0.75rem;border-radius:999px;border:1px solid #EBEBEB;">
                Humans only use 10% of their brain
            </span>
            <span style="font-size:0.75rem;color:#AAA;background:#F9F9F9;padding:0.3rem 0.75rem;border-radius:999px;border:1px solid #EBEBEB;">
                Einstein failed math in school
            </span>
        </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")
st.markdown('<p style="text-align:center;color:#CCC;font-size:0.7rem;letter-spacing:0.1em;">VERITASAI · MULTI-AGENT VERIFICATION · LANGSMITH TRACED · RAGAS EVALUATED</p>', unsafe_allow_html=True)
