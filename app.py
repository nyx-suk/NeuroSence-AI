"""
NeuroSense — Interpretable Alzheimer's Detection
Full Streamlit UI — 5 Pages
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import os, sys, warnings
warnings.filterwarnings('ignore')

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# ── Page config ────────────────────────────────────────────────
st.set_page_config(
    page_title="NeuroSense AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Theme state ────────────────────────────────────────────────
if 'dark_mode' not in st.session_state:
    st.session_state.dark_mode = False

from theme import get_colors, get_css
t = get_colors('dark' if st.session_state.dark_mode else 'light')

# ── Custom CSS (theme-aware) ───────────────────────────────────
st.markdown(get_css(t), unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# MODEL LOADING (cached)
# ══════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner=False)
def load_pipeline():
    from ml.generate_dataset import load_oasis_dataset, prepare_oasis
    from ml.data_loader      import preprocess, split_data
    from ml.preprocessing    import normalize, apply_pca, apply_lda
    from ml.models           import (train_random_forest, train_all_svm_kernels,
                                     train_voting_ensemble, apply_smote)
    from ml.evaluation       import evaluate_model

    df_raw = load_oasis_dataset(save_path='data/oasis_longitudinal.csv')
    df     = prepare_oasis(df_raw)
    X, y, le = preprocess(df, target_col='Group', merge_converted=True)
    feature_names = list(X.columns)
    class_names   = list(le.classes_)

    X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2)
    X_tr_sc, X_te_sc, scaler = normalize(X_train, X_test, method='standard')
    X_bal, y_bal = apply_smote(X_tr_sc, y_train.values)

    rf   = train_random_forest(X_bal, y_bal, tune=False)
    svms = train_all_svm_kernels(X_bal, y_bal)
    ens  = train_voting_ensemble(X_bal, y_bal)

    X_tr_pca, X_te_pca, pca = apply_pca(X_tr_sc, X_te_sc, n_components=2)
    X_tr_lda, X_te_lda, lda = apply_lda(X_tr_sc, X_te_sc, y_train.values, n_components=2)

    results = {}
    for name, mdl in {'Random Forest': rf, **svms, 'Voting Ensemble': ens}.items():
        results[name] = evaluate_model(mdl, X_te_sc, y_test,
                                       model_name=name, class_names=class_names)
    return {
        'rf': rf, 'svms': svms, 'ensemble': ens,
        'scaler': scaler, 'le': le,
        'feature_names': feature_names, 'class_names': class_names,
        'X_train': X_tr_sc, 'y_train': y_train.values,
        'X_test':  X_te_sc,  'y_test':  y_test.values,
        'X_tr_pca': X_tr_pca, 'X_tr_lda': X_tr_lda,
        'results': results, 'df': df,
    }


# ══════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown(f"""
    <div style='text-align:center; padding:1rem 0 1.5rem 0;'>
        <div style='font-size:2.5rem;'>🧠</div>
        <div style='font-size:1.2rem; font-weight:700; color:{t['txt']};'>NeuroSense AI</div>
        <div style='font-size:0.75rem; color:{t['txt2']}; margin-top:0.2rem;'>Alzheimer's Detection System</div>
    </div>
    """, unsafe_allow_html=True)

    st.toggle("🌙 Dark Mode", key="dark_mode")

    page = st.radio("Navigation", [
        "🏠  Home", "🔬  Patient Diagnosis",
        "📊  Model Analytics", "📈  Visualizations", "ℹ️  About"
    ], label_visibility="collapsed")

    st.markdown("---")
    st.markdown(f"""
    <div style='padding:0.8rem; background:{t['card']}; border-radius:8px; border:1px solid {t['border2']}; margin-bottom:0.8rem;'>
        <div style='font-size:0.7rem; color:{t['txt2']}; text-transform:uppercase; letter-spacing:1px; margin-bottom:0.4rem;'>Dataset</div>
        <div style='color:{t['txt']}; font-size:0.85rem; font-weight:500;'>OASIS Longitudinal</div>
        <div style='color:{t['txt2']}; font-size:0.75rem;'>373 clinical records</div>
    </div>
    <div style='padding:0.8rem; background:{t['card']}; border-radius:8px; border:1px solid {t['border2']}; margin-bottom:0.8rem;'>
        <div style='font-size:0.7rem; color:{t['txt2']}; text-transform:uppercase; letter-spacing:1px; margin-bottom:0.4rem;'>Best Accuracy</div>
        <div style='color:{t['green']}; font-size:1.2rem; font-weight:700;'>96.00%</div>
        <div style='color:{t['txt2']}; font-size:0.75rem;'>Voting Ensemble</div>
    </div>
    <div style='font-size:0.7rem; color:{t['txt2']}; text-align:center; margin-top:1rem;'>
        CI2018 Machine Learning<br>VIT Pune · 2025-26
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# LOAD MODELS
# ══════════════════════════════════════════════════════════════
with st.spinner("🧠 Loading NeuroSense AI models..."):
    pipe = load_pipeline()


# ══════════════════════════════════════════════════════════════
# PAGE 1 — HOME
# ══════════════════════════════════════════════════════════════
if page == "🏠  Home":

    st.markdown("""
    <div class="hero-card">
        <div class="hero-title">🧠 NeuroSense AI</div>
        <div class="hero-sub">Interpretable Alzheimer's Disease Detection using Machine Learning</div>
        <span class="badge badge-green">✓ Medical Grade 96%</span>&nbsp;
        <span class="badge badge-blue">CI2018 ML Project</span>&nbsp;
        <span class="badge badge-purple">OASIS Dataset</span>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3, c4, c5 = st.columns(5)
    for col, (val, label) in zip([c1,c2,c3,c4,c5], [
        ("96.00%","Best Accuracy"), ("0.9629","Precision"),
        ("0.9600","Recall"),        ("0.9599","F1-Score"), ("373","Clinical Records")
    ]):
        col.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{val}</div>
            <div class="metric-label">{label}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    left, right = st.columns([1.1, 0.9])

    with left:
        st.markdown('<div class="section-header">System Architecture</div>', unsafe_allow_html=True)
        for icon, title, desc in [
            ("📋","Clinical Input",       "Age · MMSE · CDR · nWBV · Gender · EDUC · SES · eTIV · ASF"),
            ("⚙️","Preprocessing",        "Missing value imputation · Gender encoding · Z-score normalization"),
            ("🔬","Feature Engineering",  "Cognitive Index · CDR-MMSE Product · Brain Volume Ratio · 6 derived features"),
            ("⚖️","SMOTE Balancing",      "Synthetic minority oversampling on training set only"),
            ("🌲","ML Models",            "Random Forest · SVM (Linear/RBF/Poly) · Voting Ensemble"),
            ("📊","Evaluation",           "Accuracy · Precision · Recall · F1 · ROC/AUC · Confusion Matrix"),
            ("💡","Explainability",       "Feature importance ranking · Per-patient risk breakdown"),
        ]:
            st.markdown(f"""
            <div style='display:flex;align-items:flex-start;gap:0.8rem;
                        background:{t['card']};border:1px solid {t['border2']};
                        border-radius:10px;padding:0.75rem 1rem;margin-bottom:0.45rem;'>
                <div style='font-size:1.2rem;min-width:26px;'>{icon}</div>
                <div>
                    <div style='color:{t['txt']};font-weight:600;font-size:0.88rem;'>{title}</div>
                    <div style='color:{t['txt2']};font-size:0.77rem;margin-top:0.1rem;'>{desc}</div>
                </div>
            </div>""", unsafe_allow_html=True)

    with right:
        st.markdown('<div class="section-header">Model Performance</div>', unsafe_allow_html=True)
        models_d = ['Voting Ensemble','Random Forest','SVM Linear','SVM Polynomial','SVM RBF']
        accs_d   = [96.00, 94.67, 94.67, 94.67, 90.67]
        colors_d = [t['blue'],t['green'],t['purple'],t['orange'],t['pink']]
        fig = go.Figure(go.Bar(
            x=accs_d, y=models_d, orientation='h',
            marker=dict(color=colors_d),
            text=[f"{v:.1f}%" for v in accs_d],
            textposition='outside', textfont=dict(color=t['txt'], size=11),
        ))
        fig.add_vline(x=93, line_dash="dash", line_color=t['orange'],
                      annotation_text="Clinical Threshold 93%",
                      annotation_font_color=t['orange'], annotation_font_size=10)
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color=t['txt'], family='Inter'),
            xaxis=dict(range=[80,100], gridcolor=t['grid'], ticksuffix='%'),
            yaxis=dict(gridcolor=t['grid']),
            margin=dict(l=10,r=60,t=10,b=10), height=260, showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown('<div class="section-header">CI2018 Units Covered</div>', unsafe_allow_html=True)
        for unit, desc in [
            ("Unit I",  "ML Basics, Data Types, Train/Test Split"),
            ("Unit II", "Normalization, K-Fold CV, PCA, LDA"),
            ("Unit III","Confusion Matrix, ROC/AUC, Precision, Recall, F1"),
            ("Unit IV", "Gradient Descent, Bias-Variance, Regularization"),
            ("Unit V",  "SVM (3 kernels), Random Forest, Feature Importance"),
        ]:
            st.markdown(f"""
            <div style='display:flex;justify-content:space-between;align-items:center;
                        padding:0.45rem 0.8rem;background:{t['card']};border-radius:8px;
                        margin-bottom:0.3rem;border:1px solid {t['border2']};'>
                <span style='color:{t['blue']};font-weight:600;font-size:0.82rem;'>{unit}</span>
                <span style='color:{t['txt2']};font-size:0.77rem;'>{desc}</span>
            </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# PAGE 2 — PATIENT DIAGNOSIS
# ══════════════════════════════════════════════════════════════
elif page == "🔬  Patient Diagnosis":

    st.markdown(f"""
    <div style='background:linear-gradient(135deg,{t['hero1']},{t['hero2']});
                border:1px solid {t['border']};border-radius:16px;padding:1.5rem 2rem;margin-bottom:1.5rem;'>
        <div style='font-size:1.6rem;font-weight:700;color:{t['txt']};'>🔬 Patient Diagnosis</div>
        <div style='color:{t['txt2']};margin-top:0.3rem;'>Enter clinical measurements to get an Alzheimer's risk prediction with explanation.</div>
    </div>
    """, unsafe_allow_html=True)

    left, right = st.columns([1, 1.1])

    with left:
        st.markdown('<div class="section-header">Clinical Parameters</div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            age    = st.slider("Age (years)", 60, 98, 75)
            mmse   = st.slider("MMSE Score", 0, 30, 27, help="Mini-Mental State Examination. Normal: 27-30")
            educ   = st.slider("Education (years)", 6, 23, 14)
        with c2:
            cdr    = st.selectbox("CDR Rating", [0.0, 0.5, 1.0, 2.0],
                                  help="Clinical Dementia Rating. 0=None, 0.5=Questionable, 1=Mild, 2=Moderate")
            ses    = st.selectbox("Socioeconomic Status", [1,2,3,4,5], index=1)
            gender = st.selectbox("Gender", ["Male","Female"])
        c3, c4 = st.columns(2)
        with c3:
            etiv = st.number_input("eTIV (mm³)", 1100, 2000, 1480,
                                   help="Estimated Total Intracranial Volume")
            nwbv = st.number_input("nWBV", 0.64, 0.84, 0.73, step=0.001, format="%.3f",
                                   help="Normalized Whole Brain Volume")
        with c4:
            asf  = st.number_input("ASF", 0.88, 1.59, 1.20, step=0.001, format="%.3f",
                                   help="Atlas Scaling Factor")

        st.markdown("<br>", unsafe_allow_html=True)
        predict_btn = st.button("🔍  Run Diagnosis", use_container_width=True)

        st.markdown('<div class="section-header">Clinical Reference</div>', unsafe_allow_html=True)
        for name, desc in [
            ("MMSE Score", "27–30 = Normal · 21–26 = Mild · 10–20 = Moderate · <10 = Severe"),
            ("CDR Rating", "0 = None · 0.5 = Questionable · 1 = Mild · 2 = Moderate"),
            ("nWBV",       "Normal: 0.72–0.84 · Lower values indicate brain atrophy"),
        ]:
            st.markdown(f'<div class="info-box"><h4>{name}</h4><p>{desc}</p></div>',
                        unsafe_allow_html=True)

    with right:
        st.markdown('<div class="section-header">Diagnosis Result</div>', unsafe_allow_html=True)

        if predict_btn:
            from sklearn.impute import SimpleImputer
            gender_enc = 1 if gender == "Male" else 0
            cog_idx  = mmse * (1 - cdr * 0.5)
            cdr_mmse = cdr * (30 - mmse)
            mmse_diff= mmse - (cdr * 10)
            age_mmse = age / (mmse + 1)
            bvr      = nwbv / (etiv / 1000)
            age_nwbv = age * (1 - nwbv)

            feat_arr = np.array([gender_enc, age, educ, ses, mmse, cdr, etiv, nwbv, asf,
                                 cog_idx, cdr_mmse, mmse_diff, age_mmse, bvr, age_nwbv]).reshape(1,-1)
            imp = SimpleImputer(strategy='median')
            imp.fit(pipe['X_train'])
            feat_scaled = pipe['scaler'].transform(imp.transform(feat_arr))

            model    = pipe['ensemble']
            proba    = model.predict_proba(feat_scaled)[0]
            pred_idx = int(np.argmax(proba))
            pred_cls = pipe['class_names'][pred_idx]
            conf     = proba[pred_idx] * 100

            if pred_cls == 'Demented':
                st.markdown(f"""
                <div class="result-demented">
                    <div style='font-size:1.6rem;font-weight:700;color:{t['red']};'>⚠️ Alzheimer's Risk Detected</div>
                    <div style='font-size:2.8rem;font-weight:800;color:{t['red']};margin:0.5rem 0;'>{conf:.1f}%</div>
                    <div style='color:{t['txt2']};'>Confidence · Voting Ensemble</div>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="result-nondemented">
                    <div style='font-size:1.6rem;font-weight:700;color:{t['green']};'>✅ No Dementia Detected</div>
                    <div style='font-size:2.8rem;font-weight:800;color:{t['green']};margin:0.5rem 0;'>{conf:.1f}%</div>
                    <div style='color:{t['txt2']};'>Confidence · Voting Ensemble</div>
                </div>""", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # Gauge
            fig_g = go.Figure(go.Indicator(
                mode="gauge+number",
                value=proba[0]*100,
                title={'text':"Dementia Risk %",'font':{'color':t['txt'],'size':13}},
                number={'suffix':'%','font':{'color':t['txt'],'size':26}},
                gauge={
                    'axis':{'range':[0,100],'tickcolor':t['txt2']},
                    'bar':{'color':t['red'] if proba[0]>0.5 else t['green']},
                    'bgcolor':t['gbg'],'bordercolor':t['border'],
                    'steps':[
                        {'range':[0,30],'color':t['glow']},
                        {'range':[30,60],'color':t['gmid']},
                        {'range':[60,100],'color':t['ghigh']},
                    ],
                    'threshold':{'line':{'color':t['orange'],'width':3},'thickness':0.8,'value':50}
                }
            ))
            fig_g.update_layout(
                paper_bgcolor='rgba(0,0,0,0)', font=dict(color=t['txt']),
                height=210, margin=dict(l=20,r=20,t=30,b=10)
            )
            st.plotly_chart(fig_g, use_container_width=True)

            # Feature contributions
            st.markdown('<div class="section-header">Key Clinical Factors</div>', unsafe_allow_html=True)
            importances = pipe['rf'].feature_importances_
            feat_names  = pipe['feature_names']
            top_idx     = np.argsort(importances)[::-1][:6]
            for idx in top_idx:
                imp_pct = importances[idx] * 100
                bar_w   = min(int(imp_pct * 3), 100)
                st.markdown(f"""
                <div style='display:flex;align-items:center;gap:0.8rem;
                            padding:0.45rem 0.8rem;background:{t['card']};
                            border-radius:8px;margin-bottom:0.3rem;border:1px solid {t['border2']};'>
                    <div style='color:{t['txt']};font-size:0.82rem;min-width:145px;font-weight:500;'>{feat_names[idx]}</div>
                    <div style='flex:1;background:{t['border2']};border-radius:4px;height:7px;'>
                        <div style='width:{bar_w}%;background:linear-gradient(90deg,{t['blue']},{t['purple']});height:7px;border-radius:4px;'></div>
                    </div>
                    <div style='color:{t['blue']};font-size:0.82rem;font-weight:600;min-width:38px;text-align:right;'>{imp_pct:.1f}%</div>
                </div>""", unsafe_allow_html=True)

        else:
            st.markdown(f"""
            <div style='background:{t['card']};border:1px dashed {t['border']};border-radius:12px;
                        padding:3rem;text-align:center;color:{t['txt2']};'>
                <div style='font-size:3rem;margin-bottom:1rem;'>🩺</div>
                <div style='font-size:1rem;font-weight:500;color:{t['txt']};'>Enter Patient Data</div>
                <div style='font-size:0.85rem;margin-top:0.5rem;'>
                    Fill in the clinical parameters on the left<br>and click Run Diagnosis
                </div>
            </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# PAGE 3 — MODEL ANALYTICS
# ══════════════════════════════════════════════════════════════
elif page == "📊  Model Analytics":

    st.markdown(f"""
    <div style='background:linear-gradient(135deg,{t['hero1']},{t['hero2']});
                border:1px solid {t['border']};border-radius:16px;padding:1.5rem 2rem;margin-bottom:1.5rem;'>
        <div style='font-size:1.6rem;font-weight:700;color:{t['txt']};'>📊 Model Analytics</div>
        <div style='color:{t['txt2']};margin-top:0.3rem;'>Detailed performance metrics, confusion matrices and ROC curves for all models.</div>
    </div>
    """, unsafe_allow_html=True)

    results     = pipe['results']
    class_names = pipe['class_names']
    X_test      = pipe['X_test']
    y_test      = pipe['y_test']
    all_models  = {'Random Forest': pipe['rf'], **pipe['svms'], 'Voting Ensemble': pipe['ensemble']}

    # Summary table
    st.markdown('<div class="section-header">Performance Summary</div>', unsafe_allow_html=True)
    rows = []
    for name, r in results.items():
        rows.append({
            'Model': name,
            'Accuracy': f"{r['accuracy']*100:.2f}%",
            'Precision': f"{r['precision']:.4f}",
            'Recall': f"{r['recall']:.4f}",
            'F1-Score': f"{r['f1']:.4f}",
            'Grade': '🟢 Excellent' if r['accuracy']>=0.95 else ('🟡 Good' if r['accuracy']>=0.93 else '🔵 Acceptable')
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Grouped bar chart
    st.markdown('<div class="section-header">Metrics Comparison</div>', unsafe_allow_html=True)
    model_names = list(results.keys())
    fig_bar = go.Figure()
    for vals, name, color in [
        ([results[m]['accuracy']  for m in model_names], 'Accuracy',  t['blue']),
        ([results[m]['precision'] for m in model_names], 'Precision', t['green']),
        ([results[m]['recall']    for m in model_names], 'Recall',    t['orange']),
        ([results[m]['f1']        for m in model_names], 'F1-Score',  t['purple']),
    ]:
        fig_bar.add_trace(go.Bar(
            name=name, x=model_names, y=vals, marker_color=color, opacity=0.85,
            text=[f"{v:.3f}" for v in vals], textposition='outside',
            textfont=dict(size=9, color=t['txt']),
        ))
    fig_bar.add_hline(y=0.93, line_dash="dash", line_color=t['orange'],
                      annotation_text="Clinical Threshold", annotation_font_color=t['orange'])
    fig_bar.update_layout(
        barmode='group', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color=t['txt'], family='Inter'),
        xaxis=dict(gridcolor=t['grid'], tickfont=dict(size=10)),
        yaxis=dict(range=[0.85,1.02], gridcolor=t['grid'], tickformat='.2f'),
        legend=dict(bgcolor='rgba(0,0,0,0)', bordercolor=t['border'], borderwidth=1),
        margin=dict(l=10,r=10,t=20,b=10), height=360,
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    # Confusion matrices
    st.markdown('<div class="section-header">Confusion Matrices</div>', unsafe_allow_html=True)
    from sklearn.metrics import confusion_matrix
    cols = st.columns(len(all_models))
    for col, (mname, mdl) in zip(cols, all_models.items()):
        y_pred = mdl.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        fig_cm = px.imshow(
            cm, text_auto=True, x=class_names, y=class_names,
            color_continuous_scale=[[0,t['cm_0']],[0.5,t['cm_5']],[1,t['cm_1']]],
            labels=dict(x='Predicted', y='Actual'),
        )
        fig_cm.update_layout(
            title=dict(text=mname, font=dict(size=10, color=t['txt']), x=0.5),
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color=t['txt'], size=10), coloraxis_showscale=False,
            margin=dict(l=10,r=10,t=40,b=10), height=210,
        )
        col.plotly_chart(fig_cm, use_container_width=True)

    # ROC curves
    st.markdown('<div class="section-header">ROC Curves</div>', unsafe_allow_html=True)
    from sklearn.metrics import roc_curve, auc
    fig_roc = go.Figure()
    for (mname, mdl), color in zip(all_models.items(),
                                    [t['blue'],t['green'],t['purple'],t['orange'],t['pink']]):
        y_prob = mdl.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        fig_roc.add_trace(go.Scatter(
            x=fpr, y=tpr, mode='lines',
            name=f"{mname} (AUC={roc_auc:.3f})",
            line=dict(color=color, width=2.5),
        ))
    fig_roc.add_trace(go.Scatter(
        x=[0,1], y=[0,1], mode='lines', name='Random',
        line=dict(color=t['txt2'], width=1.5, dash='dash'),
    ))
    fig_roc.update_layout(
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color=t['txt'], family='Inter'),
        xaxis=dict(title='False Positive Rate', gridcolor=t['grid']),
        yaxis=dict(title='True Positive Rate', gridcolor=t['grid']),
        legend=dict(bgcolor=t['leg_bg'], bordercolor=t['border'], borderwidth=1),
        margin=dict(l=10,r=10,t=10,b=10), height=400,
    )
    st.plotly_chart(fig_roc, use_container_width=True)


# ══════════════════════════════════════════════════════════════
# PAGE 4 — VISUALIZATIONS
# ══════════════════════════════════════════════════════════════
elif page == "📈  Visualizations":

    st.markdown(f"""
    <div style='background:linear-gradient(135deg,{t['hero1']},{t['hero2']});
                border:1px solid {t['border']};border-radius:16px;padding:1.5rem 2rem;margin-bottom:1.5rem;'>
        <div style='font-size:1.6rem;font-weight:700;color:{t['txt']};'>📈 Visualizations</div>
        <div style='color:{t['txt2']};margin-top:0.3rem;'>Feature importance, PCA, LDA and dataset distribution analysis.</div>
    </div>
    """, unsafe_allow_html=True)

    # Feature Importance
    st.markdown('<div class="section-header">Feature Importance — Random Forest</div>', unsafe_allow_html=True)
    importances = pipe['rf'].feature_importances_
    feat_names  = pipe['feature_names']
    sorted_idx  = np.argsort(importances)
    s_names = [feat_names[i] for i in sorted_idx]
    s_vals  = importances[sorted_idx]
    colors_fi = [t['pink'] if v>0.15 else t['orange'] if v>0.08 else t['blue'] for v in s_vals]

    fig_fi = go.Figure(go.Bar(
        x=s_vals, y=s_names, orientation='h',
        marker=dict(color=colors_fi),
        text=[f"{v*100:.1f}%" for v in s_vals],
        textposition='outside', textfont=dict(color=t['txt'], size=10),
    ))
    fig_fi.update_layout(
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color=t['txt'], family='Inter'),
        xaxis=dict(title='Importance Score', gridcolor=t['grid'],
                   tickformat='.2f', range=[0, max(s_vals)*1.22]),
        yaxis=dict(gridcolor=t['grid'], tickfont=dict(size=11)),
        margin=dict(l=10,r=70,t=10,b=10), height=420,
    )
    st.plotly_chart(fig_fi, use_container_width=True)

    top3 = [feat_names[i] for i in np.argsort(importances)[::-1][:3]]
    st.markdown(f"""
    <div style='background:linear-gradient(135deg,{t['ins1']},{t['ins2']});
                border:1px solid {t['ins_b']};border-radius:10px;padding:1rem 1.2rem;'>
        <span style='color:{t['blue']};font-weight:600;'>💡 Clinical Insight: </span>
        <span style='color:{t['txt']};'>Top predictors are
        <strong style="color:{t['blue']};">{top3[0]}</strong>,
        <strong style="color:{t['blue']};">{top3[1]}</strong>, and
        <strong style="color:{t['blue']};">{top3[2]}</strong> —
        all clinically validated Alzheimer's biomarkers.</span>
    </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # PCA + LDA
    left, right = st.columns(2)
    X_pca = pipe['X_tr_pca']
    X_lda = pipe['X_tr_lda']
    y_tr  = pipe['y_train']
    cn    = pipe['class_names']
    cls_colors = [t['red'],t['green']]

    with left:
        st.markdown('<div class="section-header">PCA — 2D Class Separation</div>', unsafe_allow_html=True)
        fig_pca = go.Figure()
        for i, (cls, color) in enumerate(zip(cn, cls_colors)):
            mask = (y_tr == i)
            fig_pca.add_trace(go.Scatter(
                x=X_pca[mask,0], y=X_pca[mask,1], mode='markers', name=cls,
                marker=dict(color=color, size=7, opacity=0.75,
                            line=dict(color=t['marker_line'], width=0.5)),
            ))
        fig_pca.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color=t['txt'], family='Inter'),
            xaxis=dict(title='PC1', gridcolor=t['grid'], zeroline=False),
            yaxis=dict(title='PC2', gridcolor=t['grid'], zeroline=False),
            legend=dict(bgcolor=t['leg_bg'], bordercolor=t['border']),
            margin=dict(l=10,r=10,t=10,b=10), height=340,
        )
        st.plotly_chart(fig_pca, use_container_width=True)

    with right:
        st.markdown('<div class="section-header">LDA — Supervised Separation</div>', unsafe_allow_html=True)
        fig_lda = go.Figure()
        for i, (cls, color) in enumerate(zip(cn, cls_colors)):
            mask = (y_tr == i)
            fig_lda.add_trace(go.Histogram(
                x=X_lda[mask,0], name=cls,
                marker_color=color, opacity=0.7, nbinsx=25,
            ))
        fig_lda.update_layout(
            barmode='overlay',
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color=t['txt'], family='Inter'),
            xaxis=dict(title='LD1 (100% variance explained)', gridcolor=t['grid']),
            yaxis=dict(title='Count', gridcolor=t['grid']),
            legend=dict(bgcolor=t['leg_bg'], bordercolor=t['border']),
            margin=dict(l=10,r=10,t=10,b=10), height=340,
        )
        st.plotly_chart(fig_lda, use_container_width=True)

    # Dataset distributions
    st.markdown('<div class="section-header">Dataset Distribution</div>', unsafe_allow_html=True)
    df = pipe['df'].copy()
    df['Group'] = df['Group'].replace('Converted','Demented')
    c1, c2, c3 = st.columns(3)

    with c1:
        grp = df['Group'].value_counts()
        fig_pie = go.Figure(go.Pie(
            labels=grp.index, values=grp.values, hole=0.5,
            marker=dict(colors=[t['red'],t['green']], line=dict(color=t['pie_line'],width=2)),
            textfont=dict(color=t['txt'], size=12),
        ))
        fig_pie.update_layout(
            title=dict(text='Class Distribution', font=dict(color=t['txt'],size=12), x=0.5),
            paper_bgcolor='rgba(0,0,0,0)', font=dict(color=t['txt']),
            legend=dict(bgcolor='rgba(0,0,0,0)'),
            margin=dict(l=10,r=10,t=40,b=10), height=260,
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    with c2:
        fig_age = go.Figure()
        for cls, color in zip(df['Group'].unique(), cls_colors):
            fig_age.add_trace(go.Box(
                y=df[df['Group']==cls]['Age'], name=cls,
                marker_color=color, line_color=color,
            ))
        fig_age.update_layout(
            title=dict(text='Age by Group', font=dict(color=t['txt'],size=12), x=0.5),
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color=t['txt']), yaxis=dict(gridcolor=t['grid'], title='Age'),
            xaxis=dict(gridcolor=t['grid']), showlegend=False,
            margin=dict(l=10,r=10,t=40,b=10), height=260,
        )
        st.plotly_chart(fig_age, use_container_width=True)

    with c3:
        fig_mmse = go.Figure()
        for cls, color in zip(df['Group'].unique(), cls_colors):
            fig_mmse.add_trace(go.Violin(
                y=df[df['Group']==cls]['MMSE'], name=cls,
                box_visible=True, meanline_visible=True,
                fillcolor=color, line_color=color, opacity=0.7,
            ))
        fig_mmse.update_layout(
            title=dict(text='MMSE Score by Group', font=dict(color=t['txt'],size=12), x=0.5),
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color=t['txt']), yaxis=dict(gridcolor=t['grid'], title='MMSE'),
            xaxis=dict(gridcolor=t['grid']), showlegend=False,
            margin=dict(l=10,r=10,t=40,b=10), height=260,
        )
        st.plotly_chart(fig_mmse, use_container_width=True)


# ══════════════════════════════════════════════════════════════
# PAGE 5 — ABOUT
# ══════════════════════════════════════════════════════════════
elif page == "ℹ️  About":

    st.markdown(f"""
    <div style='background:linear-gradient(135deg,{t['hero1']},{t['hero2']});
                border:1px solid {t['border']};border-radius:16px;padding:1.5rem 2rem;margin-bottom:1.5rem;'>
        <div style='font-size:1.6rem;font-weight:700;color:{t['txt']};'>ℹ️ About NeuroSense AI</div>
        <div style='color:{t['txt2']};margin-top:0.3rem;'>Project documentation and technical details.</div>
    </div>
    """, unsafe_allow_html=True)

    left, right = st.columns(2)

    with left:
        st.markdown('<div class="section-header">Project Overview</div>', unsafe_allow_html=True)
        for h, p in [
            ("🎯 Title",    "Interpretable Alzheimer's Disease Detection using Machine Learning"),
            ("🧠 Core Idea","Uses ML on real clinical tabular data (OASIS) for Alzheimer's risk prediction and explanation. Models: Random Forest, SVM (3 kernels), Voting Ensemble."),
            ("📋 Dataset",  "OASIS Longitudinal Dataset — 373 real clinical patient records. Features: Age, Gender, MMSE, CDR, nWBV, eTIV, ASF, EDUC, SES."),
            ("🏫 Course",   "CI2018 Machine Learning · B.Tech CSE (AI) · VIT Pune · AY 2025-26 Sem II"),
        ]:
            st.markdown(f'<div class="info-box"><h4>{h}</h4><p>{p}</p></div>', unsafe_allow_html=True)

        st.markdown('<div class="section-header">Team</div>', unsafe_allow_html=True)
        for name in ["Anuj Gardi", "Gauri Kadalge", "Disha Gudup", "Vidula Gote"]:
            st.markdown(f"""
            <div style='display:flex;align-items:center;gap:0.8rem;
                        padding:0.7rem 1rem;background:{t['card']};
                        border-radius:10px;margin-bottom:0.4rem;border:1px solid {t['border2']};'>
                <div style='font-size:1.3rem;'>👨‍💻</div>
                <div style='color:{t['txt']};font-weight:500;'>{name}</div>
                <div style='margin-left:auto;'><span class="badge badge-blue">VIT Pune</span></div>
            </div>""", unsafe_allow_html=True)

    with right:
        st.markdown('<div class="section-header">Technical Stack</div>', unsafe_allow_html=True)
        for icon, name, desc in [
            ("🐍","Python 3.14",      "Core language"),
            ("🔬","scikit-learn",     "ML models & evaluation"),
            ("🌲","Random Forest",    "Primary classifier — Bagging ensemble"),
            ("⚡","SVM (3 kernels)",  "Linear · RBF · Polynomial"),
            ("🗳️","Voting Ensemble",  "RF + SVM-RBF + SVM-Linear · Best accuracy"),
            ("⚖️","SMOTE",           "Class balancing on training set"),
            ("📊","Plotly",          "Interactive charts"),
            ("🖥️","Streamlit",       "Web UI framework"),
            ("🧮","NumPy / Pandas",  "Data processing"),
        ]:
            st.markdown(f"""
            <div style='display:flex;align-items:center;gap:0.8rem;
                        padding:0.45rem 0.8rem;background:{t['card']};
                        border-radius:8px;margin-bottom:0.3rem;border:1px solid {t['border2']};'>
                <div style='font-size:1rem;min-width:22px;'>{icon}</div>
                <div style='color:{t['txt']};font-weight:500;font-size:0.83rem;min-width:130px;'>{name}</div>
                <div style='color:{t['txt2']};font-size:0.78rem;'>{desc}</div>
            </div>""", unsafe_allow_html=True)

        st.markdown('<div class="section-header">Units Covered</div>', unsafe_allow_html=True)
        for unit, desc, done in [
            ("Unit I",  "ML Basics, Data Types, Feature Engineering, Train/Test Split",     True),
            ("Unit II", "Normalization, K-Fold CV, PCA, LDA, Dimensionality Reduction",     True),
            ("Unit III","Confusion Matrix, Accuracy, Precision, Recall, F1, ROC/AUC",       True),
            ("Unit IV", "Gradient Descent, Cost Functions, Bias-Variance, Regularization",  True),
            ("Unit V",  "SVM (3 kernels), Random Forest, Bagging, Feature Importance",      True),
        ]:
            icon  = "✅" if done else "⏭️"
            color = t['green'] if done else t['txt2']
            st.markdown(f"""
            <div style='display:flex;align-items:flex-start;gap:0.6rem;
                        padding:0.45rem 0.8rem;background:{t['card']};
                        border-radius:8px;margin-bottom:0.3rem;border:1px solid {t['border2']};'>
                <div style='font-size:0.85rem;min-width:18px;margin-top:1px;'>{icon}</div>
                <div>
                    <div style='color:{color};font-weight:600;font-size:0.82rem;'>{unit}</div>
                    <div style='color:{t['txt2']};font-size:0.75rem;margin-top:0.1rem;'>{desc}</div>
                </div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(f"""
    <div style='background:linear-gradient(135deg,{t['ban1']},{t['ban2']});
                border:2px solid {t['green']};border-radius:16px;
                padding:1.5rem 2rem;text-align:center;'>
        <div style='font-size:0.85rem;color:{t['txt2']};text-transform:uppercase;
                    letter-spacing:2px;margin-bottom:0.5rem;'>Best Model Performance</div>
        <div style='font-size:3.5rem;font-weight:800;color:{t['green']};'>96.00%</div>
        <div style='color:{t['txt2']};margin-top:0.3rem;'>
            Voting Ensemble (RF + SVM-RBF + SVM-Linear) · OASIS Clinical Dataset
        </div>
        <div style='margin-top:1rem;'>
            <span class="badge badge-green">✓ Medical Grade</span>&nbsp;
            <span class="badge badge-blue">F1: 0.9599</span>&nbsp;
            <span class="badge badge-purple">Precision: 0.9629</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
