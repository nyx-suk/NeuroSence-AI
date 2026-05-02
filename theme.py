"""
NeuroSense AI — Theme System
Light / Dark mode color palettes and CSS generation.
"""

def get_colors(mode='light'):
    """Return a dict of named colors for the given theme mode."""
    if mode == 'dark':
        return dict(
            bg='#0a0e1a',
            sb1='#0d1117', sb2='#161b27', sb_border='#21262d',
            card='#161b27', card2='#1c2333',
            hero1='#1a1f35', hero2='#0d1117',
            border='#30363d', border2='#21262d',
            txt='#e6edf3', txt2='#8b949e',
            blue='#58a6ff', green='#3fb950', red='#f85149',
            purple='#bc8cff', orange='#f0883e', pink='#ff7b72',
            hg1='#58a6ff', hg2='#bc8cff', hg3='#ff7b72',
            bg_green='#1a3d2b', bg_blue='#1a2d4a', bg_purple='#2d1a4a',
            dem1='#3d1a1a', dem2='#2d1515',
            ndem1='#1a3d2b', ndem2='#152d1f',
            glow='#1a3d2b', gmid='#3d3a1a', ghigh='#3d1a1a',
            gbg='#161b27',
            grid='#21262d',
            ins1='#1a2d4a', ins2='#161b27', ins_b='#1f6feb',
            ban1='#1a3d2b', ban2='#152d1f',
            btn1='#1f6feb', btn2='#388bfd',
            leg_bg='rgba(22,27,39,0.9)',
            pie_line='#0d1117',
            marker_line='#0d1117',
            cm_0='#161b27', cm_5='#1f6feb', cm_1='#58a6ff',
        )
    else:
        return dict(
            bg='#f5f7fa',
            sb1='#ffffff', sb2='#f0f2f5', sb_border='#d0d7de',
            card='#ffffff', card2='#f6f8fa',
            hero1='#e8f0fe', hero2='#ffffff',
            border='#d0d7de', border2='#e1e4e8',
            txt='#1f2328', txt2='#656d76',
            blue='#0969da', green='#1a7f37', red='#cf222e',
            purple='#8250df', orange='#bc4c00', pink='#a40e26',
            hg1='#0969da', hg2='#8250df', hg3='#cf222e',
            bg_green='#dafbe1', bg_blue='#ddf4ff', bg_purple='#fbefff',
            dem1='#ffebe9', dem2='#fff1e5',
            ndem1='#dafbe1', ndem2='#e6ffec',
            glow='#dafbe1', gmid='#fff8c5', ghigh='#ffebe9',
            gbg='#f6f8fa',
            grid='#e1e4e8',
            ins1='#ddf4ff', ins2='#f6f8fa', ins_b='#0969da',
            ban1='#dafbe1', ban2='#e6ffec',
            btn1='#0969da', btn2='#218bff',
            leg_bg='rgba(255,255,255,0.9)',
            pie_line='#ffffff',
            marker_line='#ffffff',
            cm_0='#f6f8fa', cm_5='#218bff', cm_1='#0969da',
        )


def get_css(t):
    """Return the full CSS style block using theme colors dict `t`."""
    return f"""<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
html, body, [class*="css"] {{ font-family: 'Inter', sans-serif; }}
.main {{ background: {t['bg']}; }}
[data-testid="stAppViewContainer"] {{ background: {t['bg']}; }}
[data-testid="stApp"] {{ background: {t['bg']}; }}
[data-testid="stHeader"] {{ background: {t['bg']}; }}
[data-testid="stToolbar"] {{ background: {t['bg']}; }}
[data-testid="stBottom"] {{ background: {t['bg']}; }}
[data-testid="stSidebar"] {{
    background: linear-gradient(180deg, {t['sb1']} 0%, {t['sb2']} 100%);
    border-right: 1px solid {t['sb_border']};
}}
.hero-card {{
    background: linear-gradient(135deg, {t['hero1']} 0%, {t['hero2']} 50%, {t['hero1']} 100%);
    border: 1px solid {t['border']}; border-radius: 16px;
    padding: 2.5rem; text-align: center; margin-bottom: 1.5rem;
}}
.hero-title {{
    font-size: 3rem; font-weight: 700;
    background: linear-gradient(135deg, {t['hg1']}, {t['hg2']}, {t['hg3']});
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    margin-bottom: 0.5rem;
}}
.hero-sub {{ color: {t['txt2']}; font-size: 1.1rem; margin-bottom: 1.5rem; }}
.metric-card {{
    background: linear-gradient(135deg, {t['card']}, {t['card2']});
    border: 1px solid {t['border']}; border-radius: 12px;
    padding: 1.2rem 1rem; text-align: center;
    transition: transform 0.2s, border-color 0.2s;
    min-height: 120px;
    display: flex; flex-direction: column; justify-content: center; align-items: center;
}}
.metric-card:hover {{ transform: translateY(-2px); border-color: {t['blue']}; }}
.metric-value {{ font-size: 1.9rem; font-weight: 700; color: {t['blue']}; white-space: nowrap; }}
.metric-label {{ font-size: 0.75rem; color: {t['txt2']}; text-transform: uppercase; letter-spacing: 1px; margin-top: 0.2rem; white-space: nowrap; }}
.section-header {{
    font-size: 1.3rem; font-weight: 600; color: {t['txt']};
    border-left: 4px solid {t['blue']}; padding-left: 0.8rem;
    margin: 1.5rem 0 1rem 0;
}}
.result-demented {{
    background: linear-gradient(135deg, {t['dem1']}, {t['dem2']});
    border: 2px solid {t['red']}; border-radius: 16px; padding: 2rem; text-align: center;
}}
.result-nondemented {{
    background: linear-gradient(135deg, {t['ndem1']}, {t['ndem2']});
    border: 2px solid {t['green']}; border-radius: 16px; padding: 2rem; text-align: center;
}}
.info-box {{
    background: {t['card']}; border: 1px solid {t['border']};
    border-radius: 10px; padding: 1rem 1.2rem; margin-bottom: 0.8rem;
}}
.info-box h4 {{ color: {t['blue']}; margin: 0 0 0.4rem 0; font-size: 0.95rem; }}
.info-box p  {{ color: {t['txt2']}; margin: 0; font-size: 0.85rem; line-height: 1.5; }}
.badge {{
    display: inline-block; padding: 0.25rem 0.75rem;
    border-radius: 20px; font-size: 0.75rem; font-weight: 600; letter-spacing: 0.5px;
}}
.badge-green  {{ background: {t['bg_green']}; color: {t['green']}; border: 1px solid {t['green']}; }}
.badge-blue   {{ background: {t['bg_blue']}; color: {t['blue']}; border: 1px solid {t['blue']}; }}
.badge-purple {{ background: {t['bg_purple']}; color: {t['purple']}; border: 1px solid {t['purple']}; }}
.stButton > button {{
    background: linear-gradient(135deg, {t['btn1']}, {t['btn2']}) !important;
    color: white !important; border: none !important;
    border-radius: 8px !important; font-weight: 600 !important;
    padding: 0.6rem 2rem !important; font-size: 1rem !important;
}}
.stButton > button:hover {{ box-shadow: 0 4px 15px rgba(31,111,235,0.4) !important; }}
.stSelectbox label, .stSlider label, .stNumberInput label {{ color: {t['txt2']} !important; font-size: 0.85rem !important; }}
</style>"""
