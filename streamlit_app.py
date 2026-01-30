import streamlit as st
import time
import pandas as pd
import plotly.express as px
from main import (
    PrimalityTestManager, TestType, Result, TestResult
)

# Configuration de la page
st.set_page_config(
    page_title="PrimeCheck Pro",
    page_icon="∑",
    layout="wide",
)

# CSS Moderne Glassmorphism
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600&display=swap');

    :root {
        --glass-bg: rgba(255, 255, 255, 0.05);
        --glass-border: rgba(255, 255, 255, 0.1);
        --accent-gradient: linear-gradient(135deg, #00A8E8 0%, #0077B6 100%);
        --success-gradient: linear-gradient(135deg, #26C485 0%, #1A8B5F 100%);
        --error-gradient: linear-gradient(135deg, #FF5757 0%, #D43F3F 100%);
        --text-primary: #FFFFFF;
        --text-secondary: #B0B0B0;
    }

    html, body, [class*="st-"] {
        font-family: 'Inter', sans-serif;
    }

    .main {
        background: radial-gradient(circle at 50% 50%, #121821 0%, #080B10 100%);
        color: var(--text-primary);
    }

    /* Sidebar Glassmorphism */
    [data-testid="stSidebar"] {
        background-color: rgba(15, 20, 25, 0.7) !important;
        backdrop-filter: blur(15px);
        border-right: 1px solid var(--glass-border);
    }

    /* Cartes Glassmorphism */
    .glass-card {
        background: var(--glass-bg);
        backdrop-filter: blur(10px);
        border: 1px solid var(--glass-border);
        border-radius: 12px;
        padding: 24px;
        margin-bottom: 20px;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    
    .glass-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.4);
        border: 1px solid rgba(255, 255, 255, 0.2);
    }

    .result-badge {
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        text-transform: uppercase;
    }
    
    .prime-badge { background: var(--success-gradient); }
    .composite-badge { background: var(--error-gradient); }
    .prob-prime-badge { background: linear-gradient(135deg, #FFB84D 0%, #E6951D 100%); }

    h1, h2, h3 {
        background: var(--accent-gradient);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 600;
    }

    .stButton>button {
        background: var(--accent-gradient);
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 0.6rem 1.2rem !important;
        font-weight: 600 !important;
        transition: all 0.3s ease !important;
        width: 100%;
    }

    .stButton>button:hover {
        opacity: 0.9;
        transform: scale(1.02);
    }

    /* Style des métriques */
    [data-testid="stMetricValue"] {
        font-size: 1.8rem !important;
        font-weight: 600 !important;
    }
</style>
""", unsafe_allow_html=True)

# Initialisation de l'état de session
if 'history' not in st.session_state:
    st.session_state.history = []
if 'manager' not in st.session_state:
    st.session_state.manager = PrimalityTestManager()

def get_complexity_info(test_type_value, n, iterations=10):
    complexities = {
        "Miller-Rabin": ("O(k·log³n)", f"4⁻ᵏ (≈4⁻{iterations})"),
        "Solovay-Strassen": ("O(k·log³n)", f"2⁻ᵏ (≈2⁻{iterations})"),
        "Fermat": ("O(log³n)", "Peu fiable (Exception de Carmichael)"),
        "Baillie-PSW": ("O(log³n)", "Aucun contre-exemple connu"),
        "Lucas-Lehmer": ("O(p·log²p)", "Déterministe (Mersenne uniquement)"),
        "Trial Division": ("O(√n)", "Déterministe"),
        "AKS (Deterministic)": ("O(log⁶n)", "Zéro prouvé")
    }
    return complexities.get(test_type_value, ("O(?)", "?"))

# En-tête
st.title("PrimeCheck Pro")
st.markdown("<p style='opacity: 0.7; margin-bottom: 2rem;'>Suite avancée de test de primalité</p>", unsafe_allow_html=True)

# Barre latérale
with st.sidebar:
    st.markdown("### Configuration")
    
    # Exemples
    examples = {
        "Personnalisé": None,
        "Premier (104729)": 104729,
        "Composé (888888)": 888888,
        "Carmichael (561)": 561,
        "Premier de Mersenne (2^31-1)": 2**31 - 1,
    }
    selected_example = st.selectbox("Exemples rapides", list(examples.keys()))
    
    default_val = examples[selected_example] if examples[selected_example] else 1000003
    number_to_test = st.number_input("Entier à tester", min_value=1, value=default_val)
    
    st.markdown("---")
    st.markdown("### Méthodes d'analyse")
    selected_tests = []
    for test_type in TestType:
        if st.checkbox(test_type.value, value=True, key=f"fixed_{test_type.name}"):
            selected_tests.append(test_type)
            
    iterations = st.slider("Itérations Miller-Rabin/Solovay", 1, 100, 10)
    
    run_btn = st.button("LANCER L'ANALYSE", type="primary")
    
    if st.button("RÉINITIALISER LES DONNÉES"):
        st.session_state.history = []
        st.rerun()

# Onglets principaux
tab_analysis, tab_compare, tab_analytics, tab_education = st.tabs([
    "Analyse", "Mode Comparatif", "Statistiques", "Éducation"
])

def render_test_result(r, n, iterations):
    t_comp, e_prob = get_complexity_info(r.test_type.value if hasattr(r.test_type, 'value') else r.test_type, n, iterations)
    
    r_val = r.result.value if hasattr(r.result, 'value') else r.result
    badge_cls = "prob-prime-badge"
    
    # Traduction des résultats pour l'affichage
    display_result = r_val
    if r_val == Result.PRIME.value: 
        badge_cls = "prime-badge"
        display_result = "PREMIER"
    elif r_val == Result.COMPOSITE.value: 
        badge_cls = "composite-badge"
        display_result = "COMPOSÉ"
    elif r_val == Result.PROBABLY_PRIME.value:
        badge_cls = "prob-prime-badge"
        display_result = "PROB. PREMIER"
    elif r_val == Result.ERROR.value: 
        badge_cls = "composite-badge" 
        display_result = "ERREUR"
    
    st.markdown(f"""
    <div class="glass-card">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
            <span style="font-weight: 600; font-size: 1.1rem;">{r.test_type.value}</span>
            <span class="result-badge {badge_cls}">{display_result}</span>
        </div>
        <div style="font-size: 0.9rem; color: var(--text-secondary); line-height: 1.6;">
            Latence : <b>{r.execution_time*1000:.3f}ms</b><br/>
            Complexité : <b>{t_comp}</b><br/>
            Observation : <i>{r.message}</i>
        </div>
    </div>
    """, unsafe_allow_html=True)

with tab_analysis:
    if run_btn:
        start_total = time.perf_counter()
        run_results = []
        
        with st.status("Calcul en cours...", expanded=False) as status:
            for i, test_type in enumerate(selected_tests):
                is_mersenne = st.session_state.manager._is_mersenne_exponent(number_to_test) is not None
                if (test_type.value if hasattr(test_type, 'value') else test_type) == TestType.LUCAS_LEHMER.value and not is_mersenne:
                    continue
                    
                test_result = st.session_state.manager.run_test(test_type, number_to_test, iterations)
                run_results.append(test_result)
                
                st.session_state.history.append({
                    'Horodatage': time.strftime("%H:%M:%S"),
                    'Valeur': str(number_to_test),
                    'Bits': number_to_test.bit_length(),
                    'Méthode': test_type.value,
                    'Résultat': test_result.result.value,
                    'Durée (ms)': test_result.execution_time * 1000
                })
            
            total_time = time.perf_counter() - start_total
            status.update(label=f"Analyse de {number_to_test} terminée en {total_time:.3f}s", state="complete")

        # En-tête du Verdict
        valid_results = [r for r in run_results if (r.result.value if hasattr(r.result, 'value') else r.result) != Result.ERROR.value]
        
        if not valid_results:
            v_title = "ERREUR D'ANALYSE"
            v_color = "#FFB84D" # Orange d'avertissement
            v_desc = "Impossible d'obtenir des résultats valides avec les tests sélectionnés."
        else:
            is_prime = all((r.result.value if hasattr(r.result, 'value') else r.result) in [Result.PRIME.value, Result.PROBABLY_PRIME.value] for r in valid_results)
            v_title = "EST PREMIER" if is_prime else "EST COMPOSÉ"
            v_color = "#26C485" if is_prime else "#FF5757"
            v_desc = f"Testé : {number_to_test} ({number_to_test.bit_length()} bits)"
        
        st.markdown(f"""
        <div style="background: {v_color}22; border-left: 5px solid {v_color}; padding: 20px; border-radius: 8px; margin-bottom: 2rem;">
            <h2 style="margin:0; background:none; -webkit-text-fill-color:{v_color};">{v_title}</h2>
            <p style="margin:0; opacity:0.8;">{v_desc}</p>
        </div>
        """, unsafe_allow_html=True)

        m1, m2, m3 = st.columns(3)
        with m1: st.metric("Longueur binaire", number_to_test.bit_length())
        with m2: st.metric("Chiffres décimaux", len(str(number_to_test)))
        with m3: st.metric("Latence totale", f"{total_time*1000:.2f}ms")

        st.subheader("Détails par méthode")
        cols = st.columns(2)
        for idx, r in enumerate(run_results):
            with cols[idx % 2]:
                render_test_result(r, number_to_test, iterations)
    else:
        st.info("Configurez les paramètres dans la barre latérale et cliquez sur LANCER L'ANALYSE.")

with tab_compare:
    st.subheader("Comparer des nombres")
    st.markdown("Analysez deux nombres côte à côte pour comparer les performances et les résultats.")
    
    col_a, col_b = st.columns(2)
    with col_a:
        num_a = st.number_input("Nombre A", value=1000003, key="num_a")
    with col_b:
        num_b = st.number_input("Nombre B", value=1000037, key="num_b")
        
    if st.button("LANCER LA COMPARAISON"):
        cols = st.columns(2)
        for i, (n, col) in enumerate(zip([num_a, num_b], cols)):
            with col:
                st.markdown(f"### Analyse : {n}")
                for t in selected_tests:
                    is_mersenne = st.session_state.manager._is_mersenne_exponent(n) is not None
                    if (t.value if hasattr(t, 'value') else t) == TestType.LUCAS_LEHMER.value and not is_mersenne: continue
                    res = st.session_state.manager.run_test(t, n, iterations)
                    render_test_result(res, n, iterations)

with tab_analytics:
    if st.session_state.history:
        st.subheader("Performances du moteur")
        df = pd.DataFrame(st.session_state.history)
        
        c1, c2 = st.columns([2, 1])
        with c1:
            fig = px.scatter(df, x="Bits", y="Durée (ms)", color="Méthode", 
                            title="Latence vs Échelle (Bits)", 
                            template="plotly_dark",
                            color_discrete_sequence=px.colors.qualitative.Safe)
            st.plotly_chart(fig, width='stretch')
        
        with c2:
            st.markdown("### Statistiques moyennes")
            st.write(df.groupby("Méthode")["Durée (ms)"].mean().rename("Latence moy. (ms)"))
            
        st.markdown("### Historique d'analyse")
        st.dataframe(df, width='stretch')
    else:
        st.info("Lancez des tests pour voir les analyses de performance.")

with tab_education:
    st.subheader("Méthodologies de Test")
    
    methods_data = [
        {"Méthode": "Trial Division", "Type": "Déterministe", "Avantages": "Correction à 100%", "Inconvénients": "Très lent pour les grands n"},
        {"Méthode": "Miller-Rabin", "Type": "Probabiliste", "Avantages": "Extrêmement rapide", "Inconvénients": "Probabilité d'erreur infime"},
        {"Méthode": "Fermat", "Type": "Probabiliste", "Avantages": "Rapide", "Inconvénients": "Échoue sur les nombres de Carmichael"},
        {"Méthode": "AKS", "Type": "Déterministe", "Avantages": "Polynomial prouvé", "Inconvénients": "Facteurs constants très élevés"},
        {"Méthode": "Lucas-Lehmer", "Type": "Spécialisé", "Avantages": "Ultra rapide", "Inconvénients": "Uniquement pour les exposants de Mersenne"}
    ]
    st.table(methods_data)
    
    st.markdown("""
    ### Pourquoi plusieurs tests ?
    Aucun test unique n'est parfait pour tous les scénarios. 
    1. **Tests probabilistes** (Miller-Rabin) : Utilisés pour les très grands nombres (cryptographie).
    2. **Tests déterministes** : Utilisés lorsqu'une certitude mathématique absolue est requise.
    3. **Tests spécialisés** (Lucas-Lehmer) : Permettent de trouver les plus grands nombres premiers connus.
    """)
