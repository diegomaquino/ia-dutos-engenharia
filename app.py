import streamlit as st
import pandas as pd
import pickle

st.set_page_config(page_title="Integridade de Dutos - IA", layout="centered")

st.title("🛢️ Sistema de Avaliação de Integridade de Dutos")
st.markdown("Suporte à Decisão em Manutenção Preditiva (Óleo e Gás)")

# 1. Carregando os modelos e transformadores treinados
@st.cache_resource
def carregar_arquivos():
    modelo = pickle.load(open('modelo_final.sav', 'rb'))
    scaler = pickle.load(open('scaler_final.sav', 'rb'))
    preprocessador = pickle.load(open('preprocessador_final.sav', 'rb'))
    return modelo, scaler, preprocessador

try:
    modelo, scaler, preprocessador = carregar_arquivos()
except Exception as e:
    st.error(f"Erro ao carregar arquivos .sav: {e}. Certifique-se de que estão na mesma pasta.")
    st.stop()

# 2. Formulário de entrada dos parâmetros operacionais e construtivos
st.subheader("📋 Parâmetros do Segmento Inspecionado")

col1, col2 = st.columns(2)

with col1:
    pipe_size = st.number_input("Diâmetro Nominal (Pipe_Size_mm)", min_value=50, max_value=1500, value=500, step=50)
    thickness = st.number_input("Espessura Original (Thickness_mm)", min_value=3.0, max_value=50.0, value=18.5, step=0.1)
    material = st.selectbox("Material Construtivo", ['Carbon Steel', 'Stainless Steel', 'PVC', 'HDPE', 'Fiberglass'])
    grade = st.selectbox("Norma/Grade de Fabricação", ['ASTM A333 Grade 6', 'ASTM A106 Grade B', 'API 5L X42', 'API 5L X52', 'API 5L X65'])
    pressure = st.number_input("Pressão Máxima (Max_Pressure_psi)", min_value=150, max_value=2500, value=200, step=50)

with col2:
    temp = st.number_input("Temperatura Operacional (°C)", min_value=-50.0, max_value=150.0, value=25.0, step=0.5)
    corrosion = st.number_input("Impacto Corrosivo (%)", min_value=0.0, max_value=20.0, value=2.0, step=0.1)
    thickness_loss = st.number_input("Perda de Espessura Medida (Thickness_Loss_mm)", min_value=0.0, max_value=10.0, value=0.4, step=0.05)
    material_loss = st.number_input("Perda de Material Calculada (%)", min_value=0.0, max_value=100.0, value=2.16, step=0.1)
    time_years = st.number_input("Tempo em Serviço (Time_Years)", min_value=1, max_value=25, value=2, step=1)

# 3. Execução da inferência ao clicar no botão
if st.button("Avaliar Condição Estrutural", type="primary"):
    entrada = pd.DataFrame([{
        'Pipe_Size_mm': pipe_size,
        'Thickness_mm': thickness,
        'Material': material,
        'Grade': grade,
        'Max_Pressure_psi': pressure,
        'Temperature_C': temp,
        'Corrosion_Impact_Percent': corrosion,
        'Thickness_Loss_mm': thickness_loss,
        'Material_Loss_Percent': material_loss,
        'Time_Years': time_years
    }])

    # Pré-processamento
    entrada_enc = preprocessador.transform(entrada)
    predicao = modelo.predict(entrada_enc)[0]
    probabilidades = modelo.predict_proba(entrada_enc)[0]

    st.divider()
    st.subheader("🔍 Diagnóstico do Modelo")

    # Mapeamento visual das recomendações
    recs = {
        'Normal': ('🟢 Normal', 'Continuar a operação. Linha íntegra dentro dos parâmetros normativos de projeto.', 'success'),
        'Moderate': ('🟡 Moderate', 'Aumentar o monitoramento e programar inspeção detalhada por ultrassom no próximo ciclo.', 'warning'),
        'Critical': ('🔴 Critical', 'Interromper para avaliação técnica e programar manutenção imediata (risco de ruptura).', 'error')
    }
    
    rotulo, recomendacao, status = recs.get(predicao, ('Indeterminado', 'Avaliar manualmente.', 'info'))

    if status == 'success':
        st.success(f"**Condição Prevista:** {rotulo}")
    elif status == 'warning':
        st.warning(f"**Condição Prevista:** {rotulo}")
    else:
        st.error(f"**Condição Prevista:** {rotulo}")

    st.info(f"**Recomendação de Engenharia:** {recomendacao}")

    # Exibição das probabilidades
    st.write("**Probabilidades Estimadas:**")
    df_prob = pd.DataFrame([probabilidades * 100], columns=modelo.classes_, index=['Certeza (%)']).round(2)
    st.dataframe(df_prob)
