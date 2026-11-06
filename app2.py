import os
import joblib
import pandas as pd
import sklearn as sk
import streamlit as st

ARQUIVO_MODELO = "modelo_desempenho.pkl"


@st.cache_resource  # Cache para carregar o modelo apenas uma vez
def carregar_modelo(caminho_modelo):
    """Carrega o modelo do arquivo .pkl"""
    if not os.path.exists(caminho_modelo):
        return None, None
    try:
        modelo = joblib.load(caminho_modelo)
        classes_modelo = modelo.classes_
        return modelo, classes_modelo
    except Exception as e:
        st.error(f"Erro ao carregar o modelo: {e}")
        return None, None


def main():
    """Função principal que executa o App Streamlit"""
    # --- Carregar o modelo ---
    modelo, classes_modelo = carregar_modelo(ARQUIVO_MODELO)

    # Título para o aplicativo
    st.title("Projeto 1 - Classificação K-Éssimo Vizinho")
    st.subheader("Estudo de caso")

    if modelo is None:
        st.error(f"Arquivo do modelo ('{ARQUIVO_MODELO}') não encontrado.")
        st.warning(
            "Execute o script 'python treinamento.py' no terminal para treinar e criar o modelo."
        )
        st.stop()  # Para a execução do app

    # Criando um cabeçalho para a barra lateral
    st.sidebar.header("Insira os dados do Aluno:")

    # Guardando os valores informados nas variáveis
    faltas = st.sidebar.slider(
        "Informe a quantidade de faltas: ", min_value=0, max_value=10, value=0
    )

    nota_1 = st.sidebar.number_input(
        "Nota da primeira prova (0-10)",
        min_value=0.0,
        max_value=10.0,
        value=5.0,
        step=0.5,
    )

    nota_2 = st.sidebar.number_input(
        "Nota da segunda prova (0-10)",
        min_value=0.0,
        max_value=10.0,
        value=5.0,
        step=0.5,
    )

    # Botão para realizar a previsão
    if st.sidebar.button("Prever Desempenho"):
        dados_entrada = pd.DataFrame(
            [[faltas, nota_1, nota_2]], columns=["faltas", "nota_1", "nota_2"]
        )

        st.write("### Dados de Entrada:")
        st.dataframe(dados_entrada)

        # Previsão do modelo
        probabilidades = modelo.predict_proba(dados_entrada)
        previsao = modelo.predict(dados_entrada)[0]

        st.write("### Resultado da Previsão:")
        st.success(f"Classe Prevista: **{previsao}**")

        # Exibindo as probabilidades por classe
        st.write("Probabilidades por classe:")
        df_prob = pd.DataFrame(probabilidades, columns=classes_modelo)
        st.dataframe(df_prob)

        st.header(f"Resultado da Previsão: {previsao}")

        # Mensagem de acordo com o resultado
        if str(previsao).lower() == "aprovado":
            st.success("Este aluno tem alta probabilidade de ser aprovado.")
        else:
            st.error("Este aluno está em zona de risco.")

        # Exibindo as métricas de confiança
        st.write("---")
        st.write("### Análise de confiança da IA")

        classes_list = list(classes_modelo)
        try:
            idx_aprovado = classes_list.index("aprovado")
            idx_reprovado = classes_list.index("reprovado")

            prob_aprovado = probabilidades[0][idx_aprovado]
            prob_reprovado = probabilidades[0][idx_reprovado]

            col1, col2 = st.columns(2)
            col1.metric("Confiança de 'aprovado'", f"{prob_aprovado * 100:.2f}%")
            col2.metric("Confiança de 'reprovado'", f"{prob_reprovado * 100:.2f}%")
        except ValueError:
            # Caso as classes no modelo tenham nomes diferentes (ex: maiúsculas ou 0/1)
            st.info("Classes detalhadas exatas:")
            for cls, prob in zip(classes_modelo, probabilidades[0]):
                st.write(f"- **{cls}**: {prob * 100:.2f}%")


if __name__ == "__main__":
    main()