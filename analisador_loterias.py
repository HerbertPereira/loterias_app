import pandas as pd
import requests
from bs4 import BeautifulSoup
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import collections
import random
import os

# --- CONSTANTES GLOBAIS ---
PRIMOS_MEGA_SENA = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59]
PRIMOS_LOTOFACIL = [p for p in PRIMOS_MEGA_SENA if p <= 25]
MEGA_SENA_PRICES = {6: 5.00, 7: 35.00, 8: 140.00, 9: 420.00, 10: 1050.00, 11: 2310.00, 12: 4620.00, 13: 8580.00, 14: 15015.00, 15: 25025.00}
LOTOFACIL_PRICES = {15: 3.00, 16: 48.00, 17: 408.00, 18: 2448.00, 19: 11628.00, 20: 46512.00}

# --- FUNÇÕES DE WEB SCRAPING (COM TRATAMENTO DE ERROS) ---
@st.cache_data(ttl=3600)
def fetch_megasena_data():
    url = "https://www.megasena.com/resultados"
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
    try:
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        h2 = soup.find('h2', string='Resultados anteriores')
        if not h2:
            st.error("Não foi possível encontrar a seção 'Resultados anteriores' no site da Mega-Sena. O layout pode ter mudado.")
            return None
        target_table = h2.find_next_sibling('table')
        if not target_table:
            st.error("Não foi possível encontrar a tabela de resultados da Mega-Sena.")
            return None
        data = []
        rows = target_table.find_all('tr')
        for row in rows:
            if 'tbhead' in row.get('class', []) or 'table-banner' in row.get('class', []): continue
            cols = row.find_all('td')
            if len(cols) >= 2 and cols[0].find('a'):
                concurso_num = cols[0].find('a').text.replace('Concurso ', '').strip()
                dezenas = [int(li.text.strip()) for li in cols[1].find_all('li', class_='ball')]
                if concurso_num and len(dezenas) == 6: data.append([concurso_num] + dezenas)
        df = pd.DataFrame(data, columns=['Concurso'] + [f'Dezena{i}' for i in range(1, 7)])
        df['Concurso'] = pd.to_numeric(df['Concurso'], errors='coerce')
        return df.sort_values(by='Concurso', ascending=True).reset_index(drop=True)
    except requests.exceptions.RequestException as e:
        st.error(f"Erro de conexão ao buscar dados da Mega-Sena: {e}")
        return None
    except Exception as e:
        st.error(f"Ocorreu um erro inesperado ao processar os dados da Mega-Sena: {e}")
        return None

@st.cache_data(ttl=3600)
def fetch_lotofacil_data():
    url = "https://www.lotodicas.com.br/resultados-da-lotofacil"
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
    try:
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        data = []
        tables = soup.find_all('table', class_='table-bordered')
        if not tables:
            st.error("Nenhuma tabela de resultados encontrada no site da Lotofácil. O layout pode ter mudado.")
            return None
        for table in tables:
            rows = table.find_all('tr')
            if len(rows) < 2: continue
            header_text = rows[0].th.get_text(strip=True)
            if "Concurso" not in header_text: continue
            concurso_num = header_text.split(' ')[1]
            dezenas_tags = rows[1].find_all('td', class_='bg-primary')
            if len(dezenas_tags) == 15:
                dezenas = [int(tag.text.strip()) for tag in dezenas_tags]
                data.append([concurso_num] + dezenas)
        columns = ['Concurso'] + [f'Dezena{i}' for i in range(1, 16)]
        df = pd.DataFrame(data, columns=columns)
        df['Concurso'] = pd.to_numeric(df['Concurso'], errors='coerce')
        return df.sort_values(by='Concurso', ascending=True).reset_index(drop=True)
    except requests.exceptions.RequestException as e:
        st.error(f"Erro de conexão ao buscar dados da Lotofácil: {e}")
        return None
    except Exception as e:
        st.error(f"Ocorreu um erro inesperado ao processar os dados da Lotofácil: {e}")
        return None

# --- FUNÇÕES DE ANÁLISE E SUGESTÃO ---
@st.cache_data
def analyze_numbers(data, num_dezenas_sorteadas):
    if data is None or data.empty: return pd.Series(), [], []
    numbers = []
    for i in range(1, num_dezenas_sorteadas + 1):
        numbers.extend(data[f'Dezena{i}'].dropna().tolist())
    if not numbers: return pd.Series(), [], []
    freq = pd.Series(numbers).value_counts()
    most_common = freq.head(2).index.tolist()
    least_common = freq.tail(1).index.tolist()
    return freq, most_common, least_common

@st.cache_data
def plot_frequencies(frequencies, title, universe_max):
    if frequencies.empty: return
    full_range_freq = pd.Series(0, index=range(1, universe_max + 1))
    full_range_freq.update(frequencies)
    fig, ax = plt.subplots(figsize=(15, 7))
    ax.bar(full_range_freq.index, full_range_freq.values, color='skyblue')
    ax.set_title(title, fontsize=16)
    ax.set_xlabel('Número', fontsize=14); ax.set_ylabel('Frequência', fontsize=14)
    ax.set_xticks(np.arange(1, universe_max + 1, 1))
    ax.tick_params(axis='x', labelrotation=90, labelsize=8)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout(); st.pyplot(fig); plt.close(fig)

def sugestao_estrategica(most_common, least_common, primos_lista, universo_max, num_dezenas_total):
    jogo_sugerido = set()
    jogo_sugerido.update(most_common); jogo_sugerido.update(least_common)
    primos_disponiveis = list(set(primos_lista) - jogo_sugerido)
    random.shuffle(primos_disponiveis)
    num_primos_a_add = min(len(primos_disponiveis), 2)
    for i in range(num_primos_a_add): jogo_sugerido.add(primos_disponiveis[i])
    numeros_disponiveis = list(set(range(1, universo_max + 1)) - jogo_sugerido)
    random.shuffle(numeros_disponiveis)
    while len(jogo_sugerido) < num_dezenas_total and numeros_disponiveis: jogo_sugerido.add(numeros_disponiveis.pop(0))
    return sorted(list(jogo_sugerido))

# --- INTERFACE PRINCIPAL DO STREAMLIT ---
def main():
    st.set_page_config(page_title="Analisador de Loterias", layout="wide")
    st.sidebar.title("Menu de Opções")
    loteria_selecionada = st.sidebar.radio("Escolha a Loteria:", ("Mega-Sena", "Lotofácil"))
    
    if loteria_selecionada == "Mega-Sena":
        st.title("🔢 Analisador Estratégico - Mega-Sena")
        if st.button("📊 Buscar Resultados da Mega-Sena"):
            with st.spinner("Buscando dados..."):
                st.session_state['ms_data'] = fetch_megasena_data()
        if 'ms_data' in st.session_state and st.session_state['ms_data'] is not None:
            freq, most, least = analyze_numbers(st.session_state['ms_data'], 6)
            st.header("Análise de Frequência (Mega-Sena)")
            col1, col2 = st.columns(2); col1.subheader("🔥 2 Mais Sorteados"); col1.write(f"**Números:** {most}"); col2.subheader("❄️ 1 Menos Sorteado"); col2.write(f"**Número:** {least}")
            plot_frequencies(freq, "Frequência dos Números - Mega-Sena", 60)
            st.header("🎲 Gerador de Jogo (Mega-Sena)")
            num_dezenas = st.slider("Quantidade de dezenas:", 6, 15, 6, key='ms_slider')
            if st.button("Gerar Jogo Estratégico!", key='ms_gerar'):
                sugestao = sugestao_estrategica(most, least, PRIMOS_MEGA_SENA, 60, num_dezenas)
                st.subheader("✅ Seu Jogo Sugerido:"); formatted_numbers = " ".join([f"**{num:02d}**" for num in sugestao]); st.markdown(f"### {formatted_numbers}")
                preco = MEGA_SENA_PRICES.get(num_dezenas, "N/A"); st.markdown(f"**Valor estimado:** R$ {preco:.2f}")

    elif loteria_selecionada == "Lotofácil":
        st.title("🍀 Analisador Estratégico - Lotofácil")
        if st.button("📊 Buscar Resultados da Lotofácil"):
            with st.spinner("Buscando dados..."):
                st.session_state['lf_data'] = fetch_lotofacil_data()
        if 'lf_data' in st.session_state and st.session_state['lf_data'] is not None:
            freq, most, least = analyze_numbers(st.session_state['lf_data'], 15)
            st.header("Análise de Frequência (Lotofácil)")
            col1, col2 = st.columns(2); col1.subheader("🔥 2 Mais Sorteados"); col1.write(f"**Números:** {most}"); col2.subheader("❄️ 1 Menos Sorteado"); col2.write(f"**Número:** {least}")
            plot_frequencies(freq, "Frequência dos Números - Lotofácil", 25)
            st.header("🎲 Gerador de Jogo (Lotofácil)")
            num_dezenas = st.slider("Quantidade de dezenas:", 15, 20, 15, key='lf_slider')
            if st.button("Gerar Jogo Estratégico!", key='lf_gerar'):
                sugestao = sugestao_estrategica(most, least, PRIMOS_LOTOFACIL, 25, num_dezenas)
                st.subheader("✅ Seu Jogo Sugerido:"); formatted_numbers = " ".join([f"**{num:02d}**" for num in sugestao]); st.markdown(f"### {formatted_numbers}")
                preco = LOTOFACIL_PRICES.get(num_dezenas, "N/A"); st.markdown(f"**Valor estimado:** R$ {preco:.2f}")

if __name__ == "__main__":
    main()
