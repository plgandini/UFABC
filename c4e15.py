import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr

# --- PASSO 1: Carregar e Preparar os Dados ---
try:
    file_path = '/Users/plgandini/Coding/UFABC/EASG/Avaliação_Alunos.xlsx'
    df = pd.read_excel(file_path, engine="calamine")

except FileNotFoundError:
    print(f"Erro: O arquivo '{file_path}' não foi encontrado.")
    exit()
except Exception as e:
    print(f"Ocorreu um erro ao ler o arquivo: {e}")
    exit()

# --- CORREÇÃO: Renomear as colunas para um padrão consistente ---
# Com base no seu diagnóstico, os nomes originais são ['Nome', 'Po', 'EstatíStica', 'OperaçõEs', 'FinançAs']
# Vamos renomeá-los para os nomes que o resto do script espera.
try:
    df.columns = ['Nome', 'Pesquisa Operacional', 'Estatística', 'Gestão de Operações', 'Finanças']
except ValueError:
    print("Aviso: O número de colunas no arquivo não corresponde ao esperado (5). Verifique o arquivo.")
    # Se houver um número diferente de colunas, use os nomes que existem
    pass


# Garantir que todas as colunas de notas são numéricas
colunas_de_notas = ['Pesquisa Operacional', 'Estatística', 'Gestão de Operações', 'Finanças']
for col in colunas_de_notas:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

df.dropna(subset=colunas_de_notas, inplace=True)


# --- PASSO 2: Definir os pares de variáveis para análise ---
pares_para_analise = [
    ('Pesquisa Operacional', 'Estatística'),
    ('Gestão de Operações', 'Finanças'),
    ('Pesquisa Operacional', 'Gestão de Operações')
]

# --- PASSO 3: Iterar, Calcular e Plotar para cada par ---

print("--- Análise de Correlação de Pearson ---")

for i, (var_x, var_y) in enumerate(pares_para_analise):
    
    # Verificar se as colunas existem no DataFrame
    if var_x not in df.columns or var_y not in df.columns:
        print(f"\nAVISO: Uma ou ambas as colunas '{var_x}' e '{var_y}' não foram encontradas no arquivo após a renomeação.")
        continue

    # Extrair os dados
    x_data = df[var_x]
    y_data = df[var_y]
    
    # Calcular o Coeficiente de Correlação de Pearson
    r, p_valor = pearsonr(x_data, y_data)
    
    # Interpretação da força da correlação
    if abs(r) >= 0.7:
        forca = "Forte"
    elif abs(r) >= 0.4:
        forca = "Moderada"
    else:
        forca = "Fraca"
        
    # Interpretação da significância
    significancia = "estatisticamente significativa" if p_valor < 0.05 else "não é estatisticamente significativa"

    print(f"\nAnálise do Par {chr(97+i)}): '{var_x}' vs '{var_y}'")
    print(f"  - Coeficiente de Correlação de Pearson (r): {r:.4f}")
    print(f"  - P-valor: {p_valor:.4f}")
    print(f"  - Interpretação: Há uma correlação {forca}, {'positiva' if r > 0 else 'negativa'}, e que {significancia}.")
    
    # Construir o Diagrama de Dispersão
    plt.figure(figsize=(8, 6))
    sns.regplot(x=x_data, y=y_data, ci=None, scatter_kws={'alpha':0.6, 'color':'#005A9C'}, line_kws={'color':'#D43F4F'})
    
    plt.title(f'Diagrama de Dispersão: {var_x} vs {var_y}', fontsize=16)
    plt.xlabel(f'Nota de {var_x}', fontsize=12)
    plt.ylabel(f'Nota de {var_y}', fontsize=12)
    
    # Adicionar o valor da correlação no gráfico
    plt.text(0.05, 0.95, f'r = {r:.4f}\np-valor = {p_valor:.4f}', 
             transform=plt.gca().transAxes, fontsize=12,
             verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', fc='wheat', alpha=0.5))
             
    plt.grid(True)
    
    # Salvar o gráfico
    nome_arquivo = f"dispersao_{var_x.replace(' ', '')}_vs_{var_y.replace(' ', '')}.png"
    plt.savefig(nome_arquivo)
    plt.close() # Fecha a figura para não exibir todas de uma vez

print("\n--- Gráficos Gerados ---")
print("Os diagramas de dispersão para cada par foram salvos como arquivos .png.")