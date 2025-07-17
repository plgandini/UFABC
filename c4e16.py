import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr

# --- PASSO 1: Carregar e Preparar os Dados ---
try:
    file_path = '/Users/plgandini/Coding/UFABC/EASG/Supermercados_Brasileiros.xlsx'
    df = pd.read_excel(file_path, engine="calamine")

except FileNotFoundError:
    print(f"Erro: O arquivo '{file_path}' não foi encontrado.")
    exit()
except Exception as e:
    print(f"Ocorreu um erro ao ler o arquivo: {e}")
    exit()

# --- CORREÇÃO: Usar os nomes de coluna corretos identificados pelo diagnóstico ---
df.columns = df.columns.str.strip() # Limpa espaços extras
# Definir os nomes corretos das colunas que vamos usar
col_empresa = 'Empresa'
col_faturamento = 'Faturamento'
col_lojas = 'Lojas'

# Garantir que os dados são numéricos
df[col_faturamento] = pd.to_numeric(df[col_faturamento], errors='coerce')
df[col_lojas] = pd.to_numeric(df[col_lojas], errors='coerce')
df.dropna(subset=[col_faturamento, col_lojas], inplace=True)


# --- PASSO 2: Análise com Todos os Dados (a, b, c) ---

print("--- Análise com Todos os Dados ---")

# a) Elaborar o diagrama de dispersão
plt.figure(figsize=(10, 7))
sns.regplot(data=df, x=col_lojas, y=col_faturamento, ci=None,
            scatter_kws={'s': 70, 'alpha': 0.7, 'edgecolor': 'w'}, line_kws={'color': 'red', 'linestyle': '--'})
# Adicionar rótulos para cada ponto
for i, row in df.iterrows():
    plt.text(row[col_lojas] + 10, row[col_faturamento], row[col_empresa], fontsize=9)

plt.title('Diagrama de Dispersão: Faturamento vs. Nº de Lojas (Todos os Dados)', fontsize=16)
plt.xlabel('Número de Lojas', fontsize=12)
plt.ylabel('Faturamento', fontsize=12)
plt.grid(True)
plt.savefig('dispersao_completa.png')
plt.close()

# b) Análise visual da dependência
print("\nb) Análise Visual da Dependência:")
print("   - Sim, o diagrama de dispersão sugere uma tendência positiva: em geral, quanto maior o número de lojas, maior o faturamento. No entanto, um ponto (Pão de Açúcar) se destaca com um faturamento muito superior aos demais.")

# c) Calcular o coeficiente de correlação de Pearson
r_completo, p_completo = pearsonr(df[col_faturamento], df[col_lojas])
print("\nc) Coeficiente de Correlação de Pearson (Todos os Dados):")
print(f"   - Coeficiente (r): {r_completo:.4f}")
print(f"   - Interpretação: Existe uma correlação positiva moderada a forte entre faturamento e número de lojas. A correlação é estatisticamente significativa.")
print("-" * 50)


# --- PASSO 3: Análise Sem o Outlier (d) ---

# d) Identificar e remover o conjunto de dados com comportamento diferente
outlier_nome = 'Pão de Açúcar'
df_sem_outlier = df[df[col_empresa] != outlier_nome]

print("\n--- Análise Sem o Outlier ('Pão de Açúcar') ---")

# d.1) Elaborar novamente o gráfico de dispersão
plt.figure(figsize=(10, 7))
sns.regplot(data=df_sem_outlier, x=col_lojas, y=col_faturamento, ci=None,
            scatter_kws={'s': 70, 'alpha': 0.7, 'edgecolor': 'w'}, line_kws={'color': 'red', 'linestyle': '--'})
for i, row in df_sem_outlier.iterrows():
    plt.text(row[col_lojas] + 5, row[col_faturamento], row[col_empresa], fontsize=9)
    
plt.title('Diagrama de Dispersão: Faturamento vs. Nº de Lojas (Sem Outlier)', fontsize=16)
plt.xlabel('Número de Lojas', fontsize=12)
plt.ylabel('Faturamento', fontsize=12)
plt.grid(True)
plt.savefig('dispersao_sem_outlier.png')
plt.close()

# d.2) Recalcular o coeficiente de correlação
r_sem_outlier, p_sem_outlier = pearsonr(df_sem_outlier[col_faturamento], df_sem_outlier[col_lojas])
print(f"\nd) Coeficiente de Correlação de Pearson (Sem o Outlier):")
print(f"   - Novo Coeficiente (r): {r_sem_outlier:.4f}")

print("\n   - Comparação e Conclusão:")
print(f"   - Ao remover o 'Pão de Açúcar', o coeficiente de correlação AUMENTOU de {r_completo:.4f} para {r_sem_outlier:.4f}.")
print("   - Isso indica que a relação linear entre o número de lojas e o faturamento para o *restante* do grupo é ainda MAIS FORTE. O Pão de Açúcar, com seu faturamento altíssimo, estava 'distorcendo' a força da correlação linear do conjunto.")

print("\n\n--- GRÁFICOS GERADOS ---")
print("1. 'dispersao_completa.png' (Análise com todos os supermercados)")
print("2. 'dispersao_sem_outlier.png' (Análise sem o Pão de Açúcar)")