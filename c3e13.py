import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# --- PASSO 1: Carregar os Dados do Arquivo Real ---
try:
    file_path = '/Users/plgandini/Coding/UFABC/EASG/Serviços.xls'
    df_servicos = pd.read_excel(file_path)

except FileNotFoundError:
    print(f"Erro: O arquivo '{file_path}' não foi encontrado.")
    exit()
except Exception as e:
    print(f"Ocorreu um erro ao ler o arquivo: {e}")
    exit()

# Converte todas as colunas para numérico, tratando possíveis erros
for col in df_servicos.columns:
    df_servicos[col] = pd.to_numeric(df_servicos[col], errors='coerce')

# Remove linhas que possam ter ficado com valores nulos após a conversão
df_servicos.dropna(inplace=True)


# --- PASSO 2: Cálculos e Exibição dos Resultados ---

# Itera sobre cada coluna (serviço) do DataFrame para calcular as métricas
for servico in df_servicos.columns:
    dados_servico = df_servicos[servico]

    # a) Medidas de Posição
    media = dados_servico.mean()
    mediana = dados_servico.median()
    moda = dados_servico.mode().tolist()

    # b) Medidas de Dispersão
    variancia = dados_servico.var(ddof=1)
    desvio_padrao = dados_servico.std(ddof=1)
    erro_padrao = dados_servico.sem(ddof=1)

    # c) Quartis e Outliers
    q1 = dados_servico.quantile(0.25)
    q3 = dados_servico.quantile(0.75)
    iqr = q3 - q1
    limite_inferior = q1 - 1.5 * iqr
    limite_superior = q3 + 1.5 * iqr
    outliers = dados_servico[(dados_servico < limite_inferior) | (dados_servico > limite_superior)]

    # d) Assimetria e Curtose de Fisher
    assimetria_g1 = dados_servico.skew()
    curtose_g2 = dados_servico.kurtosis()

    if assimetria_g1 > 0.5:
        tipo_assimetria = "Assimétrica Positiva"
    elif assimetria_g1 < -0.5:
        tipo_assimetria = "Assimétrica Negativa"
    else:
        tipo_assimetria = "Aproximadamente Simétrica"

    if curtose_g2 > 0.5:
        tipo_curtose = "Leptocúrtica (mais 'pontuda' que a Normal)"
    elif curtose_g2 < -0.5:
        tipo_curtose = "Platicúrtica (mais 'achatada' que a Normal)"
    else:
        tipo_curtose = "Mesocúrtica (similar à Normal)"

    # Exibição dos resultados para o serviço atual
    print(f"\n--- ANÁLISE DO '{servico}' ---")
    print("\nMedidas de Posição:")
    print(f"  - Média: {media:.2f} min | Mediana: {mediana:.2f} min | Moda: {moda if len(moda) < 5 else 'Nenhuma moda clara'}")
    print("\nMedidas de Dispersão:")
    print(f"  - Variância: {variancia:.2f} min² | Desvio Padrão: {desvio_padrao:.2f} min | Erro Padrão: {erro_padrao:.2f} min")
    print("\nQuartis e Outliers:")
    print(f"  - Q1: {q1:.2f} min | Q3: {q3:.2f} min")
    if not outliers.empty:
        print(f"  - Indícios de Outliers: {outliers.round(2).tolist()}")
    else:
        print("  - Não há indícios de outliers.")
    print("\nSimetria e Curtose:")
    print(f"  - Assimetria (g1): {assimetria_g1:.2f} ({tipo_assimetria})")
    print(f"  - Curtose (g2): {curtose_g2:.2f} ({tipo_curtose})")
    print("-" * 50)

# --- PASSO 3: Geração dos Gráficos ---

# Gráfico de Barras para comparar as médias
plt.figure(figsize=(8, 6))
df_servicos.mean().sort_values().plot(kind='bar', color=['lightcoral', 'mediumseagreen', 'cornflowerblue'], edgecolor='black')
plt.title('Tempo Médio de Atendimento por Serviço', fontsize=16)
plt.ylabel('Tempo Médio (minutos)')
plt.xlabel('Tipo de Serviço')
plt.xticks(rotation=0)
plt.grid(axis='y', linestyle='--')
plt.savefig('grafico_barras_comparativo.png')
plt.close()

# Boxplot comparativo
plt.figure(figsize=(10, 7))
sns.boxplot(data=df_servicos)
plt.title('Boxplot Comparativo dos Tempos de Atendimento', fontsize=16)
plt.ylabel('Tempo (minutos)')
plt.xlabel('Serviços')
plt.savefig('boxplot_comparativo.png')
plt.close()

# Histogramas individuais
fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
fig.suptitle('Histograma de Frequência por Serviço', fontsize=18)
colors = ['lightcoral', 'mediumseagreen', 'cornflowerblue']
for i, servico in enumerate(df_servicos.columns):
    sns.histplot(df_servicos[servico], ax=axes[i], kde=True, bins=12, color=colors[i])
    axes[i].set_title(servico)
    axes[i].set_xlabel("Tempo (minutos)")
axes[0].set_ylabel("Frequência")
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig('histogramas_individuais.png')
plt.close()

print("\n\n--- GRÁFICOS GERADOS ---")
print("1. 'grafico_barras_comparativo.png' (Comparação das médias)")
print("2. 'boxplot_comparativo.png' (Comparação das distribuições)")
print("3. 'histogramas_individuais.png' (Distribuição de cada serviço)")