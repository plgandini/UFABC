import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
import stemgraphic

# --- PASSO 1: Carregar os Dados do Arquivo ---
# Garante que o arquivo seja encontrado e lido corretamente.
try:
    file_path = '/Users/plgandini/Coding/UFABC/EASG/Desidratação.xls'
    df = pd.read_excel(file_path)

except FileNotFoundError:
    print(f"Erro: O arquivo '{file_path}' não foi encontrado. Verifique o nome e o local do arquivo.")
    exit()
except Exception as e:
    print(f"Ocorreu um erro ao ler o arquivo: {e}")
    exit()

# Extrai os dados da primeira coluna e converte para formato numérico
column_name = df.columns[0]
tempos = pd.to_numeric(df[column_name], errors='coerce').dropna()

if tempos.empty:
    print("Nenhum dado numérico foi encontrado na coluna principal. Verifique o formato do arquivo.")
    exit()


# --- PASSO 2: Realizar todos os Cálculos Estatísticos ---
# (Esta seção permanece inalterada)

# a) Medidas de Posição
media = tempos.mean()
mediana = tempos.median()
moda_series = tempos.mode()
moda = moda_series.tolist() if not moda_series.empty and len(moda_series) < len(tempos) else "Nenhuma moda clara"

# b) Quartis e verificação de Outliers
q1 = tempos.quantile(0.25)
q3 = tempos.quantile(0.75)
iqr = q3 - q1
limite_inferior = q1 - 1.5 * iqr
limite_superior = q3 + 1.5 * iqr
outliers = tempos[(tempos < limite_inferior) | (tempos > limite_superior)]

# c) Percentis de ordem 10 e 90
p10 = tempos.quantile(0.10)
p90 = tempos.quantile(0.90)

# d) Decis de ordem 3 e 6
d3 = tempos.quantile(0.30)
d6 = tempos.quantile(0.60)

# e) Medidas de Dispersão
amplitude = tempos.max() - tempos.min()
desvio_medio_abs = (tempos - media).abs().mean()
variancia = tempos.var(ddof=1)
desvio_padrao = tempos.std(ddof=1)
erro_padrao = tempos.sem(ddof=1)
coef_variacao = (desvio_padrao / media) * 100 if media != 0 else 0

# f) Verificação da Simetria (Assimetria)
assimetria = tempos.skew()
if assimetria > 0.5:
    tipo_assimetria = "Assimétrica Positiva (cauda à direita)"
elif assimetria < -0.5:
    tipo_assimetria = "Assimétrica Negativa (cauda à esquerda)"
else:
    tipo_assimetria = "Aproximadamente Simétrica"

# g) Cálculo da Curtose (Achatamento)
curtose = tempos.kurtosis() # Curtose em excesso (padrão de Fisher)
if curtose > 0.5:
    tipo_curtose = "Leptocúrtica (mais pontuda, caudas pesadas)"
elif curtose < -0.5:
    tipo_curtose = "Platicúrtica (mais achatada, caudas leves)"
else:
    tipo_curtose = "Mesocúrtica (similar à distribuição Normal)"

# --- PASSO 3: Exibir Resultados no Console com 4 CASAS DECIMAIS ---

print("--- ANÁLISE ESTATÍSTICA DO TEMPO DE DESIDRATAÇÃO ---")
print("\na) Medidas de Posição:")
print(f"   - Média Aritmética: {media:.4f} s")
print(f"   - Mediana: {mediana:.4f} s")
print(f"   - Moda: {moda}")

print("\nb) Quartis e Outliers:")
print(f"   - Primeiro Quartil (Q1): {q1:.4f} s")
print(f"   - Terceiro Quartil (Q3): {q3:.4f} s")
print(f"   - Intervalo Interquartil (IQR): {iqr:.4f} s")
print(f"   - Limites para Outliers: Inferior < {limite_inferior:.4f} s | Superior > {limite_superior:.4f} s")
if not outliers.empty:
    print(f"   - Indícios de Outliers Encontrados: {sorted(outliers.round(4).tolist())}")
else:
    print("   - Não há indícios de outliers.")

print("\nc) Percentis:")
print(f"   - Percentil 10 (P10): {p10:.4f} s")
print(f"   - Percentil 90 (P90): {p90:.4f} s")

print("\nd) Decis:")
print(f"   - Decil 3 (D3): {d3:.4f} s")
print(f"   - Decil 6 (D6): {d6:.4f} s")

print("\ne) Medidas de Dispersão:")
print(f"   - Amplitude Total: {amplitude:.4f} s")
print(f"   - Desvio Médio Absoluto: {desvio_medio_abs:.4f} s")
print(f"   - Variância Amostral: {variancia:.4f} s²")
print(f"   - Desvio-Padrão Amostral: {desvio_padrao:.4f} s")
print(f"   - Erro-Padrão da Média: {erro_padrao:.4f} s")
print(f"   - Coeficiente de Variação: {coef_variacao:.4f}%")

print("\nf) Análise de Simetria:")
print(f"   - Coeficiente de Assimetria (Skewness): {assimetria:.4f}")
print(f"   - Classificação: {tipo_assimetria}")

print("\ng) Análise de Curtose (Achatamento):")
print(f"   - Coeficiente de Curtose (Excesso): {curtose:.4f}")
print(f"   - Classificação: {tipo_curtose}")

# --- PASSO 4: Gerar e Salvar os Gráficos ---
# (Esta seção permanece inalterada)

sns.set_style("whitegrid")
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle('Análise Gráfica dos Tempos de Desidratação', fontsize=16)

# Histograma
sns.histplot(tempos, kde=True, color='royalblue', ax=axes[0])
axes[0].set_title('Histograma')
axes[0].set_xlabel('Tempo (s)')
axes[0].set_ylabel('Frequência')

# Boxplot
sns.boxplot(y=tempos, color='mediumseagreen', ax=axes[1])
axes[1].set_title('Boxplot')
axes[1].set_ylabel('Tempo (s)')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig('analise_grafica_real.png')

# Gráfico Ramo-e-Folhas (salvo em arquivo separado)
try:
    fig_ramo, ax_ramo = stemgraphic.stem_graphic(tempos, scale=10)
    plt.savefig('ramo_e_folhas_real.png')
    print("\n--- GRÁFICOS GERADOS ---")
    print("1. 'analise_grafica_real.png' (Histograma e Boxplot)")
    print("2. 'ramo_e_folhas_real.png' (Ramo-e-Folhas)")
except Exception as e:
    print(f"\nNão foi possível gerar o gráfico Ramo-e-Folhas: {e}")