import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency

# Carregar os dados
df = pd.read_csv("dados_financeiros.csv")

print("\nAnálise de Associação entre Variáveis Qualitativas\n")

# Escolher duas variáveis qualitativas para análise
# Agora usaremos 'porte_empresa' como uma das variáveis qualitativas
variavel_qualitativa_1 = 'porte_empresa'
variavel_qualitativa_2 = 'state'

# Remover linhas com valores ausentes nas variáveis selecionadas
df_filtered = df[[variavel_qualitativa_1, variavel_qualitativa_2]].dropna()

print(f"Variáveis selecionadas: {variavel_qualitativa_1} e {variavel_qualitativa_2}\n")

# 1. Tabelas de Distribuição de Frequência
print("1. Tabelas de Distribuição de Frequência")
print("----------------------------------------")

# Tabela de contingência (frequências absolutas)
tabela_contingencia = pd.crosstab(df_filtered[variavel_qualitativa_1], df_filtered[variavel_qualitativa_2])
print("a) Tabela de Contingência (Frequências Absolutas):\n")
print(tabela_contingencia)
print("\n")

# Tabela de frequências relativas por linha (percentual de linha)
tabela_linha = tabela_contingencia.apply(lambda r: r/r.sum(), axis=1)
print("b) Tabela de Frequências Relativas por Linha (Percentual de Linha):\n")
print(tabela_linha.round(4) * 100)
print("\n")

# Tabela de frequências relativas por coluna (percentual de coluna)
tabela_coluna = tabela_contingencia.apply(lambda c: c/c.sum(), axis=0)
print("c) Tabela de Frequências Relativas por Coluna (Percentual de Coluna):\n")
print(tabela_coluna.round(4) * 100)
print("\n")

# Tabela de frequências relativas total (percentual total)
tabela_total = tabela_contingencia / tabela_contingencia.sum().sum()
print("d) Tabela de Frequências Relativas Total (Percentual Total):\n")
print(tabela_total.round(4) * 100)
print("\n")

print("Interpretação das Tabelas de Frequência:\n")
print("   Analisando as tabelas de frequência relativa (por linha, coluna e total), podemos observar a distribuição das empresas por porte e estado. Se a distribuição percentual de uma variável (ex: porte_empresa) muda significativamente entre as categorias da outra variável (ex: estado), isso sugere uma possível associação. Caso contrário, se a distribuição for similar, a associação é menos provável.\n")

# 2. Teste Qui-Quadrado
print("2. Teste Qui-Quadrado")
print("---------------------")

# Cálculo da estatística Qui-Quadrado e valores esperados
chi2, p_valor, dof, expected = chi2_contingency(tabela_contingencia)

print(f"a) Estatística Qui-Quadrado: {chi2:.4f}")
print(f"b) Valor-p: {p_valor:.4f}")
print(f"c) Graus de Liberdade (dof): {dof}")
print("d) Frequências Esperadas:\n")
print(pd.DataFrame(expected, index=tabela_contingencia.index, columns=tabela_contingencia.columns).round(2))
print("\n")

print("Cálculo Detalhado do Qui-Quadrado:\n")
print("Fórmula: χ² = Σ[(O_ij - E_ij)² / E_ij]\n")
print("Onde O_ij = frequência observada e E_ij = frequência esperada\n")

chi2_calculo_detalhado = 0
for i in range(tabela_contingencia.shape[0]):
    for j in range(tabela_contingencia.shape[1]):
        observado = tabela_contingencia.iloc[i, j]
        esperado = expected[i, j]
        if esperado > 0: # Evitar divisão por zero
            termo = ((observado - esperado)**2) / esperado
            chi2_calculo_detalhado += termo
            print(f"   Termo para O_({tabela_contingencia.index[i]}, {tabela_contingencia.columns[j]}): ({observado} - {esperado:.2f})² / {esperado:.2f} = {termo:.4f}")
        else:
            print(f"   Termo para O_({tabela_contingencia.index[i]}, {tabela_contingencia.columns[j]}): Frequência esperada é zero, termo não calculado.")

print(f"\n   Soma dos termos (Qui-Quadrado Calculado): {chi2_calculo_detalhado:.4f}\n")

print("Interpretação do Teste Qui-Quadrado:\n")
print("   O teste Qui-Quadrado avalia se existe uma associação estatisticamente significativa entre as duas variáveis qualitativas. A hipótese nula (H0) é que não há associação (as variáveis são independentes), e a hipótese alternativa (H1) é que existe associação (as variáveis são dependentes).\n")
print(f"   Com base no valor-p ({p_valor:.4f}):\n")
if p_valor < 0.05:
    print("   - Se o valor-p for menor que o nível de significância (geralmente 0.05), rejeitamos a hipótese nula. Isso indica que há evidências estatísticas de uma associação significativa entre as variáveis.\n")
else:
    print("   - Se o valor-p for maior que o nível de significância (geralmente 0.05), não rejeitamos a hipótese nula. Isso indica que não há evidências estatísticas suficientes para afirmar uma associação significativa entre as variáveis.\n")

print("Análise de associação entre variáveis qualitativas concluída.")