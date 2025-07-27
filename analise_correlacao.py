import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Carregar os dados
df = pd.read_csv("dados_financeiros.csv")

print("\nAnálise de Correlação entre Variáveis Quantitativas\n")

# Escolher duas variáveis quantitativas para análise
variavel_quantitativa_1 = 'volatilidade'
variavel_quantitativa_2 = 'market_cap'

# Remover linhas com valores ausentes nas variáveis selecionadas
df_filtered = df[[variavel_quantitativa_1, variavel_quantitativa_2]].dropna()

dados_x = df_filtered[variavel_quantitativa_1].values
dados_y = df_filtered[variavel_quantitativa_2].values

print(f"Variáveis selecionadas: {variavel_quantitativa_1} e {variavel_quantitativa_2}\n")

# 1. Diagrama de Dispersão
print("1. Diagrama de Dispersão")
print("------------------------")

plt.figure(figsize=(10, 6))
sns.scatterplot(x=dados_x, y=dados_y, color='steelblue') # Usando steelblue para tom de azul
plt.title(f'Diagrama de Dispersão: {variavel_quantitativa_1} vs {variavel_quantitativa_2}')
plt.xlabel(variavel_quantitativa_1)
plt.ylabel(variavel_quantitativa_2)
plt.grid(True, linestyle='--', alpha=0.6)
plt.savefig('diagrama_dispersao.png')
print("   Diagrama de Dispersão salvo em 'diagrama_dispersao.png'\n")

print("Interpretação do Diagrama de Dispersão:\n")
print("   O diagrama de dispersão visualiza a relação entre as duas variáveis. Uma nuvem de pontos que segue uma tendência (crescente ou decrescente) sugere dependência, enquanto uma nuvem de pontos dispersa aleatoriamente sugere ausência de dependência linear.\n")

# 2. Coeficiente de Correlação de Pearson
print("2. Coeficiente de Correlação de Pearson")
print("---------------------------------------")

# Cálculo da média de X e Y
media_x = sum(dados_x) / len(dados_x)
media_y = sum(dados_y) / len(dados_y)

print(f"   Média de {variavel_quantitativa_1} (X̄): {media_x:.2f}")
print(f"   Média de {variavel_quantitativa_2} (Ȳ): {media_y:.2f}\n")

# Cálculo detalhado do numerador (soma dos produtos dos desvios)
print("Cálculo Detalhado do Numerador (Soma dos Produtos dos Desvios):\n")
numerador = 0
for i in range(len(dados_x)):
    desvio_x = dados_x[i] - media_x
    desvio_y = dados_y[i] - media_y
    produto_desvios = desvio_x * desvio_y
    numerador += produto_desvios
    print(f"   Observação {i+1}: ({dados_x[i]:.2f} - {media_x:.2f}) × ({dados_y[i]:.2f} - {media_y:.2f}) = {desvio_x:.2f} × {desvio_y:.2f} = {produto_desvios:.2f}")

print(f"\n   Soma dos produtos dos desvios (Numerador): {numerador:.2f}\n")

# Cálculo do denominador (produto dos desvios padrão)
print("Cálculo Detalhado do Denominador:\n")

# Desvio padrão de X
soma_quadrados_desvios_x = 0
for x in dados_x:
    soma_quadrados_desvios_x += (x - media_x)**2
variancia_x = soma_quadrados_desvios_x / (len(dados_x) - 1)
desvio_padrao_x = variancia_x**0.5

print(f"   Soma dos quadrados dos desvios de X: {soma_quadrados_desvios_x:.2f}")
print(f"   Variância de X: {variancia_x:.2f}")
print(f"   Desvio Padrão de X: {desvio_padrao_x:.2f}\n")

# Desvio padrão de Y
soma_quadrados_desvios_y = 0
for y in dados_y:
    soma_quadrados_desvios_y += (y - media_y)**2
variancia_y = soma_quadrados_desvios_y / (len(dados_y) - 1)
desvio_padrao_y = variancia_y**0.5

print(f"   Soma dos quadrados dos desvios de Y: {soma_quadrados_desvios_y:.2f}")
print(f"   Variância de Y: {variancia_y:.2f}")
print(f"   Desvio Padrão de Y: {desvio_padrao_y:.2f}\n")

denominador = (len(dados_x) - 1) * desvio_padrao_x * desvio_padrao_y

print(f"   Denominador: (n - 1) × Desvio Padrão de X × Desvio Padrão de Y")
print(f"   Denominador: ({len(dados_x)} - 1) × {desvio_padrao_x:.2f} × {desvio_padrao_y:.2f} = {denominador:.2f}\n")

# Evitar divisão por zero
if denominador == 0:
    coef_correlacao_pearson = 0 # Ou NaN, dependendo da convenção
else:
    coef_correlacao_pearson = numerador / denominador

print(f"   Coeficiente de Correlação de Pearson (r): {numerador:.2f} / {denominador:.2f} = {coef_correlacao_pearson:.4f}\n")

print("Conclusão em relação à dependência das variáveis:\n")
if abs(coef_correlacao_pearson) >= 0.7:
    print("   Há uma forte correlação linear entre as variáveis.")
elif abs(coef_correlacao_pearson) >= 0.3:
    print("   Há uma correlação linear moderada entre as variáveis.")
elif abs(coef_correlacao_pearson) > 0:
    print("   Há uma correlação linear fraca entre as variáveis.")
else:
    print("   Não há correlação linear entre as variáveis.")

if coef_correlacao_pearson > 0:
    print("   A correlação é positiva: quando uma variável aumenta, a outra tende a aumentar.")
elif coef_correlacao_pearson < 0:
    print("   A correlação é negativa: quando uma variável aumenta, a outra tende a diminuir.")
else:
    print("   Não há direção de correlação linear.")

print("\nAnálise de correlação entre variáveis quantitativas concluída.")