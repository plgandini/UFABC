import pandas as pd
import numpy as np

# Carregar os dados
df = pd.read_csv("dados_financeiros.csv")

print("\nAnálise de Regressão Linear Múltipla\n")

# Definir variáveis
Y_var = 'preco_atual'
X_vars = ['market_cap', 'volatilidade']

# Remover linhas com valores ausentes nas variáveis selecionadas
df_filtered = df[[Y_var] + X_vars].dropna()

Y = df_filtered[Y_var].values
X = df_filtered[X_vars].values

n = len(Y)
k = X.shape[1] # Número de variáveis explicativas

print(f"Variável Dependente (Y): {Y_var}")
print(f"Variáveis Explicativas (X): {X_vars}\n")
print(f"Número de observações (n): {n}\n")

# Adicionar uma coluna de 1s para o intercepto ao X
X_com_intercepto = np.c_[np.ones(n), X]

# 1. Cálculo dos Coeficientes (Beta)
print("1. Cálculo dos Coeficientes (Beta)\n")
print("Fórmula: β = (XᵀX)⁻¹XᵀY\n")

# X transposto
X_transposto = X_com_intercepto.T
print("   Xᵀ (Transposta de X com Intercepto):\n", X_transposto)

# X transposto vezes X
X_transposto_X = np.dot(X_transposto, X_com_intercepto)
print("\n   XᵀX:\n", X_transposto_X)

# Inversa de (X transposto vezes X)
X_transposto_X_inv = np.linalg.inv(X_transposto_X)
print("\n   (XᵀX)⁻¹:\n", X_transposto_X_inv)

# X transposto vezes Y
X_transposto_Y = np.dot(X_transposto, Y)
print("\n   XᵀY:\n", X_transposto_Y)

# Coeficientes Beta
beta_hat = np.dot(X_transposto_X_inv, X_transposto_Y)

print("\n   Coeficientes (β̂):\n")
print(f"   Intercepto (β₀): {beta_hat[0]:.4f}")
for i in range(k):
    print(f"   Coeficiente para {X_vars[i]} (β{i+1}): {beta_hat[i+1]:.4f}")
print("\n")

# 2. Cálculo dos Valores Preditos (Ŷ) e Resíduos (e)
print("2. Cálculo dos Valores Preditos (Ŷ) e Resíduos (e)\n")

Y_pred = np.dot(X_com_intercepto, beta_hat)
residuos = Y - Y_pred

print("   Primeiros 5 Valores Observados (Y):", Y[:5].round(2))
print("   Primeiros 5 Valores Preditos (Ŷ):", Y_pred[:5].round(2))
print("   Primeiros 5 Resíduos (e):", residuos[:5].round(2))
print("\n")

# 3. Soma dos Quadrados
print("3. Soma dos Quadrados\n")

# Soma Total dos Quadrados (SST)
SST = np.sum((Y - np.mean(Y))**2)
print(f"   Soma Total dos Quadrados (SST): {SST:.2f}")

# Soma dos Quadrados da Regressão (SSR)
SSR = np.sum((Y_pred - np.mean(Y))**2)
print(f"   Soma dos Quadrados da Regressão (SSR): {SSR:.2f}")

# Soma dos Quadrados dos Resíduos (SSE)
SSE = np.sum(residuos**2)
print(f"   Soma dos Quadrados dos Resíduos (SSE): {SSE:.2f}\n")

# 4. R-quadrado (Coeficiente de Determinação)
print("4. R-quadrado (Coeficiente de Determinação)\n")

R_quadrado = SSR / SST
print(f"   R-quadrado = SSR / SST = {SSR:.2f} / {SST:.2f} = {R_quadrado:.4f}\n")

# 5. Erro Padrão dos Coeficientes e Testes t
print("5. Erro Padrão dos Coeficientes e Testes t\n")

# Variância do Erro (MSE)
MSE = SSE / (n - k - 1)
print(f"   Variância do Erro (MSE) = SSE / (n - k - 1) = {SSE:.2f} / ({n} - {k} - 1) = {MSE:.2f}\n")

# Matriz de Covariância dos Coeficientes
matriz_cov_beta = MSE * X_transposto_X_inv
print("   Matriz de Covariância dos Coeficientes:\n", matriz_cov_beta)

# Erro Padrão dos Coeficientes (diagonal da matriz de covariância)
erro_padrao_beta = np.sqrt(np.diag(matriz_cov_beta))

print("\n   Erros Padrão dos Coeficientes:\n")
print(f"   EP(β₀): {erro_padrao_beta[0]:.4f}")
for i in range(k):
    print(f"   EP(β{i+1}): {erro_padrao_beta[i+1]:.4f}")
print("\n")

# Valores t
valores_t = beta_hat / erro_padrao_beta

print("   Valores t:\n")
print(f"   t(β₀): {valores_t[0]:.4f}")
for i in range(k):
    print(f"   t(β{i+1}): {valores_t[i+1]:.4f}")
print("\n")

# P-valores (usando distribuição t, para simplificar, usaremos 2 * (1 - cdf(abs(t))))
# Para um cálculo mais preciso de p-valores, precisaríamos da função cdf da distribuição t
# do scipy.stats. Para este exercício, vamos apenas indicar a interpretação.

print("   P-valores (Interpretação):\n")
print("   Para obter os p-valores exatos, seria necessário usar a função de distribuição t (e.g., from scipy.stats import t). No entanto, a interpretação é baseada na comparação do valor t com valores críticos da distribuição t com (n - k - 1) graus de liberdade. Se |t| > t_critico, o coeficiente é estatisticamente significativo.\n")

# 6. Teste F (Significância do Modelo)
print("6. Teste F (Significância do Modelo)\n")

# Média dos Quadrados da Regressão (MSR)
MSR = SSR / k
print(f"   Média dos Quadrados da Regressão (MSR) = SSR / k = {SSR:.2f} / {k} = {MSR:.2f}")

# Estatística F
F_estatistica = MSR / MSE
print(f"   Estatística F = MSR / MSE = {MSR:.2f} / {MSE:.2f} = {F_estatistica:.4f}\n")

print("   Interpretação do Teste F:\n")
print("   O teste F avalia a significância global do modelo de regressão. A hipótese nula (H0) é que todos os coeficientes das variáveis explicativas são zero (o modelo não é significativo). A hipótese alternativa (H1) é que pelo menos um coeficiente é diferente de zero (o modelo é significativo). Para obter o p-valor exato, seria necessário usar a função de distribuição F (e.g., from scipy.stats import f). Se o p-valor for menor que o nível de significância (e.g., 0.05), rejeitamos H0, indicando que o modelo é estatisticamente significativo.\n")

print("Análise de regressão linear múltipla concluída.")