import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Carregar os dados
df = pd.read_csv("/home/ubuntu/dados_financeiros.csv")

# Definir variáveis
Y_var = 'preco_atual'
X_vars = ['market_cap', 'volatilidade']

# Remover linhas com valores ausentes nas variáveis selecionadas
df_filtered = df[[Y_var] + X_vars].dropna()

Y = df_filtered[Y_var].values
X = df_filtered[X_vars].values

n = len(Y)
k = X.shape[1] # Número de variáveis explicativas

# Adicionar uma coluna de 1s para o intercepto ao X
X_com_intercepto = np.c_[np.ones(n), X]

# Calcular os coeficientes (Beta)
X_transposto = X_com_intercepto.T
X_transposto_X = np.dot(X_transposto, X_com_intercepto)
X_transposto_X_inv = np.linalg.inv(X_transposto_X)
X_transposto_Y = np.dot(X_transposto, Y)
beta_hat = np.dot(X_transposto_X_inv, X_transposto_Y)

# Calcular valores preditos e resíduos
Y_pred = np.dot(X_com_intercepto, beta_hat)
residuos = Y - Y_pred

# 1. Gráfico de Valores Observados vs Preditos
plt.figure(figsize=(10, 6))
sns.scatterplot(x=Y_pred, y=Y, color='steelblue')
plt.plot([Y.min(), Y.max()], [Y.min(), Y.max()], 'r--', lw=2)  # Linha de referência
plt.title('Valores Observados vs Valores Preditos')
plt.xlabel('Valores Preditos')
plt.ylabel('Valores Observados')
plt.grid(True, alpha=0.3)
plt.savefig('/home/ubuntu/observados_vs_preditos.png')
print("Gráfico de Valores Observados vs Preditos salvo em 'observados_vs_preditos.png'")

# 2. Gráfico de Resíduos vs Valores Preditos
plt.figure(figsize=(10, 6))
sns.scatterplot(x=Y_pred, y=residuos, color='steelblue')
plt.axhline(y=0, color='r', linestyle='--')
plt.title('Resíduos vs Valores Preditos')
plt.xlabel('Valores Preditos')
plt.ylabel('Resíduos')
plt.grid(True, alpha=0.3)
plt.savefig('/home/ubuntu/residuos_vs_preditos.png')
print("Gráfico de Resíduos vs Preditos salvo em 'residuos_vs_preditos.png'")

# 3. Histograma dos Resíduos
plt.figure(figsize=(10, 6))
sns.histplot(residuos, kde=True, color='skyblue')
plt.title('Distribuição dos Resíduos')
plt.xlabel('Resíduos')
plt.ylabel('Frequência')
plt.grid(axis='y', alpha=0.3)
plt.savefig('/home/ubuntu/histograma_residuos.png')
print("Histograma dos Resíduos salvo em 'histograma_residuos.png'")

# 4. Q-Q Plot dos Resíduos (implementação simplificada)
from scipy import stats
plt.figure(figsize=(10, 6))
stats.probplot(residuos, dist="norm", plot=plt)
plt.title('Q-Q Plot dos Resíduos')
plt.grid(True, alpha=0.3)
plt.savefig('/home/ubuntu/qq_plot_residuos.png')
print("Q-Q Plot dos Resíduos salvo em 'qq_plot_residuos.png'")

# 5. Gráfico de Correlação entre todas as variáveis quantitativas
plt.figure(figsize=(12, 8))
# Selecionar algumas variáveis quantitativas para a matriz de correlação
vars_correlacao = ['preco_atual', 'market_cap', 'volatilidade', 'retorno_anual', 'pe_ratio', 'beta']
df_corr = df[vars_correlacao].dropna()
correlation_matrix = df_corr.corr()

sns.heatmap(correlation_matrix, annot=True, cmap='Blues', center=0, 
            square=True, fmt='.3f', cbar_kws={"shrink": .8})
plt.title('Matriz de Correlação entre Variáveis Quantitativas')
plt.tight_layout()
plt.savefig('/home/ubuntu/matriz_correlacao.png')
print("Matriz de Correlação salva em 'matriz_correlacao.png'")

print("\nTodos os gráficos de regressão foram gerados com sucesso!")