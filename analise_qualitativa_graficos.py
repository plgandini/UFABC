import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Carregar os dados
df = pd.read_csv("dados_financeiros.csv")

# Escolher duas variáveis qualitativas para análise
variavel_qualitativa_1 = 'porte_empresa'
variavel_qualitativa_2 = 'state'

# Remover linhas com valores ausentes nas variáveis selecionadas
df_filtered = df[[variavel_qualitativa_1, variavel_qualitativa_2]].dropna()

# Criar um gráfico de barras empilhadas ou agrupadas para visualizar a associação
plt.figure(figsize=(12, 7))
sns.countplot(data=df_filtered, x=variavel_qualitativa_2, hue=variavel_qualitativa_1, palette='Blues')
plt.title(f'Distribuição de {variavel_qualitativa_1} por {variavel_qualitativa_2}')
plt.xlabel(variavel_qualitativa_2)
plt.ylabel('Contagem de Empresas')
plt.xticks(rotation=45, ha='right')
plt.legend(title=variavel_qualitativa_1)
plt.tight_layout()
plt.savefig('distribuicao_porte_estado.png')
print("Gráfico de barras salvo em 'distribuicao_porte_estado.png'")