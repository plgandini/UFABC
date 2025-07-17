import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency

# --- PASSO 1: Carregar e Preparar os Dados ---
try:
    file_path = '/Users/plgandini/Coding/UFABC/EASG/Inadimplência.xlsx'
    df = pd.read_excel(file_path)

    # Selecionar apenas as duas primeiras colunas que contêm os dados
    df = df.iloc[:, [0, 1]]
    # Renomear as colunas para facilitar o manuseio
    df.columns = ['Faixa_Etaria_Num', 'Inadimplencia_Num']

except FileNotFoundError:
    print(f"Erro: O arquivo '{file_path}' não foi encontrado.")
    exit()
except Exception as e:
    print(f"Ocorreu um erro ao ler o arquivo: {e}")
    exit()

# --- PASSO 2: Mapear Códigos Numéricos para Textos ---

# Dicionários para o mapeamento
mapa_idade = {
    1: 'Até 20 anos',
    2: '21 a 30 anos',
    3: '31 a 40 anos',
    4: '41 a 50 anos',
    5: '51 a 60 anos',
    6: 'Acima de 60 anos'
}

mapa_inad = {
    1: 'Não tem dívidas',
    2: 'Pouco Endividado',
    3: 'Mais ou menos Endividado',
    4: 'Muito Endividado'
}

# Aplicar o mapeamento para criar novas colunas com os textos
df['Faixa Etária'] = df['Faixa_Etaria_Num'].map(mapa_idade)
df['Inadimplência'] = df['Inadimplencia_Num'].map(mapa_inad)

# Definir a ordem correta das categorias de texto para as tabelas
ordem_idade = ['Até 20 anos', '21 a 30 anos', '31 a 40 anos', '41 a 50 anos', '51 a 60 anos', 'Acima de 60 anos']
ordem_inad = ['Não tem dívidas', 'Pouco Endividado', 'Mais ou menos Endividado', 'Muito Endividado']

df['Faixa Etária'] = pd.Categorical(df['Faixa Etária'], categories=ordem_idade, ordered=True)
df['Inadimplência'] = pd.Categorical(df['Inadimplência'], categories=ordem_inad, ordered=True)


# --- PASSO 3: Respostas para os Itens ---

# a) Construção das Tabelas de Distribuição Conjunta
print("--- a) Tabelas de Distribuição Conjunta ---")

# 1. Frequências Absolutas (Observadas)
tabela_obs = pd.crosstab(df['Faixa Etária'], df['Inadimplência'])
print("\n1. Tabela de Frequências Absolutas (Observadas):\n")
print(tabela_obs)

# 2. Frequências Relativas ao Total Geral
total_geral = tabela_obs.sum().sum()
tabela_rel_total = (tabela_obs / total_geral) * 100
print("\n2. Tabela de Frequências Relativas (%) - Em relação ao Total Geral:\n")
print(tabela_rel_total.round(2))

# 3. Frequências Relativas em Relação ao Total de Cada Linha
tabela_rel_linha = pd.crosstab(df['Faixa Etária'], df['Inadimplência'], normalize='index') * 100
print("\n3. Tabela de Frequências Relativas (%) - Em relação ao Total de Cada Linha:\n")
print(tabela_rel_linha.round(2))

# 4. Frequências Relativas em Relação ao Total de Cada Coluna
tabela_rel_coluna = pd.crosstab(df['Faixa Etária'], df['Inadimplência'], normalize='columns') * 100
print("\n4. Tabela de Frequências Relativas (%) - Em relação ao Total de Cada Coluna:\n")
print(tabela_rel_coluna.round(2))

# 5. Frequências Esperadas (se não houvesse associação)
chi2, p_valor, dof, esperadas_array = chi2_contingency(tabela_obs)
tabela_esp = pd.DataFrame(esperadas_array, index=tabela_obs.index, columns=tabela_obs.columns)
print("\n5. Tabela de Frequências Esperadas:\n")
print(tabela_esp.round(2))
print("-" * 50)


# b) Porcentagem de indivíduos na faixa etária entre 31 e 40 anos
porc_31_40 = (tabela_obs.loc['31 a 40 anos'].sum() / total_geral) * 100
print(f"\nb) Porcentagem de indivíduos entre 31 e 40 anos: {porc_31_40:.2f}%")

# c) Porcentagem de indivíduos muito endividados
porc_muito_endividados = (tabela_obs['Muito Endividado'].sum() / total_geral) * 100
print(f"c) Porcentagem de indivíduos muito endividados: {porc_muito_endividados:.2f}%")

# d) Porcentagem daqueles que são da faixa etária até 20 anos e que não tem dívidas
porc_ate20_semdivida = (tabela_obs.loc['Até 20 anos', 'Não tem dívidas'] / total_geral) * 100
print(f"d) % de 'Até 20 anos' e 'Não tem dívidas': {porc_ate20_semdivida:.2f}%")

# e) Dentre os indivíduos da faixa etária acima de 60 anos, quantos por cento são pouco endividados?
porc_acima60_pouco = tabela_rel_linha.loc['Acima de 60 anos', 'Pouco Endividado']
print(f"e) Dentre os 'Acima de 60 anos', {porc_acima60_pouco:.2f}% são 'Pouco Endividado'")

# f) Dentre os indivíduos mais ou menos endividados, quantos por cento pertencem à faixa etária entre 41 e 50 anos?
porc_maismenos_41a50 = tabela_rel_coluna.loc['41 a 50 anos', 'Mais ou menos Endividado']
print(f"f) Dentre os 'Mais ou menos Endividado', {porc_maismenos_41a50:.2f}% são da faixa '41 a 50 anos'")
print("-" * 50)

# g) Há indícios de dependência entre as variáveis?
print("\ng) Análise de Indícios de Dependência:")
print("Sim, há fortes indícios de dependência. Ao comparar a tabela de frequências observadas com as esperadas, notamos grandes diferenças.")
print("Por exemplo, na célula ('Até 20 anos', 'Não tem dívidas'), observamos 52 indivíduos, mas esperaríamos apenas 21.62 se não houvesse associação.")
print("Além disso, as porcentagens nas linhas (item a.3) não são constantes. Por exemplo, 74.29% da faixa 'Até 20 anos' não tem dívidas, enquanto apenas 8.11% da faixa '51 a 60 anos' está nessa condição. Se fossem independentes, essas porcentagens seriam similares.")
print("-" * 50)

# h) Confirmação com o teste Qui-Quadrado
print("\nh) Teste Qui-Quadrado de Independência:")
alpha = 0.05 # Nível de significância padrão
print(f"   - Hipótese Nula (H0): As variáveis 'Faixa Etária' e 'Inadimplência' são independentes.")
print(f"   - Hipótese Alternativa (H1): As variáveis são dependentes (associadas).")
print(f"\n   - Estatística Qui-Quadrado (χ²): {chi2:.4f}")
print(f"   - Graus de Liberdade (dof): {dof}")
print(f"   - p-valor: {p_valor:.10f}")
print(f"\n   - Decisão (com alpha = {alpha}):")
if p_valor < alpha:
    print(f"     Como o p-valor ({p_valor:.10f}) é menor que o nível de significância ({alpha}), rejeitamos a Hipótese Nula.")
    print("     Conclusão: Existe uma associação estatisticamente significativa entre a faixa etária e o grau de inadimplência.")
else:
    print(f"     Como o p-valor ({p_valor:.10f}) é maior ou igual ao nível de significância ({alpha}), não rejeitamos a Hipótese Nula.")
    print("     Conclusão: Não há evidência estatística suficiente para afirmar que existe uma associação entre as variáveis.")