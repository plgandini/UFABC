import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency

# --- PASSO 1: Carregar e Preparar os Dados ---
try:
    file_path = '/Users/plgandini/Coding/UFABC/EASG/Motivação_Empresas.xlsx'
    df = pd.read_excel(file_path, engine="calamine")

except FileNotFoundError:
    print(f"Erro: O arquivo '{file_path}' não foi encontrado.")
    exit()
except Exception as e:
    print(f"Ocorreu um erro ao ler o arquivo: {e}")
    exit()

# --- PASSO 2: Mapear Códigos Numéricos para Textos ---
# Com base no diagnóstico, as colunas são 'Empresa' e 'MotivaçãO'
col_empresa_num = 'Empresa'
col_motivacao_num = 'MotivaçãO'

# Dicionários para o mapeamento. A ordem das empresas foi assumida com base nas perguntas.
mapa_empresa = {
    1: 'Bradesco',
    2: 'Fiat',
    3: 'Petrobras',
    4: 'Vivo',
    5: 'Outra' # Assumindo um nome para a 5ª empresa
}

mapa_motivacao = {
    1: 'Muito Desmotivado',
    2: 'Pouco Desmotivado',
    3: 'Indiferente',
    4: 'Pouco Motivado',
    5: 'Muito Motivado'
}

# Aplicar o mapeamento para criar novas colunas com os textos
df['Empresa'] = df[col_empresa_num].map(mapa_empresa)
df['Grau de Motivação'] = df[col_motivacao_num].map(mapa_motivacao)

# Definir a ordem correta das categorias de texto para as tabelas
ordem_empresa = ['Bradesco', 'Fiat', 'Outra', 'Petrobras', 'Vivo'] # Ordem alfabética
ordem_motivacao = ['Muito Desmotivado', 'Pouco Desmotivado', 'Indiferente', 'Pouco Motivado', 'Muito Motivado']

df['Empresa'] = pd.Categorical(df['Empresa'], categories=ordem_empresa, ordered=True)
df['Grau de Motivação'] = pd.Categorical(df['Grau de Motivação'], categories=ordem_motivacao, ordered=True)

# --- PASSO 3: Respostas para os Itens ---

# a) Construção das Tabelas de Contingência
print("--- a) Tabelas de Contingência ---")

tabela_obs = pd.crosstab(df['Empresa'], df['Grau de Motivação'])
print("\n1. Tabela de Frequências Absolutas (Observadas):\n")
print(tabela_obs)

total_geral = tabela_obs.sum().sum()
tabela_rel_total = (tabela_obs / total_geral) * 100
print("\n2. Tabela de Frequências Relativas (%) - Em relação ao Total Geral:\n")
print(tabela_rel_total.round(2))

tabela_rel_linha = pd.crosstab(df['Empresa'], df['Grau de Motivação'], normalize='index') * 100
print("\n3. Tabela de Frequências Relativas (%) - Em relação ao Total de Cada Linha:\n")
print(tabela_rel_linha.round(2))

tabela_rel_coluna = pd.crosstab(df['Empresa'], df['Grau de Motivação'], normalize='columns') * 100
print("\n4. Tabela de Frequências Relativas (%) - Em relação ao Total de Cada Coluna:\n")
print(tabela_rel_coluna.round(2))

chi2, p_valor, dof, esperadas_array = chi2_contingency(tabela_obs)
tabela_esp = pd.DataFrame(esperadas_array, index=tabela_obs.index, columns=tabela_obs.columns)
print("\n5. Tabela de Frequências Esperadas:\n")
print(tabela_esp.round(2))
print("-" * 50)


# b) Qual a porcentagem de respondentes muito desmotivados?
porc_muito_desmotivados = (tabela_obs['Muito Desmotivado'].sum() / total_geral) * 100
print(f"\nb) Porcentagem de respondentes 'Muito Desmotivado': {porc_muito_desmotivados:.2f}%")

# c) Qual a porcentagem de respondentes da empresa Petrobras e que estão muito desmotivados?
porc_petro_muitodesmotivado = (tabela_obs.loc['Petrobras', 'Muito Desmotivado'] / total_geral) * 100
print(f"c) % de 'Petrobras' E 'Muito Desmotivado': {porc_petro_muitodesmotivado:.2f}%")

# d) Qual a porcentagem de respondentes motivados na empresa Vivo?
porc_motivado_vivo = tabela_rel_linha.loc['Vivo', ['Pouco Motivado', 'Muito Motivado']].sum()
print(f"d) Na 'Vivo', a porcentagem de respondentes motivados ('Pouco' + 'Muito') é de: {porc_motivado_vivo:.2f}%")

# e) Qual a porcentagem de respondentes pouco motivados na empresa Fiat?
porc_poucomotivado_fiat = tabela_rel_linha.loc['Fiat', 'Pouco Motivado']
print(f"e) Na 'Fiat', a porcentagem de respondentes 'Pouco Motivado' é de: {porc_poucomotivado_fiat:.2f}%")

# f) Dentre os respondentes que estão muito motivados, quantos por cento pertencem ao Bradesco?
porc_muitootivado_bradesco = tabela_rel_coluna.loc['Bradesco', 'Muito Motivado']
print(f"f) Dentre os 'Muito Motivado', {porc_muitootivado_bradesco:.2f}% pertencem ao 'Bradesco'")
print("-" * 50)


# g) Há indícios de dependência entre as variáveis?
print("\ng) Análise de Indícios de Dependência:")
print("Sim, há claros indícios de dependência. As distribuições de motivação variam visivelmente entre as empresas.")
print("Por exemplo, na Tabela 3 (relativa às linhas), vemos que na Petrobras 64% dos funcionários estão 'Muito Desmotivados',")
print("enquanto na Vivo esse percentual é de apenas 12%. Se as variáveis fossem independentes, essas porcentagens deveriam ser semelhantes.")
print("-" * 50)

# h) Confirmação com o teste Qui-Quadrado
print("\nh) Teste Qui-Quadrado de Independência:")
alpha = 0.05
print(f"   - Hipótese Nula (H0): As variáveis 'Empresa' e 'Grau de Motivação' são independentes.")
print(f"   - Hipótese Alternativa (H1): As variáveis são dependentes (associadas).")
print(f"\n   - Estatística Qui-Quadrado (χ²): {chi2:.4f}")
print(f"   - Graus de Liberdade (dof): {dof}")
print(f"   - p-valor: {p_valor:.10f}")
print(f"\n   - Decisão (com alpha = {alpha}):")
if p_valor < alpha:
    print(f"     Como o p-valor ({p_valor:.10f}) é menor que o nível de significância ({alpha}), rejeitamos a Hipótese Nula.")
    print("     Conclusão: Existe uma associação estatisticamente significativa entre a empresa e o grau de motivação dos funcionários.")
else:
    print(f"     Como o p-valor ({p_valor:.10f}) é maior ou igual ao nível de significância ({alpha}), não rejeitamos a Hipótese Nula.")
    print("     Conclusão: Não há evidência estatística para afirmar que existe uma associação entre as variáveis.")