import pandas as pd

# 1. Dados da Tabela 42
dados_vendas = [
    7, 5, 9, 11, 10, 8, 9, 6, 8, 10,
    8, 5, 7, 11, 9, 11, 6, 7, 10, 9,
    8, 5, 6, 8, 6, 7, 6, 5, 10, 8
]

# 2. Criar uma Série pandas com os dados
s_vendas = pd.Series(dados_vendas)

# 3. Construir a tabela de frequências
# Contar a frequência absoluta de cada valor
freq_abs = s_vendas.value_counts().sort_index()

# Criar o DataFrame da tabela final
tabela_freq = pd.DataFrame(freq_abs)
tabela_freq.columns = ['Frequência Absoluta (fi)']
tabela_freq.index.name = 'Vendas Diárias (xi)'

# Calcular a Frequência Relativa (em decimal)
total_obs = len(dados_vendas)
tabela_freq['Frequência Relativa (fri)'] = tabela_freq['Frequência Absoluta (fi)'] / total_obs

# Calcular a Frequência Acumulada
tabela_freq['Frequência Acumulada (Fi)'] = tabela_freq['Frequência Absoluta (fi)'].cumsum()

# Calcular a Frequência Relativa Acumulada
tabela_freq['Frequência Relativa Acumulada (Fri)'] = tabela_freq['Frequência Relativa (fri)'].cumsum()

# Adicionar uma linha de Total
total_row = {
    'Frequência Absoluta (fi)': tabela_freq['Frequência Absoluta (fi)'].sum(),
    'Frequência Relativa (fri)': tabela_freq['Frequência Relativa (fri)'].sum(),
    'Frequência Acumulada (Fi)': '',  # Não se aplica
    'Frequência Relativa Acumulada (Fri)': '' # Não se aplica
}
tabela_freq.loc['Total'] = total_row

# Exibir a tabela formatada (opcional: converter para porcentagem para melhor visualização)
tabela_freq_view = tabela_freq.copy()
tabela_freq_view['Frequência Relativa (fri)'] = (tabela_freq_view['Frequência Relativa (fri)'] * 100).map('{:,.2f}%'.format)
tabela_freq_view['Frequência Relativa Acumulada (Fri)'] = tabela_freq_view['Frequência Relativa Acumulada (Fri)'].apply(
    lambda x: f"{x*100:.2f}%" if isinstance(x, (int, float)) else x
)

print(f"\n\n{tabela_freq_view.to_string()}\n\n")