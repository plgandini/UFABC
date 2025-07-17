import pandas as pd
import numpy as np

# --- PASSO 1: Inserir e preparar os dados da Tabela 43 ---
# Os dados foram fornecidos diretamente na questão.
pesos_pacientes = [
    60.4, 78.9, 65.7, 82.1, 80.9, 92.3, 85.7, 86.6, 90.3, 93.2,
    75.2, 77.3, 80.4, 62.0, 90.4, 70.4, 80.5, 75.9, 55.0, 84.3,
    81.3, 78.3, 70.5, 85.6, 71.9, 77.5, 76.1, 67.7, 80.6, 78.0,
    71.6, 74.8, 92.1, 87.7, 83.8, 93.4, 69.3, 97.8, 81.7, 72.2,
    69.3, 80.2, 90.0, 76.9, 54.7, 78.4, 55.2, 75.5, 99.3, 66.7
]

# Converter a lista para uma Série do Pandas para facilitar a manipulação
dados = pd.Series(pesos_pacientes)

# --- PASSO 2: Determinar as Classes (Intervalos) ---
# Usando a Regra de Sturges para definir o número de classes (k)
n = len(dados)
k = int(1 + 3.322 * np.log10(n))

# Calcular a amplitude e a largura da classe
amplitude = dados.max() - dados.min()
largura_classe = np.ceil(amplitude / k)

# Definir os limites das classes
limite_inferior = int(dados.min())
bins = [limite_inferior + i * largura_classe for i in range(k + 1)]

# --- PASSO 3: Construir a Tabela de Distribuição de Frequências ---

# Agrupar os dados nas classes definidas
classes_dados = pd.cut(dados, bins=bins, include_lowest=True, right=False)

# Criar os rótulos para as classes
rotulos_classes = [f'{bins[i]:.1f} |-- {bins[i+1]:.1f}' for i in range(len(bins)-1)]

# Calcular a frequência absoluta
freq_abs = classes_dados.value_counts(sort=False)

# Criar a tabela final
tabela_freq = pd.DataFrame({
    'Classes de Peso (kg)': rotulos_classes,
    'Frequência Absoluta (fi)': freq_abs.values
})

# Calcular o Ponto Médio (xi)
tabela_freq['Ponto Médio (xi)'] = [(bins[i] + bins[i+1]) / 2 for i in range(len(bins)-1)]

# Calcular Frequência Relativa (fri)
tabela_freq['Frequência Relativa (fri) %'] = (tabela_freq['Frequência Absoluta (fi)'] / n) * 100

# Calcular Frequências Acumuladas
tabela_freq['Frequência Acumulada (Fi)'] = tabela_freq['Frequência Absoluta (fi)'].cumsum()
tabela_freq['Freq. Rel. Acumulada (Fri) %'] = tabela_freq['Frequência Relativa (fri) %'].cumsum()

# Reordenar colunas para melhor visualização
tabela_freq = tabela_freq[['Classes de Peso (kg)', 'Ponto Médio (xi)', 'Frequência Absoluta (fi)', 'Frequência Relativa (fri) %', 'Frequência Acumulada (Fi)', 'Freq. Rel. Acumulada (Fri) %']]

# Adicionar linha de Total
total_row = {
    'Classes de Peso (kg)': 'Total',
    'Ponto Médio (xi)': '---',
    'Frequência Absoluta (fi)': tabela_freq['Frequência Absoluta (fi)'].sum(),
    'Frequência Relativa (fri) %': tabela_freq['Frequência Relativa (fri) %'].sum(),
    'Frequência Acumulada (Fi)': '---',
    'Freq. Rel. Acumulada (Fri) %': '---'
}
# Usar pd.concat para adicionar a linha de total
tabela_freq = pd.concat([tabela_freq, pd.DataFrame([total_row])], ignore_index=True)


# --- PASSO 4: Exibir a Tabela Final ---
print("--- Tabela de Distribuição de Frequências para o Peso dos Pacientes ---")
# Formatar a saída para melhor leitura, convertendo para string
print(tabela_freq.to_string(index=False))