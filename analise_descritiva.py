import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Carregar os dados
df = pd.read_csv("dados_financeiros.csv")

# Escolher uma variável quantitativa para análise
variavel_quantitativa = 'market_cap'
dados = df[variavel_quantitativa].dropna().values

print(f"\nAnálise Descritiva da Variável: {variavel_quantitativa}\n")

# 1. Medidas de Posição
print("1. Medidas de Posição")
print("---------------------")

# a) Média Aritmética
print("a) Média Aritmética:")
soma_valores = 0
for x in dados:
    soma_valores += x
media = soma_valores / len(dados)
print(f"   Soma dos valores: {soma_valores:.2f}")
print(f"   Número de observações: {len(dados)}")
print(f"   Média: {media:.2f}\n")

# b) Mediana
print("b) Mediana:")
dados_ordenados = np.sort(dados)
n = len(dados_ordenados)
if n % 2 == 0:
    # Se o número de observações for par
    pos1 = n // 2 - 1
    pos2 = n // 2
    mediana = (dados_ordenados[pos1] + dados_ordenados[pos2]) / 2
    print(f"   Dados ordenados: {dados_ordenados}")
    print(f"   Posições centrais: {pos1+1} e {pos2+1}")
    print(f"   Valores nas posições centrais: {dados_ordenados[pos1]:.2f} e {dados_ordenados[pos2]:.2f}")
else:
    # Se o número de observações for ímpar
    pos = n // 2
    mediana = dados_ordenados[pos]
    print(f"   Dados ordenados: {dados_ordenados}")
    print(f"   Posição central: {pos+1}")
    print(f"   Valor na posição central: {dados_ordenados[pos]:.2f}")
print(f"   Mediana: {mediana:.2f}\n")

# c) Moda
print("c) Moda:")
frequencias = {}
for x in dados:
    frequencias[x] = frequencias.get(x, 0) + 1

if not frequencias:
    moda = "Não há moda (dados vazios)"
    print(f"   {moda}\n")
else:
    max_frequencia = 0
    for freq in frequencias.values():
        if freq > max_frequencia:
            max_frequencia = freq

    modas = []
    for valor, freq in frequencias.items():
        if freq == max_frequencia:
            modas.append(valor)

    if max_frequencia == 1 and len(modas) == len(dados):
        moda = "Não há moda (todos os valores são únicos)"
    elif len(modas) == 1:
        moda = modas[0]
    else:
        moda = modas
    print(f"   Frequências: {frequencias}")
    print(f"   Moda(s): {moda}\n")

# 2. Quartis e Outliers
print("2. Quartis e Indícios de Outliers")
print("----------------------------------")

# Função para calcular percentil (generalizada para quartis, decis, percentis)
def calcular_percentil(dados_ordenados, p):
    n = len(dados_ordenados)
    posicao = (n - 1) * p / 100
    
    if posicao < 0 or posicao >= n:
        raise ValueError("Posição do percentil fora dos limites dos dados.")

    if posicao == int(posicao):
        return dados_ordenados[int(posicao)]
    else:
        k = int(posicao)
        f = posicao - k
        return dados_ordenados[k] + f * (dados_ordenados[k+1] - dados_ordenados[k])

# a) Primeiro e Terceiro Quartil
print("a) Primeiro e Terceiro Quartil:")
q1 = calcular_percentil(dados_ordenados, 25)
q3 = calcular_percentil(dados_ordenados, 75)
print(f"   Primeiro Quartil (Q1): {q1:.2f}")
print(f"   Terceiro Quartil (Q3): {q3:.2f}\n")

# Indícios de Outliers (IQR)
print("   Indícios de Outliers (Método IQR):")
iqr = q3 - q1
limite_inferior = q1 - 1.5 * iqr
limite_superior = q3 + 1.5 * iqr

outliers = []
for x in dados:
    if x < limite_inferior or x > limite_superior:
        outliers.append(x)

print(f"   Intervalo Interquartil (IQR): {iqr:.2f}")
print(f"   Limite Inferior para Outliers: {limite_inferior:.2f}")
print(f"   Limite Superior para Outliers: {limite_superior:.2f}")
if outliers:
    print(f"   Outliers encontrados: {outliers}")
else:
    print("   Não há indícios de outliers.\n")

# b) Percentis de ordem 20 e 80
print("b) Percentis de ordem 20 e 80:")
p20 = calcular_percentil(dados_ordenados, 20)
p80 = calcular_percentil(dados_ordenados, 80)
print(f"   Percentil de ordem 20 (P20): {p20:.2f}")
print(f"   Percentil de ordem 80 (P80): {p80:.2f}\n")

# c) Decis de ordem 4 e 7
print("c) Decis de ordem 4 e 7:")
d4 = calcular_percentil(dados_ordenados, 40)
d7 = calcular_percentil(dados_ordenados, 70)
print(f"   Decil de ordem 4 (D4): {d4:.2f}")
print(f"   Decil de ordem 7 (D7): {d7:.2f}\n")

# 3. Medidas de Dispersão
print("3. Medidas de Dispersão")
print("-----------------------")

# a) Amplitude
print("a) Amplitude:")
amplitude = dados_ordenados[-1] - dados_ordenados[0]
print(f"   Valor Máximo: {dados_ordenados[-1]:.2f}")
print(f"   Valor Mínimo: {dados_ordenados[0]:.2f}")
print(f"   Amplitude: {amplitude:.2f}\n")

# b) Desvio-Médio
print("b) Desvio-Médio:")
soma_desvios_abs = 0
for x in dados:
    soma_desvios_abs += abs(x - media)
desvio_medio = soma_desvios_abs / len(dados)
print(f"   Soma dos desvios absolutos em relação à média: {soma_desvios_abs:.2f}")
print(f"   Desvio-Médio: {desvio_medio:.2f}\n")

# c) Variância
print("c) Variância:")
soma_quadrados_desvios = 0
for x in dados:
    soma_quadrados_desvios += (x - media)**2
variancia = soma_quadrados_desvios / (len(dados) - 1) # Variância amostral
print(f"   Soma dos quadrados dos desvios em relação à média: {soma_quadrados_desvios:.2f}")
print(f"   Variância (amostral): {variancia:.2f}\n")

# d) Desvio-Padrão
print("d) Desvio-Padrão:")
desvio_padrao = variancia**0.5
print(f"   Desvio-Padrão (amostral): {desvio_padrao:.2f}\n")

# 4. Assimetria
print("4. Assimetria")
print("-------------")

# Coeficiente de Assimetria (Fisher-Pearson)
soma_cubos_desvios = 0
for x in dados:
    soma_cubos_desvios += (x - media)**3

# Evitar divisão por zero se desvio_padrao for 0
if desvio_padrao == 0:
    coef_assimetria = 0
else:
    coef_assimetria = (soma_cubos_desvios / len(dados)) / (desvio_padrao**3)

print(f"   Soma dos cubos dos desvios em relação à média: {soma_cubos_desvios:.2f}")
print(f"   Coeficiente de Assimetria: {coef_assimetria:.2f}")
if coef_assimetria > 0:
    print("   Distribuição: Assimétrica Positiva (ou à direita)\n")
elif coef_assimetria < 0:
    print("   Distribuição: Assimétrica Negativa (ou à esquerda)\n")
else:
    print("   Distribuição: Simétrica\n")

# 5. Curtose
print("5. Curtose")
print("----------")

# Coeficiente de Curtose (Fisher)
soma_quartas_potencias_desvios = 0
for x in dados:
    soma_quartas_potencias_desvios += (x - media)**4

# Evitar divisão por zero se desvio_padrao for 0
if desvio_padrao == 0:
    coef_curtose = 0
else:
    coef_curtose = (soma_quartas_potencias_desvios / len(dados)) / (desvio_padrao**4) - 3 # Excesso de Curtose

print(f"   Soma das quartas potências dos desvios em relação à média: {soma_quartas_potencias_desvios:.2f}")
print(f"   Coeficiente de Curtose (Excesso): {coef_curtose:.2f}")
if coef_curtose > 0:
    print("   Classificação: Leptocúrtica (mais achatada que a normal, caudas mais pesadas)\n")
elif coef_curtose < 0:
    print("   Classificação: Platicúrtica (menos achatada que a normal, caudas mais leves)\n")
else:
    print("   Classificação: Mesocúrtica (similar à normal)\n")

# 6. Gráficos
print("6. Gráficos")
print("-----------")

# a) Histograma
plt.figure(figsize=(10, 6))
sns.histplot(dados, kde=True, color='skyblue') # Usando skyblue para tom de azul
plt.title(f'Histograma de {variavel_quantitativa}')
plt.xlabel(variavel_quantitativa)
plt.ylabel('Frequência')
plt.grid(axis='y', alpha=0.75)
plt.savefig(f'/home/ubuntu/histograma_{variavel_quantitativa}.png')
print(f"   Histograma salvo em 'histograma_{variavel_quantitativa}.png'\n")

# b) Gráfico Ramo-e-Folhas (implementação simplificada)
print("b) Gráfico Ramo-e-Folhas (representação textual simplificada):\n")

def stem_and_leaf_adapted(data):
    stems = {}
    for x in data:
        # Para market_cap, que são números muito grandes, vou dividir por 1 bilhão para simplificar a visualização
        # e usar a parte inteira como ramo e o primeiro decimal como folha.
        x_scaled = x / 1_000_000_000 # Escalar para bilhões
        
        # Ajuste para lidar com a amplitude dos dados
        if x_scaled < 100:
            stem = int(x_scaled // 10)
            leaf = int(x_scaled % 10)
        elif x_scaled < 1000:
            stem = int(x_scaled // 10)
            leaf = int(x_scaled % 10)
        else:
            stem = int(x_scaled // 100)
            leaf = int((x_scaled % 100) // 10) # Pegar a dezena como folha

        if stem not in stems:
            stems[stem] = []
        stems[stem].append(leaf)
    
    for stem in sorted(stems.keys()):
        stems[stem].sort()
        print(f"{stem:4d} | {' '.join(map(str, stems[stem]))}")

stem_and_leaf_adapted([x for x in dados]) # Passar os dados originais para a função adaptada


# c) Boxplot
plt.figure(figsize=(8, 6))
sns.boxplot(y=dados, color='skyblue') # Usando skyblue para tom de azul
plt.title(f'Boxplot de {variavel_quantitativa}')
plt.xlabel('') # Remover label do eixo x para boxplot vertical
plt.ylabel(variavel_quantitativa)
plt.grid(axis='y', alpha=0.75)
plt.savefig(f'/home/ubuntu/boxplot_{variavel_quantitativa}.png')
print(f"   Boxplot salvo em 'boxplot_{variavel_quantitativa}.png'\n")

print("Análise descritiva concluída.")