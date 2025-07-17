import pandas as pd
from scipy.stats import chi2_contingency

# --- PASSO 1: Criar a Tabela de Frequências Observadas ---
# Inserimos os dados da tabela fornecida na questão em um DataFrame do Pandas.

data = {
    'SP': [15, 25, 40, 30],
    'RJ': [10, 20, 12, 8],
    'RS': [15, 10, 8, 7]
}
index = ['Micro', 'Pequena', 'Média', 'Grande']

tabela_observada = pd.DataFrame(data, index=index)

print("--- Tabela de Frequências Observadas ---")
print(tabela_observada)


# --- PASSO 2: Realizar o Teste Qui-Quadrado ---
chi2, p_valor, dof, esperadas = chi2_contingency(tabela_observada)

# Formatar a tabela de frequências esperadas para melhor visualização
tabela_esperada = pd.DataFrame(esperadas, index=index, columns=tabela_observada.columns)

print("\n--- Tabela de Frequências Esperadas ---")
print("(Valores que seriam esperados se não houvesse associação entre as variáveis)")
print(tabela_esperada.round(2))


# --- PASSO 3: Interpretar os Resultados do Teste ---
print("\n--- Análise do Teste Qui-Quadrado ---")

# Definição das hipóteses
print("\nDefinição das Hipóteses:")
print("  - Hipótese Nula (H0): NÃO HÁ associação entre Localidade e Porte da Empresa.")
print("  - Hipótese Alternativa (H1): EXISTE associação entre Localidade e Porte da Empresa.")

# Apresentação dos resultados do teste
print("\nResultados do Teste:")
print(f"  - Valor da Estatística Qui-Quadrado (χ²): {chi2:.4f}")
print(f"  - Graus de Liberdade: {dof}")
print(f"  - P-valor: {p_valor:.4f}")

# Decisão estatística
alpha = 0.05
print(f"\nDecisão (com nível de significância de {alpha*100}%):")
if p_valor < alpha:
    print(f"  - Como o p-valor ({p_valor:.4f}) é MENOR que {alpha}, REJEITAMOS a Hipótese Nula.")
    print("  - Conclusão: Há evidência estatística forte para afirmar que existe uma associação significativa entre a localidade de origem e o porte da empresa.")
    
    # --- PASSO 4: Análise Detalhada da Associação ---
    print("\n--- Análise Detalhada da Associação ---")
    print("O teste confirma a associação. Para entender onde ela ocorre, comparamos os valores observados e esperados:")
    
    print("\n1. Associação entre SP e Empresas de Maior Porte:")
    print(f"   - Empresas Médias: Observamos {tabela_observada.loc['Média', 'SP']} em SP, mas o esperado era apenas {tabela_esperada.loc['Média', 'SP']:.2f}.")
    print(f"   - Empresas Grandes: Observamos {tabela_observada.loc['Grande', 'SP']} em SP, superando as {tabela_esperada.loc['Grande', 'SP']:.2f} esperadas.")
    print("   - Conclusão: SP concentra mais empresas de médio e grande porte do que o esperado.")

    print("\n2. Associação entre RJ e Pequenas Empresas:")
    print(f"   - Empresas Pequenas: No RJ, observamos {tabela_observada.loc['Pequena', 'RJ']} empresas, significativamente mais que as {tabela_esperada.loc['Pequena', 'RJ']:.2f} esperadas.")
    print("   - Conclusão: O RJ mostra uma forte vocação para empresas de pequeno porte.")
    
    print("\n3. Associação entre RS e Microempresas:")
    print(f"   - Microempresas: No RS, observamos {tabela_observada.loc['Micro', 'RS']} empresas, quase o dobro das {tabela_esperada.loc['Micro', 'RS']:.2f} esperadas.")
    print("   - Conclusão: O RS se destaca pela alta concentração de microempresas na amostra.")

else:
    print(f"  - Como o p-valor ({p_valor:.4f}) é MAIOR ou IGUAL a {alpha}, NÃO REJEITAMOS a Hipótese Nula.")
    print("  - Conclusão: Não há evidência estatística suficiente para afirmar que existe uma associação entre a localidade de origem e o porte da empresa.")