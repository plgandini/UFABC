import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")

# Configuração para exibir gráficos em português
plt.rcParams["font.size"] = 10
plt.rcParams["figure.figsize"] = (10, 6)

# Lista de empresas financeiras dos EUA selecionadas
tickers = ["JPM", "BAC", "WFC", "C", "GS", "MS", "BLK", "SCHW", "USB", "PNC", 
           "COF", "AXP", "V", "MA", "PYPL", "TFC", "MTB", "FITB", "KEY", "RF"]

print("Coletando dados das empresas financeiras dos EUA...")
print(f"Tickers selecionados: {tickers}")

# Período de coleta de dados (últimos 2 anos)
end_date = datetime.now()
start_date = end_date - timedelta(days=730)

# Dicionário para armazenar os dados
dados_empresas = {}

# Coletando dados para cada ticker
for ticker in tickers:
    try:
        print(f"Coletando dados para {ticker}...")
        stock = yf.Ticker(ticker)
        
        # Informações da empresa
        info = stock.info
        
        # Dados históricos de preços
        hist = stock.history(start=start_date, end=end_date)
        
        # Calculando métricas
        preco_atual = hist["Close"].iloc[-1] if len(hist) > 0 else None
        preco_52w_max = hist["High"].max() if len(hist) > 0 else None
        preco_52w_min = hist["Low"].min() if len(hist) > 0 else None
        volume_medio = hist["Volume"].mean() if len(hist) > 0 else None
        volatilidade = hist["Close"].pct_change().std() * np.sqrt(252) if len(hist) > 0 else None
        
        # Retorno anual
        if len(hist) >= 252:
            retorno_anual = ((hist["Close"].iloc[-1] / hist["Close"].iloc[-252]) - 1) * 100
        else:
            retorno_anual = None
            
        dados_empresas[ticker] = {
            "nome": info.get("longName", "N/A"),
            "setor": info.get("sector", "Financial Services"),
            "industria": info.get("industry", "N/A"),
            "market_cap": info.get("marketCap", None),
            "preco_atual": preco_atual,
            "preco_52w_max": preco_52w_max,
            "preco_52w_min": preco_52w_min,
            "volume_medio": volume_medio,
            "volatilidade": volatilidade,
            "retorno_anual": retorno_anual,
            "pe_ratio": info.get("trailingPE", None),
            "pb_ratio": info.get("priceToBook", None),
            "dividend_yield": info.get("dividendYield", None),
            "beta": info.get("beta", None),
            "roa": info.get("returnOnAssets", None),
            "roe": info.get("returnOnEquity", None),
            "employees": info.get("fullTimeEmployees", None),
            "country": info.get("country", "United States"),
            "state": info.get("state", "N/A"),
            "city": info.get("city", "N/A")
        }
        
    except Exception as e:
        print(f"Erro ao coletar dados para {ticker}: {e}")
        continue

# Convertendo para DataFrame
df = pd.DataFrame.from_dict(dados_empresas, orient="index")
df.reset_index(inplace=True)
df.rename(columns={"index": "ticker"}, inplace=True)

# Adicionando variável qualitativa ordinal: Porte da Empresa (baseado em market_cap)
def classificar_porte(market_cap):
    if pd.isna(market_cap):
        return "N/A"
    elif market_cap < 50_000_000_000: # Menos de 50 bilhões
        return "Pequeno Porte"
    elif market_cap < 200_000_000_000: # Entre 50 e 200 bilhões
        return "Médio Porte"
    else: # Acima de 200 bilhões
        return "Grande Porte"

df["porte_empresa"] = df["market_cap"].apply(classificar_porte)

print(f"\nDados coletados para {len(df)} empresas")
print("\nPrimeiras 5 linhas do dataset:")
print(df.head())

# Salvando os dados
df.to_csv("dados_financeiros.csv", index=False)
print("\nDados salvos em \'dados_financeiros.csv\' ")

# Informações básicas do dataset
print("\n" + "="*50)
print("INFORMAÇÕES BÁSICAS DO DATASET")
print("="*50)
print(f"Número de empresas: {len(df)}")
print(f"Número de variáveis: {len(df.columns)}")
print("\nVariáveis do dataset:")
for i, col in enumerate(df.columns, 1):
    print(f"{i:2d}. {col}")

print("\nTipos de dados:")
print(df.dtypes)

print("\nEstatísticas descritivas das variáveis numéricas:")
print(df.describe())

# Verificando valores ausentes
print("\nValores ausentes por variável:")
print(df.isnull().sum())

print("\nDataset criado com sucesso!")

