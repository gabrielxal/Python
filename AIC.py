# -*- coding: utf-8 -*-
"""
Created on Fri Sep 27 14:55:59 2024

@author: Aluno
"""

import pandas as pd
import statsmodels.api as sm
import statsmodels.tsa.stattools as smt
import itertools
import warnings

# Ler a planilha
df = pd.read_excel("C:\\Users\\Aluno\\Documents\\Gabriel\\Darliane\\darlianemog.xlsx")

# teste de Dick-Fuller
teste_temporal = df["Resíduos (t)"]
resultado_adf = smt.adfuller(teste_temporal)

# Extrair resultados
print("Estatística do Teste ADF:", resultado_adf[0])
print("p-value:", resultado_adf[1])
print("Número de Lags Utilizados:", resultado_adf[2])
print("Número de Observações Usadas:", resultado_adf[3])
print("Valores Críticos:")
for key, value in resultado_adf[4].items():
    print(f'   {key}: {value}')
    
# Ajustar o melhor modelo SARIMAX (para sazonalidade)

y = df['Resíduos (t)']


# Definir melhor valores para p d q 
p = d = q = range(0, 3)  
pdq = list(itertools.product(p, d, q))  

# Definir possíveis valores sazonais para (P, D, Q, S) com sazonalidade de 12
seasonal_pdq = [(x[0], x[1], x[2], 12) for x in pdq]

# Ignorar avisos para uma saída mais limpa
warnings.filterwarnings("ignore")

# Variáveis para armazenar o melhor modelo
melhor_aic = float("inf")
melhor_modelo = None
melhor_param = None
melhor_sazonal = None

# Loop para testar todas as combinações de (p, d, q) e (P, D, Q, S)
for param in pdq:
    for param_seasonal in seasonal_pdq:
        try:
            # Ajustar o modelo SARIMAX
            modelo = sm.tsa.SARIMAX(y,
                                    order=param,
                                    seasonal_order=param_seasonal,
                                    enforce_stationarity=False,
                                    enforce_invertibility=False)
            resultado = modelo.fit()

            # Comparar o AIC e armazenar o melhor modelo
            if resultado.aic < melhor_aic:
                melhor_aic = resultado.aic
                melhor_modelo = resultado
                melhor_param = param
                melhor_sazonal = param_seasonal

            print(f"SARIMAX{param}x{param_seasonal} - AIC: {resultado.aic}")
        
        except Exception as e:
            print(f"Erro com SARIMAX{param}x{param_seasonal}: {e}")
            continue

# Imprimir o melhor modelo
print(f"\nMelhor Modelo: SARIMAX{melhor_param}x{melhor_sazonal} - AIC: {melhor_aic}")
print(melhor_modelo.summary())