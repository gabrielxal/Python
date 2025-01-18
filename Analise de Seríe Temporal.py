import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import statsmodels.tsa.stattools as smt
import itertools
import warnings

# Ler a planilha
df = pd.read_excel("C:\\Users\\Aluno\\Documents\\Gabriel\\Marquinhos\\serietemporal.xlsx")
                  
# Ajustar modelo ARIMA
modelo = sm.tsa.ARIMA(df["Resíduos (t)"], order=(1, 1, 1))
modelo_resultado = modelo.fit()

# Calcular previsão futura / curto periodo (12 passos à frente)
previsão = modelo_resultado.forecast(steps=12)
print(previsão)


    # aplicar o melhor modelo e valor do AIC
modelo2 = sm.tsa.SARIMAX(
    df["Resíduos (t)"],
    order=(0, 1, 0),
    seasonal_order=(2, 2, 2, 12)
)

resultado2 = modelo2.fit()

# Calcular a previsão futura / com sazonalidade (60 passos)
passos2 = resultado2.get_forecast(steps=60)

# Extrair previsão média e intervalos de confiança
previsão_medios = passos2.predicted_mean
limites = passos2.conf_int()

# Exibir previsões e intervalos de confiança
print(pd.DataFrame({
    'Previsão': previsão_medios,
    'Limite Inferior': limites.iloc[:, 0],
    'Limite Superior': limites.iloc[:, 1]
}))

resultados2 = pd.DataFrame({
    'Previsão': previsão_medios,
    'Limite Inferior': limites.iloc[:, 0],
    'Limite Superior': limites.iloc[:, 1]
})



# Plotar a previsão média
plt.plot(resultados2.index, resultados2['Previsão'], color='blue', label='Previsão')

# Preencher entre os limites inferior e superior
plt.fill_between(resultados2.index, resultados2['Limite Inferior'], resultados2['Limite Superior'], color='gray', alpha=0.3, label='Intervalo de Confiança')

# Adicionar título e rótulos
plt.title('Previsão com Limites de Confiança (SARIMAX)')
plt.xlabel('Passos no Futuro')
plt.ylabel('Resíduos (t)')
plt.legend()

# Exibir o gráfico
plt.grid(True)
plt.tight_layout()
plt.show()





#Salva em um planinha com dois sheets diferentes, um para cada método
with pd.ExcelWriter("C:\\Users\\Aluno\\Documents\\Gabriel\\Marquinhos\\Resultados Análise temporal 2.xlsx", engine='openpyxl') as writer:
     resultados2.to_excel(writer,
                         sheet_name="Modelo SARIMAX",
                         index=False)
     previsão.to_excel(writer,
                      sheet_name="Modelo ARIMA",
                      index=False)
    