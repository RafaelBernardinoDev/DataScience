import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline

# Definindo os parâmetros do modelo
beta0 = -4.0  # Ajustando beta0 para permitir resultados mais baixos
beta1 = 0.02  # Coeficiente para iluminação
beta2 = 0.05  # Coeficiente para qualidade da imagem
beta3 = 0.01  # Coeficiente para ISO
beta4 = 0.03  # Coeficiente para abertura

# Definindo os intervalos de iluminação, qualidade da imagem, ISO e abertura
iluminacao_min = 100
iluminacao_max = 700
qualidade_min = 0.0
qualidade_max = 1.0  # Aumentando o intervalo para qualidade da imagem
iso_min = 100
iso_max = 800  # Aumentando o intervalo de ISO
abertura_min = 0.5
abertura_max = 4.0

# Listas para armazenar os resultados
iluminacao_list = []
qualidade_list = []
iso_list = []
abertura_list = []
taxa_efetividade_list = []

# Função logística
def funcao_logistica(z):
    return 1 / (1 + np.exp(-z))

# Número de simulações de Monte Carlo
num_simulacoes = 10000

# Realizando simulações de Monte Carlo
for _ in range(num_simulacoes):
    ilum = np.random.uniform(iluminacao_min, iluminacao_max)
    qual = np.random.uniform(qualidade_min, qualidade_max)
    iso = np.random.uniform(iso_min, iso_max)
    abertura = np.random.uniform(abertura_min, abertura_max)

    z = (beta0 + beta1 * ilum + beta2 * qual + beta3 * iso + beta4 * abertura)
    taxa_efetividade = funcao_logistica(z)

    # Armazenando os resultados nas listas
    iluminacao_list.append(ilum)
    qualidade_list.append(qual)
    iso_list.append(iso)
    abertura_list.append(abertura)
    taxa_efetividade_list.append(taxa_efetividade)

# Criando o DataFrame
resultados_df = pd.DataFrame({
    'Iluminacao': iluminacao_list,
    'Qualidade_Imagem': qualidade_list,
    'ISO': iso_list,
    'Abertura': abertura_list,
    'Taxa_Efetividade': taxa_efetividade_list
})

# Agrupando por iluminação e média das taxas de efetividade
media_taxa_efetividade = resultados_df.groupby(['Iluminacao']).mean().reset_index()

# Cálculo da média da qualidade da imagem
media_qualidade_imagem = resultados_df['Qualidade_Imagem'].mean()

# Visualizando os resultados em gráfico de linha
plt.figure(figsize=(10, 6))
plt.plot(media_taxa_efetividade['Iluminacao'], media_taxa_efetividade['Taxa_Efetividade'], marker='o', linestyle='-', label='Taxa de Efetividade Média')
plt.axhline(y=0.75, color='r', linestyle='--', label='Linha de Efetividade 75%')  # Linha de efetividade
plt.axhline(y=media_qualidade_imagem, color='g', linestyle='--', label='Média da Qualidade da Imagem')  # Linha de qualidade da imagem
plt.xlabel('Iluminação (lux)')
plt.ylabel('Taxa de Efetividade do Reconhecimento Facial')
plt.title('Taxa de Efetividade do Reconhecimento Facial (Monte Carlo)')
plt.legend()
plt.grid()
plt.show()

# Exibindo os resultados da tabela, incluindo a qualidade da imagem
print("Resultados da Simulação:")
print(resultados_df)

# Contando reconhecimentos
reconhecidos = resultados_df[resultados_df['Taxa_Efetividade'] > 0.99].shape[0]
nao_reconhecidos = resultados_df[resultados_df['Taxa_Efetividade'] <= 0.99].shape[0]

# Exibindo a quantidade de taxas de reconhecimento positivas e negativas
print(f"Quantidade de reconhecimentos positivos: {reconhecidos}")
print(f"Quantidade de reconhecimentos negativos: {nao_reconhecidos}")

#verificar se vamos usar
import matplotlib.pyplot as plt
import seaborn as sns

# Gráfico de dispersão (scatter plot) para visualizar a relação entre iluminação e taxa de efetividade
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Iluminacao', y='Taxa_Efetividade', data=resultados_df, alpha=0.5)
plt.xlabel('Iluminação')
plt.ylabel('Taxa de Efetividade')
plt.title('Relação entre Iluminação e Taxa de Efetividade')
plt.grid(True)
plt.show()

"""Cada ponto no gráfico representa uma simulação. O eixo horizontal representa a iluminação e o eixo vertical representa a taxa de efetividade. Permite identificar se existe uma correlação entre as variáveis, ou seja, se a iluminação afeta a taxa de efetividade."""

# Gráfico de dispersão (scatter plot) para visualizar a relação entre qualidade da imagem e taxa de efetividade
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Qualidade_Imagem', y='Taxa_Efetividade', data=resultados_df, alpha=0.5)
plt.xlabel('Qualidade da Imagem')
plt.ylabel('Taxa de Efetividade')
plt.title('Relação entre Qualidade da Imagem e Taxa de Efetividade')
plt.grid(True)
plt.show()

"""Semelhante ao gráfico anterior, mas agora analisando a qualidade da imagem no eixo horizontal. Permite analisar se a qualidade da imagem impacta na taxa de efetividade."""

# Histograma para visualizar a distribuição da taxa de efetividade
plt.figure(figsize=(10, 6))
sns.histplot(resultados_df['Taxa_Efetividade'], bins=15, kde=True)
plt.xlabel('Taxa de Efetividade')
plt.ylabel('Frequência')
plt.title('Distribuição da Taxa de Efetividade')
plt.grid(True)
plt.show()

"""A curva KDE suaviza o histograma e ajuda a identificar a forma geral da distribuição da taxa de efetividade."""