# Bibliotecas para leitura e processamentos dos dados
import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy.optimize import minimize, curve_fit
from scipy.stats import chi2

# Bibliotecas para o Projeto RMN
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import r2_score


def RegressaoSDR (Dataframe_SDR,
                  N_t2 = 'T2 Ponderado Log', N_porosidade = 'Porosidade RMN',
                  N_permeabilidade = 'Permeabilidade Gas'):

    """
    A regressão dos coeficientes da modelagem SDR proposta por Kenyon et al (1988).

    Args:
        Dataframe_SDR (pandas.DataFrame): Dataframe com os dados necessários para modelagem

    Returns:
        Retorna a regressão realizada (reg_ols_log), os coeficientes da regressão (coeficientes), um dataframe com os dados previstos concatenado
        com o DataFrame informado, e o erro sigma.
    """
    # Regressão via OLS
    t2 = Dataframe_SDR[N_t2]
    phi = Dataframe_SDR[N_porosidade]
    permeabilidade = Dataframe_SDR[N_permeabilidade]
    dados_calculo = pd.DataFrame({'Log k': np.log(permeabilidade),
                                'Log φ': np.log(phi),
                                'Log T2': np.log(t2)})
    dados_calculo = sm.add_constant(dados_calculo)
    atributos = dados_calculo[['const', 'Log φ', 'Log T2']]
    rotulos = dados_calculo[['Log k']]
    reg_ols_log = sm.OLS(rotulos, atributos, hasconst=True).fit()

    # Obtenção dos coeficientes da Regressão
    coeficientes = pd.DataFrame({
        'Coeficiente': ['a', 'b', 'c', 'R2'],
        'Valor': [np.exp(reg_ols_log.params[0]),
                  reg_ols_log.params[1],
                  reg_ols_log.params[2],
                  reg_ols_log.rsquared]}).set_index('Coeficiente')

    # Cálculo da Previsão com base nos coeficientes obtidos
    a = coeficientes['Valor']['a']
    b = coeficientes['Valor']['b']
    c = coeficientes['Valor']['c']
    k = (a*(phi**b)*(t2**c))
    dados = pd.DataFrame({'Permeabilidade Prevista SDR': k})

    #Erro Sigma
    k_p = np.log10(dados['Permeabilidade Prevista SDR'])
    k_g = np.log10(permeabilidade)
    N = len(k_p)
    soma = np.sum((k_p-k_g)**2)
    raiz = np.sqrt(soma/N)
    sigma = 10**(raiz)

    return reg_ols_log, coeficientes, pd.concat([Dataframe_SDR, dados], axis = 1), sigma

##################################################################################  Próxima Função  ##################################################################################

def RegressaoTimurCoatesOriginal(Dataframe_Coates,
                                     BVI='BVI', FFI='FFI', N_porosidade='Porosidade RMN',
                                     N_Permeabilidade='Permeabilidade Gas'):
    """
    Regressão dos coeficientes da modelagem Timur-Coates com a forma log(k) = ab*log(φ_NMR/C) + b*log(FFI/BVI).

    Args:
        Dataframe_Coates (pandas.DataFrame): DataFrame com os dados necessários para modelagem.
        BVI (str): Nome da coluna BVI.
        FFI (str): Nome da coluna FFI.
        N_porosidade (str): Nome da coluna de Porosidade RMN.
        N_Permeabilidade (str): Nome da coluna de Permeabilidade.
        C (float): Constante C utilizada na equação.

    Returns:
        reg_ols_log: Objeto da regressão.
        coeficientes: DataFrame com os coeficientes estimados.
        DataFrame com dados previstos.
        sigma: erro padrão.
    """
    # Preparo das variáveis
    FFIBVI = Dataframe_Coates[FFI] / Dataframe_Coates[BVI]
    permeabilidade = Dataframe_Coates[N_Permeabilidade]
    porosidade = Dataframe_Coates[N_porosidade]

    X_c = np.power(permeabilidade/porosidade, 0.25)
    Y_c = np.power(FFIBVI, 0.5)
    X_matrix = sm.add_constant(X_c)
    modelo_C = sm.OLS(Y_c, X_c).fit()

    C = modelo_C.params[0]
    phi = Dataframe_Coates[N_porosidade] / C
    # Construção das variáveis para regressão
    X1 = np.log(phi)
    X2 = np.log(FFIBVI)
    Y = np.log(permeabilidade)

    # DataFrame com as variáveis
    dados_calculo = pd.DataFrame({'Log k': Y,
                                  'Log (phi/C)': X1,
                                  'Log (FFI/BVI)': X2})

    # Atributos e rótulo
    atributos = dados_calculo[['Log (phi/C)', 'Log (FFI/BVI)']]
    rotulo = dados_calculo['Log k']

    # Regressão OLS
    reg_ols_log = sm.OLS(rotulo, atributos).fit()

    # Coeficientes da regressão: const, ab, b
    ab = reg_ols_log.params[0]
    b = reg_ols_log.params[1]

    # Cálculo de 'a'
    if b != 0:
        a = ab / b
    else:
        a = np.nan  # evitar divisão por zero

    # DataFrame com coeficientes
    coeficientes = pd.DataFrame({
        'Coeficiente': ['ab', 'a', 'b', 'C', 'R2'],
        'Valor': [ab, a, b, C, reg_ols_log.rsquared]
    }).set_index('Coeficiente')

    # Previsão da permeabilidade usando o modelo
    k_previsto = ((phi ** (a)) * (FFIBVI)) ** b
    dados_previstos = pd.DataFrame({'Permeabilidade Prevista Timur-Coates Original': k_previsto})

    # Cálculo do erro sigma
    k_p = np.log10(k_previsto)
    k_g = np.log10(permeabilidade)
    N = len(k_p)
    soma = np.sum((k_p - k_g) ** 2)
    raiz = np.sqrt(soma / N)
    sigma = 10 ** raiz

    return reg_ols_log, coeficientes, pd.concat([Dataframe_Coates, dados_previstos], axis=1), sigma

##################################################################################  Próxima Função  ##################################################################################

def RegressaoTimurCoates (Dataframe_Coates,
                     BVI = 'BVI', FFI = 'FFI', N_porosidade = 'Porosidade RMN',
                     N_Permeabilidade = 'Permeabilidade Gas'):

    """
    A regressão dos coeficientes da modelagem Coates proposta por Coates et al (1999).

    Args:
        Dataframe_Coates (pandas.DataFrame): Dataframe com os dados necessários para modelagem.

    Returns:
        Retorna a regressão realizada (reg_ols_log), os coeficientes da regressão (coeficientes), um dataframe com os dados previstos concatenado
        com o DataFrame informado, e o erro sigma.
    """
    # Regressão via OLS
    FFIBVI = Dataframe_Coates[FFI]/Dataframe_Coates[BVI]
    phi = Dataframe_Coates[N_porosidade]
    permeabilidade = Dataframe_Coates[N_Permeabilidade]
    dados_calculo = pd.DataFrame({'Log k': np.log(permeabilidade),
                                'Log φ': np.log(phi),
                                'Log FFI/BVI': np.log(FFIBVI)})
    dados_calculo = sm.add_constant(dados_calculo)
    atributos = dados_calculo[['const', 'Log φ', 'Log FFI/BVI']]
    rotulos = dados_calculo[['Log k']]
    reg_ols_log = sm.OLS(rotulos, atributos, hasconst=True).fit()

    # Obtenção dos coeficientes da Regressão
    coeficientes = pd.DataFrame({
        'Coeficiente': ['a', 'b', 'c', 'R2'],
        'Valor': [np.exp(reg_ols_log.params[0]),
                  reg_ols_log.params[1],
                  reg_ols_log.params[2],

                  reg_ols_log.rsquared]}).set_index('Coeficiente')

    # Cálculo da Previsão com base nos coeficientes obtidos
    a = coeficientes['Valor']['a']
    b = coeficientes['Valor']['b']
    c = coeficientes['Valor']['c']
    k = (a*(phi**b)*(FFIBVI**c))
    dados = pd.DataFrame({'Permeabilidade Prevista Timur-Coates': k})

    #Erro Sigma
    k_p = np.log10(dados['Permeabilidade Prevista Timur-Coates'])
    k_g = np.log10(permeabilidade)
    N = len(k_p)
    soma = np.sum((k_p-k_g)**2)
    raiz = np.sqrt(soma/N)
    sigma = 10**(raiz)

    return reg_ols_log, coeficientes, pd.concat([Dataframe_Coates, dados], axis = 1), sigma

##################################################################################  Próxima Função  ##################################################################################

def RegressaoHan (Dataframe_Han,
                           Amostra = 'Amostra', S1 = 'S1Han', S2 = 'S2Han', S3 = 'S3Han', S4 = 'S4Han',
                           N_porosidade = 'Porosidade RMN',
                           N_permeabilidade = 'Permeabilidade Gas'):

    """
    A regressão dos coeficientes da modelagem Coates proposta por Han et al (2018).

    Args:
        Dataframe_Han (pandas.DataFrame): Dataframe com os dados necessários para modelagem.

    Returns:
        Retorna a regressão realizada (reg_novo), os coeficientes da regressão (coeficientes_novo), um dataframe com os dados previstos concatenado
        com o DataFrame informado, e o erro sigma.
    """
    # Regressão via OLS
    dados_calculo_log = pd.DataFrame({
    'Log k': np.log(Dataframe_Han[N_permeabilidade]),
    'Log φ': np.log(Dataframe_Han[N_porosidade]),
    'S1 log': (-1)*(np.log(Dataframe_Han[S1])),
    'S2 log': (-1)*(np.log(Dataframe_Han[S2])),
    'S3 log': np.log(Dataframe_Han[S3]),
    'S4 log': np.log(Dataframe_Han[S4])})
    dados_calculo = sm.add_constant(dados_calculo_log)

    atributos = dados_calculo[['const', 'Log φ', 'S3 log', 'S4 log', 'S1 log', 'S2 log']]
    rotulos = dados_calculo['Log k']
    reg_ols_log = sm.OLS(rotulos, atributos, hasconst=True, missing = 'drop').fit()

    # Obtenção dos coeficientes da Regressão
    coeficientes_novo = pd.DataFrame({
          'Coeficiente': ['a', 'b', 'c', 'd', 'e', 'f', 'R2'],
          'Valor': [np.exp(reg_ols_log.params[0]),
                    reg_ols_log.params[1],
                    reg_ols_log.params[2],
                    reg_ols_log.params[3],
                    reg_ols_log.params[4],
                    reg_ols_log.params[5],
                    reg_ols_log.rsquared]
          }).set_index('Coeficiente')

    # Cálculo da Previsão com base nos coeficientes obtidos
    a = coeficientes_novo['Valor']['a']
    b = coeficientes_novo['Valor']['b']
    c = coeficientes_novo['Valor']['c']
    d = coeficientes_novo['Valor']['d']
    e = coeficientes_novo['Valor']['e']
    f = coeficientes_novo['Valor']['f']
    phi = Dataframe_Han[N_porosidade]
    s1 = Dataframe_Han[S1]
    s2 = Dataframe_Han[S2]
    s3 = Dataframe_Han[S3]
    s4 = Dataframe_Han[S4]
    k = a*(phi**b)*(s3**c)*(s4**d)/((s1**e)*(s2**f))
    dados = pd.DataFrame({'Permeabilidade Prevista Han': k})

    #Erro Sigma
    k_p = np.log10(dados['Permeabilidade Prevista Han'])
    k_g = np.log10(Dataframe_Han[N_permeabilidade])
    N = len(k_p)
    soma = np.sum((k_p-k_g)**2)
    raiz = np.sqrt(soma/N)
    sigma = 10**(raiz)



    return reg_ols_log, coeficientes_novo, pd.concat([Dataframe_Han, dados], axis = 1), sigma

##################################################################################  Próxima Função  ##################################################################################

def RegressaoGe (Dataframe_Ge,
                 Amostra = 'Amostra', S1 = 'S1Ge', S3 = 'S3Ge', S4 = 'S4Ge',
                 N_porosidade = 'Porosidade RMN', N_permeabilidade = 'Permeabilidade Gas'):

    """
    A regressão dos coeficientes da modelagem Coates proposta por Ge et al (2017).

    Args:
        Dataframe_Ge (pandas.DataFrame): Dataframe com os dados necessários para modelagem.

    Returns:
        Retorna a regressão realizada (reg_novo), os coeficientes da regressão (coeficientes_novo), um dataframe com os dados previstos concatenado
        com o DataFrame informado, e o erro sigma.
    """
    # Regressão via OLS
    dados_calculo_log = pd.DataFrame({
    'Log k': np.log(Dataframe_Ge[N_permeabilidade]),
    'Log φ': np.log(Dataframe_Ge[N_porosidade]),
    'S1 log': (-1)*(np.log(Dataframe_Ge[S1])),
    'S3Ge': Dataframe_Ge[S3],
    'S4Ge': Dataframe_Ge[S4]})

    # Função para calcular a soma dos quadrados dos resíduos
    def residuals(params, df):
      ln_a, b, c, d, e = params
      ln_P3c_P4d = np.log(df['S3Ge']**c + df['S4Ge']**d)
      predicted_ln_k = ln_a + b * df['Log φ'] + ln_P3c_P4d - e * df['S1 log']
      return np.sum((df['Log k'] - predicted_ln_k) ** 2)

    # Valores iniciais para os parâmetros
    initial_params = [0, 0, 0, 0, 0]

    # Minimização da função de resíduos
    result = minimize(residuals, initial_params, args=(dados_calculo_log), method='BFGS')

    # Extração dos parâmetros ajustados
    ln_a, b, c, d, e = result.x
    a = np.exp(ln_a)

    # Cálculo de ln(P3^c + P4^d) com os coeficientes ajustados
    dados_calculo_log['Log S3c_S4d'] = np.log(dados_calculo_log['S3Ge']**c + dados_calculo_log['S4Ge']**d)

    # Definindo as variáveis independentes e a variável dependente
    X = dados_calculo_log[['Log φ', 'S1 log', 'Log S3c_S4d']]
    dados_calculo = sm.add_constant(X)  # Adiciona uma constante (intercepto)
    atributos = dados_calculo[['const', 'Log φ', 'Log S3c_S4d', 'S1 log']]
    rotulos = dados_calculo_log['Log k']

    # Ajustando o modelo de regressão
    reg_ols_log = sm.OLS(rotulos, atributos, hasconst=True, missing = 'drop').fit()

    # Obtenção dos coeficientes da Regressão
    coeficientes_novo = pd.DataFrame({
          'Coeficiente': ['a', 'b', 'c', 'd', 'e', 'R2'],
          'Valor': [np.exp(reg_ols_log.params[0]),
                    reg_ols_log.params[1],
                    c,
                    d,
                    reg_ols_log.params[2],
                    reg_ols_log.rsquared]
          }).set_index('Coeficiente')

    # Cálculo da Previsão com base nos coeficientes obtidos
    a = coeficientes_novo['Valor']['a']
    b = coeficientes_novo['Valor']['b']
    e = coeficientes_novo['Valor']['e']
    phi = Dataframe_Ge[N_porosidade]
    s1Ge = Dataframe_Ge[S1]
    s3Ge = Dataframe_Ge[S3]
    s4Ge = Dataframe_Ge[S4]
    k = a*(phi**b)*((s3Ge**c)+(s4Ge**d)/(s1Ge**e))
    dados = pd.DataFrame({'Permeabilidade Prevista Ge': k})


    #Erro Sigma
    k_p = np.log10(dados['Permeabilidade Prevista Ge'])
    k_g = np.log10(Dataframe_Ge[N_permeabilidade])
    N = len(k_p)
    soma = np.sum((k_p-k_g)**2)
    raiz = np.sqrt(soma/N)
    sigma = 10**(raiz)



    return reg_ols_log, coeficientes_novo, pd.concat([Dataframe_Ge, dados], axis = 1), sigma

##################################################################################  Próxima Função  ##################################################################################

def RegressaoRios(dados_treino, dados_teste, N_permeabilidade = 'Permeabilidade Gas', Log = True, N_componentes = 6,
                  C_distribuicao = ['T2 0.01',  'T2 0.011',  'T2 0.012',  'T2 0.014',  'T2 0.015',  'T2 0.017',  'T2 0.019',  'T2 0.021',  'T2 0.024',
           'T2 0.027',  'T2 0.03',  'T2 0.033',  'T2 0.037',  'T2 0.041',  'T2 0.046',  'T2 0.051',  'T2 0.057',  'T2 0.064',
           'T2 0.071',  'T2 0.079',  'T2 0.088',  'T2 0.098',  'T2 0.109',  'T2 0.122',  'T2 0.136',  'T2 0.152',  'T2 0.169',
           'T2 0.189',  'T2 0.21',  'T2 0.234',  'T2 0.261',  'T2 0.291',  'T2 0.325',  'T2 0.362',  'T2 0.404',  'T2 0.45',
           'T2 0.502',  'T2 0.56',  'T2 0.624',  'T2 0.696',  'T2 0.776',  'T2 0.865',  'T2 0.964',  'T2 1.075',  'T2 1.199',
           'T2 1.337',  'T2 1.49',  'T2 1.661',  'T2 1.852',  'T2 2.065',  'T2 2.303',  'T2 2.567',  'T2 2.862',  'T2 3.191',
           'T2 3.558',  'T2 3.967',  'T2 4.423',  'T2 4.931',  'T2 5.497',  'T2 6.129',  'T2 6.834',  'T2 7.619',  'T2 8.494',
           'T2 9.471',  'T2 10.559',  'T2 11.772',  'T2 13.125',  'T2 14.634',  'T2 16.315',  'T2 18.19',  'T2 20.281',  'T2 22.612',
           'T2 25.21',  'T2 28.107',  'T2 31.337',  'T2 34.939',  'T2 38.954',  'T2 43.431',  'T2 48.422',  'T2 53.986',  'T2 60.19',
           'T2 67.108',  'T2 74.82',  'T2 83.418',  'T2 93.004',  'T2 103.693',  'T2 115.609',  'T2 128.895',  'T2 143.708',  'T2 160.223',
           'T2 178.636',  'T2 199.165',  'T2 222.053',  'T2 247.572',  'T2 276.023',  'T2 307.744',  'T2 343.11',  'T2 382.54',  'T2 426.502',
           'T2 475.516',  'T2 530.163',  'T2 591.09',  'T2 659.019',  'T2 734.754',  'T2 819.192',  'T2 913.335',  'T2 1018.296',  'T2 1135.32',
           'T2 1265.792',  'T2 1411.258',  'T2 1573.441',  'T2 1754.262',  'T2 1955.864',  'T2 2180.633',  'T2 2431.234',  'T2 2710.634',  'T2 3022.143',
           'T2 3369.45',  'T2 3756.671',  'T2 4188.391',  'T2 4669.725',  'T2 5206.375',  'T2 5804.697',  'T2 6471.778',  'T2 7215.521',  'T2 8044.736',
           'T2 8969.245',  'T2 10000.0']):


    """
    A regressão dos coeficientes da modelagem Coates proposta por Rios et al (2011).

    Args:
        dados_treino (pandas.DataFrame): Dataframe com os dados necessários para o treinamento do modelo.
        dados_teste (pandas.DataFrame): Dataframe com os dados necessários avaliação do modelo.
    Returns:
        Retorna a regressão realizada (reg_novo), os coeficientes da regressão (coeficientes_novo), um dataframe com os dados previstos concatenado
        com o DataFrame informado, e o erro sigma.
    """

    if Log == True:
      X_treino = dados_treino[C_distribuicao]
      y_treino = np.log10(dados_treino[N_permeabilidade]*1000)

      X_teste = dados_teste[C_distribuicao]
      y_teste = np.log10(dados_teste[N_permeabilidade]*1000)

      plsr = PLSRegression(n_components=N_componentes)
      plsr.fit(X_treino, y_treino)

      y_pred_treino = plsr.predict(X_treino)
      y_pred_teste = plsr.predict(X_teste)

      dados_treino['Permeabilidade Prevista Rios'] = (10**y_pred_treino)/1000
      dados_teste['Permeabilidade Prevista Rios'] = (10**y_pred_teste)/1000

    else:

      X_treino = dados_treino[C_distribuicao]
      y_treino = dados_treino[N_permeabilidade]

      X_teste = dados_teste[C_distribuicao]
      y_teste = dados_teste[N_permeabilidade]

      plsr = PLSRegression(n_components=N_componentes)
      plsr.fit(X_treino, y_treino)

      y_pred_treino = plsr.predict(X_treino)
      y_pred_teste = plsr.predict(X_teste)

      dados_treino['Permeabilidade Prevista Rios'] = y_pred_treino
      dados_teste['Permeabilidade Prevista Rios'] = y_pred_teste


      #Erro Sigma
    k_p_treino = np.log10(dados_treino['Permeabilidade Prevista Rios'])
    k_g_treino = np.log10(dados_treino[N_permeabilidade])
    N = len(k_p_treino)
    soma = np.sum((k_p_treino-k_g_treino)**2)
    raiz = np.sqrt(soma/N)
    sigma_treino = 10**(raiz)

    k_p_teste = np.log10(dados_teste['Permeabilidade Prevista Rios'])
    k_g_teste = np.log10(dados_teste[N_permeabilidade])
    N = len(k_p_treino)
    soma = np.sum((k_p_teste-k_g_teste)**2)
    raiz = np.sqrt(soma/N)
    sigma_teste = 10**(raiz)




    return plsr, dados_treino[['Amostra', N_permeabilidade,
                               'Permeabilidade Prevista Rios']], dados_teste[['Amostra',
                                N_permeabilidade,
                                'Permeabilidade Prevista Rios']], sigma_treino, sigma_teste

##################################################################################  Próxima Função  ##################################################################################

def RegressaoFZI(Dados, Modelos = ['SDR'], N_Cluster = 'Litofacies'):
    
    """
    A regressão FZI.

    Args:
        dados (pandas.DataFrame): Dataframe com os dados necessários para modelagem.
        Modelos (list): Lista com os modelos utilizados para obter o FZI.
    Returns:
        Retorna a regressão FZI para cada litofácie.
    """

    lito = Dados[N_Cluster].unique()
    coef = []

    for i in np.arange(len(lito)):
        for j in np.arange(len(Modelos)):
            df_dados = Dados.loc[Dados[N_Cluster] == Dados[N_Cluster].unique()[i]].reset_index().drop('index', axis = 1)
            rqi = df_dados['RQI_' + Modelos[j]]

            if Modelos[j] == "Gas":
                phi = df_dados['Phi_z_Gas']
    
            else:
                phi = df_dados['Phi_z_RMN']
    
            dados_calculo = pd.DataFrame({'Phi': phi,
                                        'RQI': rqi})
    
            dados_calculo['const'] = 1
    
            dados_calculo = sm.add_constant(dados_calculo)
            atributos = dados_calculo[['const', 'Phi']]
            rotulos = dados_calculo[['RQI']]
            reg_ols_log = sm.OLS(rotulos, atributos, hasconst=True).fit()
            coef.append([Dados[N_Cluster].unique()[i] + '_' + Modelos[j], reg_ols_log.params[0], reg_ols_log.params[1], reg_ols_log.params[0]+reg_ols_log.params[1]])
    nome = 'Cluster: '+ N_Cluster
    c = pd.DataFrame(coef).rename(columns={0: nome, 1:'b', 2:'a', 3:'FZI'})
    df = pd.concat([Dados, c], axis = 1)
    return df

##################################################################################  Próxima Função  ##################################################################################

def RegressaoComponentesT2 (Dados, n = 0, P0 = (0.8, 0.01), Params_Init = [0.8, 0.001, 0.1, 0.01, 0.1, 0.1],
                            Chute = False, N_chute = 0,
                            T_relaxacao = "Tempo Relaxacao", A_relaxacao = "Amplitude Relaxacao", Amostra = 'Amostra'):
    """
    A regressão da curva de relaxação para obter as componentes T2 de uma única .

    Args:
        Dados (pandas.DataFrame): Dataframe com os dados necessários para modelagem.
        n (int): Indice da amostra que terá seus componentes avaliados.
        P0 (tuple): Tupla com oa parâmetros do coeficiente T2_nmr.
        Params_Init (list): Lista de parâmetros iniciais de cada componente T2 OBS: Caso apareça qualquer mensagem de erro ou 
        'O ajuste do modelo não é adequado. Considere revisar os parâmetros iniciais.' mudar esses valores até que a única saida seja
        'O ajuste do modelo é adequado.'
    Returns:
        Retorna um DataFrame com todos os coeficientes T2 e o erro R^2.
    """

    def exponential_decay(t, a, b):
        return a * np.exp(-b * t)

    # Função do modelo exponencial multi-termo
    def multi_exponential_decay(t, params):
        a, b, c, d, g, h = params
        return a * np.exp(-b * t) + c * np.exp(-d * t) + g * np.exp(-h * t)
    
    # Função de erro (MSE)
    def mse(params, t, y):
        y_pred = multi_exponential_decay(t, params)
        return np.mean((y - y_pred)**2)
    
    
    time = np.array(Dados[T_relaxacao][n])  # Coloque seus valores de tempo aqui
    A_t = np.array(Dados[A_relaxacao][n])/Dados[A_relaxacao][n].max()  # Coloque seus valores de A(t) aqui
    
    # Realizar o ajuste usando curve_fit
    p0 = P0  # Valores iniciais para a e b
    params, cov = curve_fit(exponential_decay, time, A_t, p0=p0)
    
    # Parâmetros ajustados
    anmr_fit, bnmr_fit = params
    
    if Chute == True:
      chute = [[0.8, 0.001, 0.001, 0.001, 0.001, 0.001],
               [0.8, 0.001, 0.01, 0.01, 0.1, 0.1],
               [0.825, 0.001, 0.01, 0.01, 0.01, 0.01],
               [0.5, 0.01, 0.1, 0.001, 0.1, 0.01],
               [1, 0.01, 0.01, 0.01, 0.1, 0.001],
               [0.5, 0.0001, 0.001, 0.01, 0.1, 0.1],
               [0.5, 0.0001, 0.0001, 0.01, 0.01, 0.01]]
      params_init = chute[N_chute]

    else:
      # Chute inicial para os parâmetros (a, b, c, d, g, h)
      params_init = Params_Init
    
    # Minimização do erro usando minimize (Método dos mínimos quadrados)
    result = minimize(mse, params_init, args=(time, A_t))
    
    # Parâmetros ajustados
    a_fit, b_fit, c_fit, d_fit, g_fit, h_fit = result.x
    
    # Função para calcular as frequências esperadas
    def expected_frequencies(params, t):
        y_pred = multi_exponential_decay(t, params)
        return y_pred
    
    # Frequências esperadas
    expected_values = expected_frequencies(params_init, time)
    
    # Cálculo do qui-quadrado
    chi_square_statistic = np.sum((A_t - expected_values)**2 / expected_values)
    
    # Número de graus de liberdade
    degrees_of_freedom = len(A_t) - len(params_init)
    
    # Valor crítico para alpha = 0.05 (95% de confiança) e graus de liberdade
    critical_value = chi2.ppf(0.95, degrees_of_freedom)
    
    # Comparação com o valor crítico
    if chi_square_statistic <= critical_value:
          print('O ajuste do modelo é adequado.')
    else:
          print('O ajuste do modelo não é adequado. Considere revisar os parâmetros iniciais.')
    
    coef = pd.DataFrame({'Amostra': Dados[Amostra][n],
                           A_relaxacao: [A_t],
                           T_relaxacao: [time],
                           'A_NMR': [anmr_fit],
                           'T2_NMR': [1/bnmr_fit]})
    coef['A1'] = [a_fit]
    coef['T21'] = [1/b_fit]
    coef['A2'] = [c_fit]
    coef['T22'] = [1/d_fit]
    coef['A3'] = [g_fit]
    coef['T23'] = [1/h_fit]
    
    ft = coef['A_NMR'][0] * np.exp((-1/coef['T2_NMR'][0]) * time)
    f1 = coef['A1'][0] * np.exp((-1/coef['T21'][0]) * time)
    f2 = coef['A2'][0] * np.exp((-1/coef['T22'][0]) * time)
    f3 = coef['A3'][0] * np.exp((-1/coef['T23'][0]) * time)
    
    r2_ft = r2_score(A_t, ft)
    r2_fc = r2_score(A_t, f1+f2+f3)
    
    coef['R2_FT'] = r2_ft
    coef['R2_FC'] = r2_fc

    return coef

##################################################################################  Próxima Função  ##################################################################################

def AjusteComponentesT2Seevers(Dados, n=0, P0=(0.5, 0.5, 1, 1)):
    """
    Ajusta a função f(t) = A1 * exp(-t/3) + A2 * exp(-t/T2S)
    para os dados de relaxação no índice n de um DataFrame.

    Inclui normalização por A_t[0] e teste de qui-quadrado.

    Parâmetros:
        Dados: DataFrame com colunas 'Tempo Relaxacao', 'Amplitude Relaxacao', 'Amostra'
        n: índice da amostra a ser ajustada
        P0: chute inicial para [A1, A2, T2S]

    Retorna:
        coef: DataFrame com parâmetros ajustados, R² e resultado do teste qui-quadrado
    """

    # Extrai e normaliza os dados

    time = np.array(Dados['Tempo Relaxacao'][n])
    A_t = np.array(Dados['Picos Relaxacao Fitting'][n])
    A_t = A_t / A_t[0]  # Normalização pelo valor inicial

    # Define o modelo
    def modelo(t, A1, A2, T2L, T2S):
        return A1 * np.exp(-t/T2L) + A2 * np.exp(-t/T2S)

    # Ajuste de parâmetros
    params, _ = curve_fit(modelo, time, A_t, p0=P0, maxfev=10000)
    A1_fit, A2_fit, T2L_fit, T2S_fit = params

    # Valores previstos
    A_pred = modelo(time, A1_fit, A2_fit, T2L_fit, T2S_fit)

    # R²
    r2 = r2_score(A_t, A_pred)

    # Qui-quadrado
    chi_square_statistic = np.sum((A_t - A_pred)**2 / A_pred)
    degrees_of_freedom = len(A_t) - len(params)
    critical_value = chi2.ppf(0.95, degrees_of_freedom)

    if chi_square_statistic <= critical_value:
        ajuste_aprovado = True
        mensagem = "O ajuste do modelo é adequado."
    else:
        ajuste_aprovado = False
        mensagem = "O ajuste do modelo não é adequado. Considere revisar os parâmetros iniciais."

    print(mensagem)

    # DataFrame de resultados
    coef = pd.DataFrame({
        'Amostra': [Dados['Amostra'][n]],
        'A1': [A1_fit],
        'A2': [A2_fit],
        'T2S': [T2S_fit],
        'T2L': [T2L_fit],
        'R2': [r2],
        'Qui2': [chi_square_statistic],
        'Qui2_Critico': [critical_value],
        'Ajuste_Adequado': [mensagem],
        'Fração Bulk': [A1_fit * np.exp(-time/T2L_fit)],
        'Fração Surface': [A2_fit * np.exp(-time/T2S_fit)],
        'Aplitude ajustada': [A1_fit * np.exp(-time/T2L_fit) + A2_fit * np.exp(-time/T2S_fit)]
    })

    return coef

##################################################################################  Próxima Função  ##################################################################################

def RegressaoArns (Dados_Arns, N_Permeabilidade = 'Permeabilidade Gas', N_T2 = 'T2 Ponderado Log',
                  N_FatorFormacao = 'Fator de Formacao'):
  permeabilidade = Dados_Arns[N_Permeabilidade]
  t2 = Dados_Arns[N_T2]
  ff = Dados_Arns[N_FatorFormacao]
  dados_calculo = pd.DataFrame({'Log T2': np.log(t2),
                                'Log FF': np.log(ff),
                                'Log k': np.log(permeabilidade)})
  dados_calculo = sm.add_constant(dados_calculo)
  atributos = dados_calculo[['const', 'Log T2', 'Log FF']]
  rotulos = dados_calculo[['Log k']]
  reg_Arns = sm.OLS(rotulos, atributos, hasconst=True).fit()

  # Obtenção dos coeficientes da Regressão
  coef_Arns = pd.DataFrame({
              'Coeficiente': ['a', 'b', 'c', 'R2'],
              'Valor': [np.exp(reg_Arns.params[0]),
                        reg_Arns.params[1],
                        reg_Arns.params[2],
                        reg_Arns.rsquared]}).set_index('Coeficiente')

  # Cálculo da Previsão com base nos coeficientes obtidos
  a = coef_Arns['Valor']['a']
  b = coef_Arns['Valor']['b']
  c = coef_Arns['Valor']['c']
  k = (a*(t2**b)*(ff**c))
  dados = pd.DataFrame({'Permeabilidade Prevista Arns': k})

  #Erro Sigma
  k_p = np.log10(dados['Permeabilidade Prevista Arns'])
  k_g = np.log10(permeabilidade)
  N = len(k_p)
  soma = np.sum((k_p-k_g)**2)
  raiz = np.sqrt(soma/N)
  sigma_Arns = 10**(raiz)

  return reg_Arns, coef_Arns, pd.concat([Dados_Arns, dados], axis = 1), sigma_Arns

##################################################################################  Próxima Função  ##################################################################################

def RegressaoParchekhari (Dataframe_Parchekhari, Params = [0, 0, 0, 0, 0, 0],
                          N_Permeabilidade = 'Permeabilidade Gas', N_T2 = 'T2 Ponderado Log', N_Porosidade = 'Fator de Formacao',
                          N_A1 = 'Amp1', N_A2 = 'Amp2', N_A3 = 'Amp3'):
    # Regressão via OLS
    dados_calculo_log = pd.DataFrame({
    'Log k': np.log10(Dataframe_Parchekhari[N_Permeabilidade]),
    'Log φ': np.log10(Dataframe_Parchekhari[N_Porosidade]),
    'Log T2': np.log10(Dataframe_Parchekhari[N_T2]),
    'A1': Dataframe_Parchekhari[N_A1],
    'A2': Dataframe_Parchekhari[N_A2],
    'A3': Dataframe_Parchekhari[N_A3]})

    # Função para calcular a soma dos quadrados dos resíduos
    def residuals(params, df):
      ln_a, b, c, d, e, f = params
      log_A1_A2_A3 = np.log(df['A1']**b + df['A2']**c + df['A3']**d)
      predicted_ln_k = ln_a + log_A1_A2_A3 + e * df['Log T2'] + f * df['Log φ']
      return np.sum((df['Log k'] - predicted_ln_k) ** 2)

    # Valores iniciais para os parâmetros
    initial_params = Params

    # Minimização da função de resíduos
    result = minimize(residuals, initial_params, args=(dados_calculo_log), method='BFGS')

    # Extração dos parâmetros ajustados
    ln_a, b, c, d, e, f = result.x
    a = np.exp(ln_a)

    # Cálculo de ln(P3^c + P4^d) com os coeficientes ajustados
    dados_calculo_log['Log A1_A2_A3'] = np.log(dados_calculo_log['A1']**b + dados_calculo_log['A2']**c + dados_calculo_log['A3']**d)

    # Definindo as variáveis independentes e a variável dependente
    X = dados_calculo_log[['Log A1_A2_A3', 'Log T2', 'Log φ']]
    dados_calculo = sm.add_constant(X)  # Adiciona uma constante (intercepto)
    atributos = dados_calculo[['const', 'Log A1_A2_A3', 'Log T2', 'Log φ']]
    rotulos = dados_calculo_log['Log k']

    # Ajustando o modelo de regressão
    reg_novo = sm.OLS(rotulos, atributos, hasconst=True, missing = 'drop').fit()

    # Obtenção dos coeficientes da Regressão
    coeficientes_novo = pd.DataFrame({
                  'Coeficiente': ['a', 'b', 'c', 'd', 'e', 'f', 'R2'],
                  'Valor': [np.exp(reg_novo.params[0]),
                            b, c, d,
                            reg_novo.params[1],
                            reg_novo.params[2],
                            reg_novo.rsquared]}).set_index('Coeficiente')

    # Cálculo da Previsão com base nos coeficientes obtidos
    ca = coeficientes_novo['Valor']['a']
    cb = coeficientes_novo['Valor']['b']
    cc = coeficientes_novo['Valor']['c']
    cd = coeficientes_novo['Valor']['d']
    ce = coeficientes_novo['Valor']['e']
    cf = coeficientes_novo['Valor']['f']
    phi = Dataframe_Parchekhari[N_Porosidade]
    t2 = Dataframe_Parchekhari[N_T2]
    A1Par = Dataframe_Parchekhari[N_A1]
    A2Par = Dataframe_Parchekhari[N_A1]
    A3Par = Dataframe_Parchekhari[N_A1]
    k = ca*((A1Par**cb)+(A2Par**cc)+(A3Par**cd))*(t2**ce)*(phi**cf)
    dados = pd.DataFrame({'Permeabilidade Prevista Parchekhari': k})


    #Erro Sigma
    k_p = np.log10(dados['Permeabilidade Prevista Parchekhari'])
    k_g = np.log10(Dataframe_Parchekhari[N_Permeabilidade])
    N = len(k_p)
    soma = np.sum((k_p-k_g)**2)
    raiz = np.sqrt(soma/N)
    sigma = 10**(raiz)


    return reg_novo, coeficientes_novo, pd.concat([Dataframe_Parchekhari, dados], axis = 1), sigma

##################################################################################  Próxima Função  ##################################################################################

def RegressaoSeevers(Dados_Seevers, T2B = 3000, N_T2L = 'T2L', N_FFI = 'FFI', ):
  permeabilidade = Dados_Seevers['Permeabilidade Gas']
  T2L = Dados_Seevers[N_T2L]
  FFI = Dados_Seevers[N_FFI]
  Alfa = FFI * (df_Par['T2L'] * T2B / (T2B - df_Par['T2L'])) ** 2

  dados_calculo = pd.DataFrame({'log Alfa': np.log(Alfa),
                                'Log k': np.log(permeabilidade)})

  dados_calculo = sm.add_constant(dados_calculo)
  atributos = dados_calculo[['const', 'log Alfa']]
  rotulos = dados_calculo[['Log k']]
  reg_ols_log_Seevers = sm.OLS(rotulos, atributos, hasconst=True).fit()

  coeficientes_Seevers = pd.DataFrame({
        'Coeficiente': ['a', 'b', 'R2'],
        'Valor': [np.exp(reg_ols_log_Seevers.params[0]),
                  reg_ols_log_Seevers.params[1],
                  reg_ols_log_Seevers.rsquared]}).set_index('Coeficiente')


  #Cálculo da Previsão com base nos coeficientes obtidos
  a = coeficientes_Seevers['Valor']['a']
  b = coeficientes_Seevers['Valor']['b']
  k = (a*(Alfa**b))
  dados = pd.DataFrame({'Alfa': Alfa,
                        'Permeabilidade Prevista Seevers': k})

  #Erro Sigma
  k_p = np.log10(dados['Permeabilidade Prevista Seevers'])
  k_g = np.log10(permeabilidade)
  N = len(k_p)
  soma = np.sum((k_p-k_g)**2)
  raiz = np.sqrt(soma/N)
  sigma_Seevers = 10**(raiz)

  return reg_ols_log_Seevers, coeficientes_Seevers, pd.concat([Dados_Seevers, dados['Alfa'],
                                                               dados['Permeabilidade Prevista Seevers']], axis = 1), sigma_Seevers

##################################################################################  Próxima Função  ##################################################################################
