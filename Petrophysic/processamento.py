# Bibliotecas mais comuns
import openpyxl

# Bibliotecas para leitura e processamentos dos dados
import pandas as pd
import numpy as np
from numpy import concatenate

# Bibliotecas para Processamento RMN
from sklearn import preprocessing
from sklearn.preprocessing import MaxAbsScaler, OneHotEncoder
from sklearn.metrics import mean_squared_error
from scipy.optimize import nnls, minimize

# Bibliotecas para georeferenciamento
import geopandas as gpd
from shapely.geometry import Point  # Exemplo de geometria


def ObtencaoDadosNiumag(Diretorio_pasta, Arquivo_niumag, Inicio_conversao, Pontos_inversao,
                        Relaxacao = False, Distribuicao = False, T2_niumag = False, Erro = False, Ruido = False,
                        T2_Amplitudes = False, Poco = False,
                        C_amostra = 9, N_poco_i = 4, N_poco_f = 0, N_amp = 3,
                        Inversao_Thikonov = False, Inversao_AntonnTalankin = False, Inversao_ThikonovSuavizada = False,
                        Nom_amplitudes = 'Picos Relaxacao', Nom_Tempo = 'Tempo Relaxacao',
                        T2_minimo = 0.01, T2_maximo = 10000, T2_range = 128, Regularizador = 0.1, Suavizador = 0.1):


    """
      Esta função trata o dados brutos de RMN que o software da Niumag exporta retornando um pandas.DataFrame com as informações
      relevantes e com aspecto mais legível.

      Args:
          Diretorio_pasta (str): O caminho do diretório onde está o arquivo excel exportado pelo software Niumag.
          Arquivo_niumag (str.xlsx): Nome do arquivo que está dentro do diretório e que é um arquivo tipo Excel.
          Inicio_conversao (int): Primeiro ponto de aquisição dos dados no arquivo Excel.
          Pontos_inversao (int): Quantos pontos da inversão foram usados.
          Relaxacao (bool) : Caso o usuário queira retornar os valores do Sinal e Distribuição T2.
          Distribuicao (bool): Caso o usuário queira retornar os valores do Sinal, Fiiting e Tempo de Relaxação.
          T2_niumag (bool): Caso o usuário queira retornar o processamento da média geométrica T2.
          Erro (bool): Caso o usuário deseje o Erro Fiiting.
          T2_Amplitudes (bool): Caso o usuário deseje os picos das amplitudes da distribuição T2.
          Poco (bool): Caso o usuário tenha as informações dos poços contidas no nome das amostras.
          C_amostra (int): Quantidade de caracteres que o nome da amostra tem.
          N_poco_i (int): Identificação do começo do nome do poço.
          N_poco_f (int): Identificação do final do nome do poço.
          N_amp (int): Identificação da quantidade de amplitudes desejadas.
          Inversao_Thikonov (bool): Inversão da curva de relaxação seguindo o método de Tikhonov.
          Inversao_AntonnTalankin (bool): Inversão da curva de relaxação seguindo o método de Antonn Talankin https://github.com/antontalankin/py-4-pp/tree/main/NMR_1D_T2_inversion/Anton.
          Nom_amplitudes (str): Nome da coluna que contém as amplitudes.
          Nom_Tempo (str): Nome da coluna que contém os tempos.
          T2_minimo (float): Valor do tempo mínimo da curva de distribuição.
          T2_maximo (float): Valor do tempo máximo da curva de distribuição.
          T2_range (int): Quantidade de pontos da curva de distribuição.
          Regularizador (float):  Regularizador de Tikhonov.

      Returns:
          pandas.DataFrame: Retorna um DataFrame com dados processados do Excel exportado pelo Software Niumag.

      Exemplos de Uso:
          Caso o usuário tenha um arquivo no estilo exportado pela Niumag que informa a amostra (no nome da amostra informa o nome do poço) essa função retornará os dados
          já específico para o processamento necessário.
      """

    niumag = str(Diretorio_pasta) + str(Arquivo_niumag)                           # Pasta do arquivp
    dados_niumag = pd.read_excel(niumag).drop('File Name', axis=1)                # Dataframe dos Dados da Niumag

    inicio = Inicio_conversao-2                                                   # Linha que se inicia os dados de da inversão
    final = inicio+Pontos_inversao                                                # Linha final da Inversão

    amostras = []
    poc = []
    tempo_relaxacao = []
    amplitude_pico_niumag = []
    amplitude_pico_fitting = []
    amplitude_sinal_niumag = []
    ruido_niumag = []
    tempo_distribuicao = []
    distribuicao_t2 = []
    t2gm_niumag = []
    t2av_niumag = []
    fitting_erro = []
    amplitudes = []

    for i in np.arange(int(len(dados_niumag.columns)/7)):
      df = dados_niumag.T.reset_index().drop('index', axis = 1).T

      nome = dados_niumag.columns[i*7][:C_amostra]
      amostras.append(nome)

      if Poco == True:
        p = nome[N_poco_f:N_poco_i]
        poc.append(p)

      if Relaxacao == True:
        time = np.array(df[i*7][inicio:].reset_index(drop = True)).astype(float)
        pico_niumag = np.array(df[i*7+1][inicio:].reset_index(drop = True)).astype(float)
        pico_fiting = np.array(df[i*7+2][inicio:].reset_index(drop = True)).astype(float)
        tempo_relaxacao.append(time)
        amplitude_pico_niumag.append(pico_niumag)
        amplitude_pico_fitting.append(pico_fiting)

      if Distribuicao == True:
        tempo = np.array(df[i*7+3][inicio:final].reset_index(drop = True)).astype(float)
        dist = np.array(df[i*7+4][inicio:final].reset_index(drop = True)).astype(float)
        tempo_distribuicao.append(tempo)
        distribuicao_t2.append(dist)

      if T2_niumag == True:
        gm = float(df[i*7+2][1][7:-4])
        av = float(df[i*7+2][2][7:-4])
        t2gm_niumag.append(gm)
        t2av_niumag.append(av)

      if Ruido == True:
        sinal = np.array(df[i*7+5][inicio:].reset_index(drop = True)).astype(float)
        ruido = np.array(df[i*7+6][inicio:].reset_index(drop = True)).astype(float)
        amplitude_sinal_niumag.append(sinal)
        ruido_niumag.append(ruido)

      if Erro == True:
        fit_erro = float(df[i*7][0][-5:])
        fitting_erro.append(fit_erro)

      if T2_Amplitudes == True:
        amp = pd.Series(np.array(df[i*7+2][5:inicio-2]).astype(float)).fillna(0).nlargest(N_amp)
        amplitudes.append(amp)

    df = pd.DataFrame({'Amostra': amostras})

    if Poco == True:
      df['Poço'] = poc

    if Relaxacao == True:
      df['Tempo Relaxacao'] = tempo_relaxacao
      df['Picos Relaxacao'] = amplitude_pico_niumag
      df['Picos Relaxacao Fitting'] = amplitude_pico_fitting

    if Ruido == True:
      df['Sinal'] = amplitude_sinal_niumag
      df['Ruido'] = ruido_niumag

    if Distribuicao == True:
      df['Tempo Distribuicao'] = tempo_distribuicao
      df['Distribuicao T2'] = distribuicao_t2
    
    if Inversao_Thikonov == True:
      def InversaoThikonov(Dataframe, N_amostra = 0, N_amplitudes = Nom_amplitudes, N_Tempo = Nom_Tempo,
                           T2_min = T2_minimo, T2_max = T2_maximo, T2_ran = T2_range, Lamb = Regularizador):
        t = Dataframe[N_Tempo][N_amostra].astype(float)
        a = df[N_amplitudes][N_amostra].astype(float)

        # Definindo os limites e a faixa de T2
        t2min = T2_min
        t2max = T2_max
        t2range = T2_ran  # número de bins de T2
        T = np.linspace(np.log10(t2min), np.log10(t2max), t2range)
        T2 = [pow(10, i) for i in T]
        x, y = np.meshgrid(t, T2)

        # Matriz de modelo
        X = np.exp(-x/y).T
        Y = np.array(a)

        # Definindo o parâmetro de regularização
        regularzador = Lamb
        n_variables = X.shape[1]

        # Matriz regularizada (Tikhonov)
        ATA = np.dot(X.T, X) + regularzador * np.eye(n_variables)
        ATb = np.dot(X.T, Y)

        # Resolvendo o sistema regularizado
        amplitude = np.linalg.solve(ATA, ATb)

        # Função de erro (mínimos quadrados regularizados)
        def FuncaoCusto(a):
            return np.linalg.norm(X @ a - Y) ** 2 + regularzador * np.linalg.norm(a) ** 2

        # Resolvendo a minimização com restrição de não negatividade
        res = minimize(FuncaoCusto, np.zeros(n_variables), bounds=[(0, None)] * n_variables)

        # Amplitude final
        amplitude = res.x

        return amplitude, T2

      inversao_t = []
      tempo_t = []

      for i in np.arange(len(df['Amostra'])):
          a_thikonov, t_thikonov = InversaoThikonov(df, N_amostra = i, N_amplitudes = Nom_amplitudes, N_Tempo = Nom_Tempo)
          inversao_t.append(a_thikonov)
          tempo_t.append(t_thikonov)

      df['Inversao Thikonov'] = inversao_t
      df['Distribuicao T2 Thikonov'] = tempo_t
    
    if Inversao_AntonnTalankin == True:
      def Inversao_AntonnTalankin(Dataframe, N_amostra = 0, N_amplitudes = Nom_amplitudes, N_Tempo = Nom_Tempo,
                           T2_min = T2_minimo, T2_max = T2_maximo, T2_ran = T2_range, Lamb = Regularizador):
        t = Dataframe[N_Tempo][N_amostra].astype(float)
        a = df[N_amplitudes][N_amostra].astype(float)
        # inversion
        # set T2 limits & range
        t2min = T2_min
        t2max = T2_max
        t2range = T2_ran # set numper of t2 bins
        T=np.linspace(np.log10(t2min),np.log10(t2max), t2range)
        T2=[pow(10,i) for i in T]
        x,y = np.meshgrid(t, T2)
        X = np.exp(-x/y).T
        Y = np.array(a)
        # set regularization parameter (0.5 as base case)
        lamb = Lamb
        n_variables=X.shape[1]
        A = concatenate([X, np.sqrt(lamb)*np.eye(n_variables)])
        B = concatenate([Y, np.zeros(n_variables)])
        A1=nnls(A,B)
        amplitude=A1[0]
        return amplitude, T2

      amplitude_a = []
      tempo_a = []

      for i in np.arange(len(df['Amostra'])):
        a_antonn, t_antonn = Inversao_AntonnTalankin(df, N_amostra = i, N_amplitudes = Nom_amplitudes, N_Tempo = Nom_Tempo)
        amplitude_a.append(a_antonn)
        tempo_a.append(t_antonn)

      df['Inversao Antonn Talankin'] = amplitude_a
      df['Distribuicao T2 Antonn Talankin'] = tempo_a

    
    if Inversao_ThikonovSuavizada == True:
      def InversaoThikonovSuavizada(Dataframe, N_amostra=0, N_amplitudes='Picos Amplitudes', N_Tempo='Tempo Relaxacao',
                                    T2_minimo = 0.01, T2_maximo = 10000, T2_range = 128, Lamb = Regularizador, Suavizacao = Suavizador):
        t = Dataframe[N_Tempo][N_amostra].astype(float)
        a = Dataframe[N_amplitudes][N_amostra].astype(float)

        # Definindo os limites e a faixa de T2
        t2min = T2_minimo
        t2max = T2_maximo
        t2range = T2_range
        T = np.linspace(np.log10(t2min), np.log10(t2max), t2range)
        T2 = [10 ** i for i in T]
        x, y = np.meshgrid(t, T2)

        # Matriz de modelo
        X = np.exp(-x / y).T
        Y = np.array(a)

        # Parâmetros de regularização
        lambd = Regularizador
        beta = Suavizacao  # Peso da suavização

        # Criando matriz de segunda diferença (suavização)
        n_variables = X.shape[1]
        D = np.zeros((n_variables - 2, n_variables))  # Matriz para diferenças segundas

        for j in range(n_variables - 2):
            D[j, j] = 1
            D[j, j + 1] = -2
            D[j, j + 2] = 1

        # Construindo a matriz do sistema regularizado
        ATA = np.dot(X.T, X) + lambd * np.eye(n_variables) + beta * np.dot(D.T, D)
        ATb = np.dot(X.T, Y)

        # Resolvendo o sistema linear regularizado
        amplitude = np.linalg.solve(ATA, ATb)

        # Função de custo para minimização
        def FuncaoCusto(a):
            erro = np.linalg.norm(X @ a - Y) ** 2
            reg = lambd * np.linalg.norm(a) ** 2
            suav = beta * np.linalg.norm(D @ a) ** 2
            return erro + reg + suav

        # Resolvendo minimização com restrição de não negatividade
        res = minimize(FuncaoCusto, np.zeros(n_variables), bounds=[(0, None)] * n_variables)

        # Amplitude final
        amplitude = res.x

        return amplitude, T2

      amplitude_ts = []
      tempo_ts = []

      for i in np.arange(len(df['Amostra'])):
        a_ThiSua, t_ThiSua = InversaoThikonovSuavizada(df, N_amostra = i, N_amplitudes = Nom_amplitudes, N_Tempo = Nom_Tempo)
        amplitude_ts.append(a_ThiSua)
        tempo_ts.append(t_ThiSua)

      df['Inversao Thikonov Suave'] = amplitude_ts
      df['Distribuicao T2 Thikonov Suave'] = tempo_ts

    if T2_niumag == True:
      df['T2 Geometrico Niumag'] = t2gm_niumag
      df['T2 Medio Niumag'] = t2av_niumag

    if Erro == True:
      df['Fitting Error'] = fitting_erro

    if T2_Amplitudes == True:
      dados_A = pd.DataFrame([[0 for col in range(N_amp)] for row in range(len(df['Amostra']))])
      colunas = []
      for i in np.arange( len(df['Amostra'])):
        for j in np.arange(N_amp):
          amp = amplitudes[i].reset_index(drop = True)[j]
          df['Amplitudes'] = amplitudes
          N_colunas = 'Amplitude ' + str(j)
          colunas.append(N_colunas)
          dados_A[j][i] = amp
      dados_A.columns = colunas[0:N_amp]
      df = pd.concat([df, dados_A], axis = 1)

    return df

##################################################################################  Próxima Função  ##################################################################################

def TratamentoDadosRMN(Diretorio_pasta, Arquivo_laboratorio, Dados_niumag, Nome_pagina = "Dados",
                       N_tempo = 'Tempo Distribuicao', N_distribuicao = 'Distribuicao T2',
                       Porosidade_i = False, poro_i = 'Porosidade RMN', T2_log = False, Componentes_t2 = False,
                       K_Kenyon = False, Coef_Kenyon_a = 0.01, Coef_Kenyon_b = 2, Coef_Kenyon_c = 4, fracao = 100,
                       N_T2 = 'T2 Ponderado Log', N_phi = 'Porosidade RMN',
                       Conversao_garganta = False, Fator_Cimentacao = False, V_artifical = 1.3, V_geral = 2.0,
                       Fracoes_T2Han = False, Fracoes_T2Ge = False, Localizacao = False,
                       Parametros_lab = ['Amostra', 'Permeabilidade Gas', 'Porosidade Gas', 'Porosidade RMN'],
                       Geometria = False, EPSG = 4326, Conversao = False, N_Conversao = 32724,
                       BVIFFI = False, T_BVIFFI = 32.0, Fator_Formacao = False, Litofacie = False, 
                       Fracoes_arg_cap_ffi = False, T_arg = 3.0, T_cap = 92.0,
                       Dados_porosidade_Transverso = False, N_transverso = 128):
    """
    Esta função trata mesclar os dados já processados de RMN (como processado pela função anterior e que tenha informações da distribuição de tamanho de poros)
    com os dados laboratoriais, que contenham dados de porosidade a gás e de RMN, permeabilidade a gas, litofácies das amostras.

    Args:
        Diretorio_pasta (str): O caminho do diretório onde está o arquivo excel exportado pelo software Niumag.
        Arquivo_laboratorio (str): Nome do arquivo contendo os dados do laboratório em excel.
        Dados_niumag (pandas.DataFrame): DataFrame com as informações selecionadas da distribuição de tamanho de poros.
        N_tempo (str): Nome da coluna que será utilizada para análise do tempo de distribuição.
        N_distribuicao (str): Nome da coluna que será utilizada para análise da amplitude da distribuição.
        Porosidade_i (bool): Caso o usuário queira a transformação do sinal de RMN em porosidade RMN.
        poro_i (str): Nome da coluna com a porosidade que o usuário deseja normalizar o sinal de amplitude da RMN.
        T2_log (bool): Caso o usuário queira calcular o T2_lm proposto por Kenyon et al (1988). OBS: Não está pronto.
        K_Kenyon (bool): Cálculo da Permeabilidade Pelo método de Kenyon com valores dos expodentes informados. Obs: Em desenvolvimento.
        Coef_Kenyon_a (float): Coeficiente a do modelo Kenyon.
        Coef_Kenyon_b (float): Coeficiente b do T2_log do modelo Kenyon.
        Coef_Kenyon_c (float): Coeficiente c da Porosidade do modelo Kenyon.
        Conversao_garganta (bool): Conversão do tempo para garganta de poro seguindo o artigo de Han et al. (2018). Obs: Em desenvolvimento.
        Componentes_t2 (bool): Caso o usuário tenha as componentes que ajustam a curva de relaxação T2.
        Fator_Cimentacao (bool): Caso o usuário queira obter o fator de cimentação.
        V_artifical (int): Fator de cimentação das amostras artificiais.
        V_geral (int): Fator de cimentação das amostras gerais.
        Fracoes_T2Han (bool): Caso o usuário queira retornar as frações da modelagem proposta por Han et al (2018).
        Fracoes_T2Ge (bool): Caso o usuário queira retornar as frações da modelagem proposta por Ge et al (2017).
        Localizacao (bool): Caso o usuário tenha informações sobre a localização das amostras.
        Parametros_lab (list): Informações dos dados laboratórios que o usuário deseja compor no dataframe final.
        Geometria (bool):  Caso o usuário queira retornar a localização em um formato geométrico.
        EPSG (int): Sistema de coordenadas da European Petroleum Survey Group que está a geometria das amostras. Obs: O formato padrão é o WGS84.
        Conversao (bool): Caso o usuário queira converter os o sistema de coordenadas dos dados lab em outro.
        N_Conversao (int): Sistema de coordenadas da European Petroleum Survey Group que o usuário deseja converter a geometrias das amostras. Obs: O formato de conversão padrão é o WTF24M.
        BVIFFI (bool): Caso o usuário queira retornar as frações da modelagem proposta por Coates et al (1999).
        T_BVIFFI (float): Informações do modelo de tempo que o usuário deseja retornar.
        Fator_Formacao (bool): Caso o usuário queira obter o fator de formação.
        Litofacie (bool): Caso o usuário tenha nos dados do laboratório informações sobre as litofácies.
        Fracoes_arg_cap_ffi (bool): Caso o usuário queira retornar as frações referente a porção argila, capilar e livre proposta por Coates et al (1999).
        T_arg (float): Tempo de Argila.
        T_cap (float): Tempo de Capilar.
        Dados_porosidade_Transverso (bool): Caso o usuário queira transformar a lista com os valores da Distribuição de Tamanho de poros em colunas.
        N_transverso (int): Quantidade de pontos da inversão da curva de relaxação T2.

    Returns:
        pandas.DataFrame: Retorna um DataFrame com dados processados do Excel exportado pelo usuário com os dados do laboratório mesclado com os dados da Niumag.

    Exemplos de Uso:
        Caso o usuário tenha um arquivo .xlsx com dados de laboratório e um pandas.DataFrame com dados de Distribuição de Tamanho de Poros
        essa função retornará os dados mesclados e prontos para regressões ou visualizações no formato pandas.DataFrame.
    """

    dados_niumag = Dados_niumag.sort_values(by = 'Amostra')
    dados_lab = pd.read_excel(Diretorio_pasta + Arquivo_laboratorio, sheet_name = Nome_pagina)
    dados_lab['Amostra'] = dados_lab['Amostra'].astype(str)
    dados_lab = dados_lab.sort_values(by = 'Amostra').reset_index(drop = True)

    tempo_distribuicao = np.array(dados_niumag[N_tempo])
    distribuicao_t2 = dados_niumag[N_distribuicao]


    porosi_i = []
    media_ponderada_log = []
    s1h = []
    s2h = []
    s3h = []
    s4h = []
    s1g = []
    s3g = []
    s4g = []
    BVI = []
    FFI = []
    A1 = []
    A2 = []
    A3 = []
    argila = []
    capilar = []
    ffi_cap = []
    k_Kenyon = []
    r_garganta = []

    df_niumag_sa = dados_niumag.drop('Amostra', axis = 1)
    df = pd.concat([dados_lab[Parametros_lab], df_niumag_sa], axis = 1)
    print(df.info())

    if Litofacie == True:
      codi_lab = preprocessing.LabelEncoder()
      categoria_lito = codi_lab.fit_transform(dados_lab['Litofacies'])
      onehot = OneHotEncoder()
      ohe = pd.DataFrame(onehot.fit_transform(dados_lab[['Litofacies']]).toarray())
      ohe.columns = onehot.categories_
      df['Litofacies'] = dados_lab['Litofacies']
      df['Categoria Litofacie'] = categoria_lito
      df = pd.concat([df, ohe], axis = 1)
      # Criação de colunas com valor de 0 ou 1 para cada litofácie

    if Localizacao == True:
      df = pd.concat([df, dados_lab[['X_loc', 'Y_loc', 'Z_loc', 'Caliper']]], axis = 1)

      if Geometria == True:
        geometria = [Point(xy) for xy in zip(dados_lab['X_loc'], dados_lab['Y_loc'])]
        gdf = gpd.GeoDataFrame(dados_lab, geometry=geometria)
        gdf.set_crs(epsg=EPSG, inplace=True)

        df['Geometria EPSG: ' + str(EPSG)] = gdf['geometry']

        if Conversao == True:
          gdf_conv = gdf.to_crs(epsg=N_Conversao)
          df['Geometria EPSG: ' + str(N_Conversao)] = gdf_conv['geometry']
      # Criação da coluna com a geometria do Poço


    if Porosidade_i == True:
      for i in np.arange(len(distribuicao_t2)):
          t2_transpose = pd.DataFrame([distribuicao_t2[i]]).T.reset_index().drop('index', axis = 1).T
          scaler = t2_transpose.T
          scaler_sum_phi = float(dados_lab[str(poro_i)][i])/float(scaler.sum())
          phi_i = []
          for j in np.arange(len(scaler)):
              sc = scaler.T
              p = float(sc[j]*scaler_sum_phi)
              phi_i.append(p)
          porosi_i.append(list(phi_i))
      df['Porosidade_i'] = porosi_i
    # Criação de uma lista de valores com distribuição de porosidade

    if T2_log == True:
      for i in np.arange(len(porosi_i)):
        phi_i = porosi_i[i]
        tempo_log = np.log(tempo_distribuicao[i])
        produto_porosidade_t2_log = pd.DataFrame(phi_i*tempo_log)
        sum_num = np.sum(produto_porosidade_t2_log)
        sum_den = np.sum(phi_i)
        razao_t2 = float(np.exp(sum_num/sum_den))
        media_ponderada_log.append((razao_t2))
    # Calculando o T2_log usado na função do Kenyon et al (1988)
      df["T2 Ponderado Log"] = media_ponderada_log
    
    if K_Kenyon == True:
      for i in np.arange(len(df[N_T2])):
        t2 = df[N_T2][i]
        phi = df[N_phi][i]/fracao
        k = Coef_Kenyon_a * (
          t2 ** Coef_Kenyon_b
          ) * (
          phi ** Coef_Kenyon_c             
            )
        k_Kenyon.append(k)
      df['Permeabilidade SDR'] = k_Kenyon

    if Conversao_garganta == True:
      for i in np.arange(len(porosi_i)):
        indices_micro = np.where(tempo_distribuicao[i] <= 30)[0]
        f_micro = indices_micro[-1]
        r_1 = 0.033*(tempo_distribuicao[i][:f_micro]**0.372)
        r_2 = 0.0003*(tempo_distribuicao[i][f_micro:]**1.84)
        r_total = np.concatenate([r_1, r_2])
        r_garganta.append(r_total)

      df['Raio Garganta'] = r_garganta

    if Componentes_t2 == True:
      df = pd.concat([df, dados_lab[['A_NMR', 'T2_NMR',
                                     'A1', 'T21',
                                     'A2', 'T22',
                                     'A3', 'T23']]], axis = 1)

    if Fator_Cimentacao == True:
      def DefinirValor(litofacies):
        if litofacies == 'Artificial':
            return V_artifical
        else:
            return V_geral
      df['Fator de Cimentacao'] = dados_lab['Litofacies'].apply(DefinirValor)
    # Incremento do valor do Fator de Cimentação

    if Fator_Formacao == True:
      df['Fator de Formacao'] = 1/dados_lab['Porosidade RMN']**df['Fator de Cimentacao']
    # Incremento do valor do Fator de Formação

    if Fracoes_T2Han == True:
      for i in np.arange(len(porosi_i)):
        indices_micro = np.where(tempo_distribuicao[i] <= 30)[0]
        indices_meso = np.where(tempo_distribuicao[i] <= 90)[0]
        indices_macro = np.where(tempo_distribuicao[i] <= 200)[0]

        f_micro = indices_micro[-1]
        f_meso = indices_meso[-1]
        f_macro = indices_macro[-1]

        phi_i = pd.Series(porosi_i[i])
        porosidade = np.sum(porosi_i[i])
        a1h = phi_i[:f_micro].sum()
        a2h = phi_i[f_micro:f_meso].sum()
        a3h = phi_i[f_meso:f_macro].sum()
        a4h = phi_i[f_macro:].sum()
        phimicroh = a1h/porosidade
        phimesoh  = a2h/porosidade
        phimacroh = a3h/porosidade
        phisuperh = a4h/porosidade

        if phimicroh <= 0.0001:
          phimicroh = 0.0001
        if phimesoh <= 0.0001:
          phimesoh = 0.0001
        if phimacroh <= 0.0001:
          phimacroh = 0.0001
        if phisuperh <= 0.0001:
          phisuperh = 0.0001

        s1h.append(phimicroh)
        s2h.append(phimesoh)
        s3h.append(phimacroh)
        s4h.append(phisuperh)

      df['S1Han'] = s1h
      df['S2Han'] = s2h
      df['S3Han'] = s3h
      df['S4Han'] = s4h
      # Cálculo das frações da modelagen Han et al (2018)

    if Fracoes_T2Ge == True:
      for i in np.arange(len(porosi_i)):
        indices_micro = np.where(tempo_distribuicao[i] <= 32)[0]
        indices_meso = np.where(tempo_distribuicao[i] <= 90)[0]
        indices_macro = np.where(tempo_distribuicao[i] <= 180)[0]

        f_micro = indices_micro[-1]
        f_meso = indices_meso[-1]
        f_macro = indices_macro[-1]

        phi_i = pd.Series(porosi_i[i])
        porosidade = np.sum(porosi_i[i])

        phimicrog = (phi_i[:f_micro].sum())/porosidade
        phimesog = (phi_i[f_micro:f_meso].sum())/porosidade
        phimacrog = (phi_i[f_meso:f_macro].sum())/porosidade
        phisuperg = (phi_i[f_macro:].sum())/porosidade


        if phimicrog <= 0.0001:
                phimicrog = 0.0001
        if phimacrog <= 0.0001:
                phimacrog = 0.0001
        if phisuperg <= 0.0001:
                phisuperg = 0.0001

        s1g.append(phimicrog)
        s3g.append(phimacrog)
        s4g.append(phisuperg)

      df['S1Ge'] = s1g
      df['S3Ge'] = s3g
      df['S4Ge'] = s4g
      # Cálculo das frações da modelagen Ge et al (2017)

    if BVIFFI == True:
      for i in np.arange(len(porosi_i)):
        indices_bviffi = np.where(tempo_distribuicao[i] <= T_BVIFFI)[0]
        f_bviffi = indices_bviffi[-1]

        phi_i = pd.Series(porosi_i[i])
        porosidade = np.sum(porosi_i[i])

        b = (phi_i[:f_bviffi].sum())/porosidade
        f = phi_i[f_bviffi:].sum()/porosidade


        if b <= 0.0001:
                b = 0.0001
        if f <= 0.0001:
                f = 0.0001

        BVI.append(b)
        FFI.append(f)

      df['BVI'] = BVI
      df['FFI'] = FFI
      # Cálculo do BVI e FFI para modelagem Coates et al (1999)
    
    if Fracoes_arg_cap_ffi == True:
      for i in np.arange(len(porosi_i)):
        indices_arg = np.where(tempo_distribuicao[i] <= T_arg)[0]
        indices_cap = np.where(tempo_distribuicao[i] <= T_cap)[0]
        f_arg = indices_arg[-1]
        f_cap = indices_cap[-1]

        phi_i = pd.Series(porosi_i[i])
        porosidade = np.sum(porosi_i[i])

        S_arg = (phi_i[:f_arg].sum())/porosidade
        S_cap = (phi_i[f_arg:f_cap].sum())/porosidade
        S_ffi = (phi_i[f_cap:].sum())/porosidade


        if S_arg <= 0.0001:
                  S_arg = 0.0001
        if S_cap <= 0.0001:
                  S_cap = 0.0001
        if S_ffi <= 0.0001:
                  S_ffi = 0.0001

        argila.append(S_arg)
        capilar.append(S_cap)
        ffi_cap.append(S_ffi)

      df['Fracao Argila'] = argila
      df['Fracao Capilar'] = capilar
      df['FFI Capilar'] = ffi_cap

    if Dados_porosidade_Transverso == True:
      dataframe_porosidade = df['Porosidade_i']
      array_amostras = df['Amostra']
      dados_T = pd.DataFrame([[0 for col in range(N_transverso)] for row in range(len(array_amostras))])
      colunas = []
      for i in range(len(array_amostras)):
        for j in np.arange(N_transverso):
          por = dataframe_porosidade[i][j]
          tempo_distribuido = tempo_distribuicao[i][j]
          string = 'T2 ' + str(tempo_distribuido)
          colunas.append(string)
          dados_T[j][i] = por
      dados_T.columns = colunas[0:N_transverso]
      df = pd.concat([df, dados_T], axis = 1)
      # Transformação da lista em colunas
    
    


    df = df.sort_values(by = 'Amostra')

    return df

##################################################################################  Próxima Função  ##################################################################################

def ProcessamentoDadosSDR (Dataframe,
                           Amostra = 'Amostra', T2 = 'T2 Ponderado Log',
                           Porosidade_Gas = 'Porosidade Gas',
                           Porosidade_RMN = 'Porosidade RMN',
                           Permeabilidade_Gas = 'Permeabilidade Gas',
                           Cluster = False, N_Cluster = 'Litofacies'):
  
  """
    Seleciona do DataFrame informado apenas os parâmetros necessário para realizar a regressção proposta por Kenyon et al (1988).

    Args:
        Dataframe (pandas.DataFrame): DataFrame com os dados da amostra, litofácies, o valor de T2_lm, porosidade RMN e Gás, e permeabilidade Gas.

    Returns:
        pandas.DataFrame: Retorna um Dataframe menor com as informações essenciais para a regressão dos dados de permeabilidade proposta por Kenyon et al (1988).

  """
  df = pd.DataFrame({
        'Amostra': Dataframe[Amostra],
        'T2': Dataframe[T2],
        'Porosidade RMN': Dataframe[Porosidade_RMN],
        'Porosidade Gas': Dataframe[Porosidade_Gas],
        'Permeabilidade Gas': Dataframe[Permeabilidade_Gas]
        })

  if Cluster == True:
    df[N_Cluster] = Dataframe[N_Cluster]
  return df

##################################################################################  Próxima Função  ##################################################################################

def ProcessamentoDadosCoates (Dataframe,
                              Amostra = 'Amostra', BVI = 'BVI', FFI = 'FFI',
                              Porosidade_Gas = 'Porosidade Gas', Porosidade_RMN = 'Porosidade RMN',
                              Permeabilidade_Gas = 'Permeabilidade Gas',
                              Cluster = False, N_Cluster = 'Litofacies'):

  """
    Seleciona do DataFrame informado apenas os parâmetros necessário para realizar a regressção proposta por Coates et al (1999).

    Args:
        Dataframe (pandas.DataFrame): DataFrame com os dados da amostra, litofácies, o valor de BVI, FFI, porosidade RMN e Gás, e permeabilidade Gas.

    Returns:
        pandas.DataFrame: Retorna um Dataframe menor com as informações essenciais para a regressão dos dados de permeabilidade proposta por Coates et al (1999).

  """
  df = pd.DataFrame({
        'Amostra': Dataframe[Amostra],
        'BVI': Dataframe[BVI],
        'FFI': Dataframe[FFI],
        'Porosidade RMN': Dataframe[Porosidade_RMN],
        'Porosidade Gas': Dataframe[Porosidade_Gas],
        'Permeabilidade Gas': Dataframe[Permeabilidade_Gas]
        })
                                
  if Cluster == True:
    df[N_Cluster] = Dataframe[N_Cluster]
                                
  return df

##################################################################################  Próxima Função  ##################################################################################

def ProcessamentoDadosHan (Dataframe,
                           Amostra = 'Amostra', S1 = 'S1Han', S2 = 'S2Han', S3 = 'S3Han', S4 = 'S4Han',
                           Porosidade_Gas = 'Porosidade Gas', Porosidade_RMN = 'Porosidade RMN',
                           Permeabilidade_Gas = 'Permeabilidade Gas',
                           Cluster = False, N_Cluster = 'Litofacies'):

  """
    Seleciona do DataFrame informado apenas os parâmetros necessário para realizar a regressção proposta por Han et al (2018).

    Args:
        Dados (pandas.DataFrame): DataFrame com os dados da amostra, litofácies, o valor das frações da curva no tempo de corte S1, S2, S3 e S$, porosidade RMN e Gás, e permeabilidade Gas.

    Returns:
        pandas.DataFrame: Retorna um Dataframe menor com as informações essenciais para a regressão dos dados de permeabilidade proposta por Han et al (2018).

  """
  df = pd.DataFrame({'Amostra': Dataframe[Amostra],
                     'Permeabilidade Gas': Dataframe[Permeabilidade_Gas],
                     'Porosidade Gas': Dataframe[Porosidade_Gas],
                     'Porosidade RMN': Dataframe[Porosidade_RMN],
                     'S1Han': Dataframe[S1],
                     'S2Han': Dataframe[S2],
                     'S3Han': Dataframe[S3],
                     'S4Han': Dataframe[S4]
                    }).replace(0, np.nan).dropna().reset_index().drop('index', axis = 1)
                             
  if Cluster == True:
    df[N_Cluster] = Dataframe[N_Cluster]

  return df

##################################################################################  Próxima Função  ##################################################################################

def ProcessamentoDadosGe (Dataframe,
                          Amostra = 'Amostra', S1 = 'S1Ge', S3 = 'S3Ge', S4 = 'S4Ge',
                          Porosidade_Gas = 'Porosidade Gas', Porosidade_RMN = 'Porosidade RMN',
                          Permeabilidade_Gas = 'Permeabilidade Gas',
                          Cluster = False, N_Cluster = 'Litofacies'):

  """
    Seleciona do DataFrame informado apenas os parâmetros necessário para realizar a regressção proposta por Ge et al (2017).

    Args:
        Dataframe (pandas.DataFrame): DataFrame com os dados da amostra, litofácies, o valor das frações da curva no tempo de corte S1, S3 e S$, porosidade RMN e Gás, e permeabilidade Gas.

    Returns:
        pandas.DataFrame: Retorna um Dataframe menor com as informações essenciais para a regressão dos dados de permeabilidade proposta por Ge et al (2017).

  """
  df = pd.DataFrame({'Amostra': Dataframe[Amostra],
                        'Permeabilidade Gas': Dataframe[Permeabilidade_Gas],
                        'Porosidade Gas': Dataframe[Porosidade_Gas],
                        'Porosidade RMN': Dataframe[Porosidade_RMN],
                        'S1Ge': Dataframe[S1],
                        'S3Ge': Dataframe[S3],
                        'S4Ge': Dataframe[S4]
                       }).replace(0, np.nan).dropna().reset_index().drop('index', axis = 1)
  if Cluster == True:
    df[N_Cluster] = Dataframe[N_Cluster]

  return df

##################################################################################  Próxima Função  ##################################################################################

def ProcessamentoDistribuicaoTreinoTeste (Dados_Treino, Dados_Teste,
                                          Log = False, Permeabilidade = 'Permeabilidade Gas',
                                          Valores = ['T2 0.01',  'T2 0.011',  'T2 0.012',  'T2 0.014',  'T2 0.015',  'T2 0.017',  'T2 0.019',  'T2 0.021',  'T2 0.024',
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
                                          'T2 8969.245',  'T2 10000']):
  
    """
    Seleciona do DataFrame informado as colunas da distribuição de Tamanho de Poros.

    Args:
        Dados_Treino (pandas.DataFrame): DataFrame com os dados da amostra, litofácies, o valor das frações da curva no tempo de corte S1, S3 e S3, porosidade RMN e Gás, e permeabilidade Gas.

    Returns:
        X_treino (pandas.DataFrame): Retorna um Dataframe com as informações essenciais para a regressão dos dados de permeabilidade que utilizem a curva de distribuição de tamanho de poros.
        X_teste (pandas.DataFrame): Retorna um Dataframe com as informações essenciais para a avaliação dos dados de permeabilidade que utilizem a curva de distribuição de tamanho de poros.
        y_treino (numpy.array): Retorna um numpy.array com as informações da permeabilidade de cada amostra para treinamento e regressão.
        y_teste (numpy.array): Retorna um numpy.array com as informações da permeabilidade de cada amostra para avaliação do modelo.

    """
    X_treino = Dados_Treino[Valores]
    y_treino = Dados_Treino[Permeabilidade]
      
    X_teste = Dados_Teste[Valores]
    y_teste = Dados_Teste[Permeabilidade]
                                            
    if Log == True:
      X_treino = Dados_Treino[Valores]
      y_treino = np.log10(Dados_Treino[Permeabilidade]*1000)
      
      X_teste = Dados_Teste[Valores]
      y_teste = np.log10(Dados_Teste[Permeabilidade]*1000)
    
    return X_treino, y_treino, X_teste, y_teste

##################################################################################  Próxima Função  ##################################################################################

def ProcessamentoReservatorio (Dados_com_Previsao, Porosidade_percentual = False,
                               K_base = 'Permeabilidade Gas', P_Base = 'Porosidade Gas', P_novo = 'Porosidade RMN',
                               
                               Modelagens = ['SDR']):

  
  """
    Realiza o processamento dos dados com valores de permeabilidade(s) modelada(s) a fim de obter parâmetros petrofísicos de reservatório proposta por Soto et al (2010).

    Args:
        Dados_com_Previsao (pandas.DataFrame): DataFrame com os dados de permeabilidade a gás e porosidade a gás e RMN.
        Modelagens (list): Lista com os nomes das modelagens que necessita de avaliação dos parâmetros do reservatório.

    Returns:
        pandas.DataFrame: Retorna um acréscimo ao DataFrame com os valores dos parâmetros petrofísicos de reservatório.

  """
  k_gas = Dados_com_Previsao[K_base]
  if Porosidade_percentual == True:
    phi_gas = Dados_com_Previsao[P_Base]/100
    phi_rmn = Dados_com_Previsao[P_novo]/100

  else:
    phi_gas = Dados_com_Previsao[P_Base]
    phi_rmn = Dados_com_Previsao[P_novo] 
  

  phi_z_gas = phi_gas/(1-phi_gas)
  phi_z_rmn = phi_rmn/(1-phi_rmn)

  rqi_gas = (np.pi/100)*np.sqrt(k_gas/phi_gas)
  fzi_gas = rqi_gas/phi_z_gas

  polar_arm_gas = phi_z_gas*(np.sqrt(fzi_gas**2 + 1))
  polar_angle_gas = np.arctan(fzi_gas)

  df = pd.DataFrame({'Phi_z_Gas': phi_z_gas,
                     'Phi_z_RMN': phi_z_rmn,
                     'RQI_Gas': rqi_gas,
                     'FZI_Gas': fzi_gas,
                     'Polar_arm_Gas': polar_arm_gas,
                     'Polar_angle_Gas': polar_angle_gas})

  for i in np.arange(len(Modelagens)):
    k_mod = Dados_com_Previsao['Permeabilidade Prevista ' + Modelagens[i]]

    rqi_mod = (np.pi/100)*np.sqrt(k_mod/phi_rmn)
    fzi_mod = rqi_mod/phi_z_rmn

    polar_arm_mod = phi_z_rmn*(np.sqrt(fzi_mod**2 + 1))
    polar_angle_mod = np.arctan(fzi_mod)


    df['RQI_' + Modelagens[i]] = rqi_mod
    df['FZI_' + Modelagens[i]] = fzi_mod
    df['Polar_arm_' + Modelagens[i]] = polar_arm_mod
    df['Polar_angle_' + Modelagens[i]] = polar_angle_mod


  return pd.concat([Dados_com_Previsao, df], axis = 1)

##################################################################################  Próxima Função  ##################################################################################

def DadosRidgeLine(Dataframe, N_t2 = 3, Cluster = False, Poco = False, N_poco = 'Poço', N_cluster = 'Litofacies',
                   Distribuicao = ['T2 0.01',  'T2 0.011',  'T2 0.012',  'T2 0.014',  'T2 0.015',  'T2 0.017',  'T2 0.019',  'T2 0.021',  'T2 0.024',
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
    Cria um DataFrame concatenado em uma única lista para todos os valores de tempo e da distribuição T2 para visualização da distribuição no formato RidgeLine.

    Args:
        Dados (pandas.DataFrame): DataFrame com os dados de distribuição T2 jpa formato em colunas.
        Distribuicao (list): Lista das colunas da distribuição T2.
        N_t2 (int): Número em que acaba a contagem das letras de cada coluna com o valor do tempo.
        Cluster (bool): Se o usuário quiser obter também as informações de alguma classificação, como as litofácies.
        N_cluster (str): Nome da coluna da classificação.
        Poco (bool): Se o usuário quiser obter também as informações do poço.
        N_poco (str): Nome da coluna do poço.

    Returns:
        pandas.DataFrame: Retorna um DataFrame com os valores em lista das distribuições T2 e os tempos respectivos.

  """

  porosidade_i = []
  tempo_distribuicao = []
  for i in np.arange(len(Dataframe)):
    phi_i = []
    tempo = []
    for j in np.arange(len(Distribuicao)):
      phi_i.append(Dataframe[Distribuicao[j]][i])
      tempo.append(float(Distribuicao[j][N_t2:]))
      porosidade_i.append(phi_i)
      tempo_distribuicao.append(tempo)

  Dados = pd.DataFrame({'Porosidade i': porosidade_i,
                        'Tempo Distribuicao': tempo_distribuicao})
    
  lista_tempo = []
  lista_amostra = []
  lista_t2 = []
  lista_litofacie = []
  lista_poco = []
  for i in np.arange(len(Dataframe)):
      for j in np.arange(len(Dados['Tempo Distribuicao'][0])):
          lista_amostra.append(Dataframe['Amostra'][i])
          lista_tempo.append(Dados['Tempo Distribuicao'][i][j])
          lista_t2.append(Dados['Porosidade i'][i][j])

  df = pd.DataFrame({'Amostra': lista_amostra,
                     'Tempo': lista_tempo,
                     'T2': lista_t2})
  if Poco == True:
    for i in np.arange(len(Dataframe)):
      for j in np.arange(len(Dados['Tempo Distribuicao'][0])):
        lista_poco.append(Dataframe[N_poco][i])
    df[N_poco] = lista_poco

  if Cluster == True:
    
    for i in np.arange(len(Dataframe)):
      for j in np.arange(len(Dados['Tempo Distribuicao'][0])):
          lista_litofacie.append(Dataframe[N_cluster][i])
    df[N_cluster] = lista_litofacie
                              
    
  return df

##################################################################################  Próxima Função  ##################################################################################

def InversaoThikonov(Dataframe, N_amostra = 0, N_amplitudes = 'Picos Amplitudes', N_Tempo = 'Tempo Relaxacao',
                     T2_min = 0.01, T2_max = 10000, T2_ran = 128, Lamb = 0.1):
                       
        t = Dataframe[N_Tempo][N_amostra].astype(float)
        a = df[N_amplitudes][N_amostra].astype(float)

        # Definindo os limites e a faixa de T2
        t2min = T2_min
        t2max = T2_max
        t2range = T2_ran  # número de bins de T2
        T = np.linspace(np.log10(t2min), np.log10(t2max), t2range)
        T2 = [pow(10, i) for i in T]
        x, y = np.meshgrid(t, T2)

        # Matriz de modelo
        X = np.exp(-x/y).T
        Y = np.array(a)

        # Definindo o parâmetro de regularização
        regularzador = Lamb
        n_variables = X.shape[1]

        # Matriz regularizada (Tikhonov)
        ATA = np.dot(X.T, X) + regularzador * np.eye(n_variables)
        ATb = np.dot(X.T, Y)

        # Resolvendo o sistema regularizado
        amplitude = np.linalg.solve(ATA, ATb)

        # Função de erro (mínimos quadrados regularizados)
        def FuncaoCusto(a):
            return np.linalg.norm(X @ a - Y) ** 2 + regularzador * np.linalg.norm(a) ** 2

        # Resolvendo a minimização com restrição de não negatividade
        res = minimize(FuncaoCusto, np.zeros(n_variables), bounds=[(0, None)] * n_variables)

        # Amplitude final
        amplitude = res.x

        return amplitude

##################################################################################  Próxima Função  ##################################################################################

def AnaliseOndaAVS(diretorio = '/content/', arquivo = 'amostra.xlsx', sheet = 'PO_PO', N = 2,
                massa = 57e-3, raio = 1.24e-2, fft_suave = False, fator_suavizacao = 1000):
  df = pd.read_excel(diretorio + arquivo, sheet_name = sheet)
  t_max = df['Time (us)'].max()
  t_min = df['Time (us)'].min()
  t_pas = (t_max-t_min)/(len(df['Time (us)']))
  tempo = np.arange(t_min, t_max, t_pas)[::N]
  amplitude = df['AWT'][::N]-100

  # FFT da Onda
  N_o = len(amplitude)
  dt = np.mean(np.diff(tempo))                                                    # Intervalo médio de amostragem (us)
  fs = 1 / dt                                                                     # Frequência de amostragem (MHz)

  awt_fft = np.fft.fft(amplitude)
  freqs = np.fft.fftfreq(N_o, d = dt)

  # Apenas a metade positiva das frequências
  idx = freqs >= 0
  freqs_pos = freqs[idx]
  FFT_amplitude = np.abs(awt_fft)[idx]

  fase = np.angle(awt_fft[np.argmax(FFT_amplitude)])                              # fase em radianos
  tempo_delta = df['Unnamed: 3'][3]/1e3
  tamanho_amostra = df['Unnamed: 12'][2]
  velocidade = tamanho_amostra/tempo_delta
  volume = np.pi*tamanho_amostra*raio**2
  densidade = massa/volume
  

  df = pd.DataFrame({'Amplitude AVS': amplitude,
                     'Tempo AVS': tempo}).reset_index(drop=True)

  FFT = pd.DataFrame({'Frequência': freqs_pos,
                     'Amplitude Frequencia': FFT_amplitude})
  
  amplitude_max = FFT_amplitude.max()
  indice_max = FFT['Amplitude Frequencia'].idxmax()
  frequencia_max = freqs_pos[indice_max]*1e6

  dados = pd.Series({'Fase': fase,
                     'Tempo de Trânsito': tempo_delta,
                     'Tamanho da Amostra': tamanho_amostra,
                     'Velocidade da Onda': velocidade,
                     'Massa': massa,
                     'Raio': raio,
                     'Volume': volume,
                     'Densidade': densidade,
                     'Amplitude Maxima': amplitude_max,
                     'Frequencia Amplitude': frequencia_max})
  
  if fft_suave == True:
    freqs_dense = np.linspace(freqs_pos.min(), freqs_pos.max(), fator_suavizacao)
    spline = make_interp_spline(freqs_pos, FFT_amplitude)
    amplitude_suave = spline(freqs_dense)
    FFT_suave = pd.DataFrame({'Frequência': freqs_dense,
                              'Amplitude Frequencia': amplitude_suave})

    return df, FFT, dados, FFT_suave
  
  else:     

    return df, FFT, dados

##################################################################################  Próxima Função  ##################################################################################

def FatorQ (resultado_t, resultado_r):
  fator_q = np.pi * resultado_r['Frequencia Amplitude'] / (
      
      resultado_r['Velocidade da Onda'] * np.log(resultado_t['Amplitude Maxima'] / resultado_r['Amplitude Maxima']
      
                                               ) / resultado_r['Tamanho da Amostra'])
  resultado_r['Amplitude Maxima Entrada'] = resultado_t['Amplitude Maxima']
  resultado_r['Fator Q'] = fator_q
  return resultado_r

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
