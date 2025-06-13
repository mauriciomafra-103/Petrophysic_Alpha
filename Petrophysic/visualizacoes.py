import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def erro_sigma(previsao, gas):
    N = len(previsao)
    k_p = np.log10(previsao)
    k_g = np.log10(gas)
    N = len(k_p)
    soma = np.sum((k_p-k_g)**2)
    raiz = np.sqrt(soma/N)
    sigma = 10**(raiz)
    return sigma

def VisualizarPredicoesPermeabilidade (Dados, modelo_previsao, Pasta_Salvamento = None, Data= None, Modelo = None,
                          Litofacies = None, Salvar = False, Sigma = False, Valor_Sigma = 3.64):
    eixo_x = 'Gas Permeability (mD)'
    eixo_y = 'NMR Permeability (mD)'
    reta = pd.DataFrame({'x' : np.arange(1000),
                         'y' : np.arange(1000)})
    sns.scatterplot(data = Dados,
                    x = 'Permeabilidade Gas',
                    y = str(modelo_previsao),
                    hue = Litofacies,
                    palette = 'Set1')
    sns.lineplot(data = reta,
                 x = 'x',
                 y = 'y')

    if Sigma == True:
        plt.plot(reta['x'], reta['y'] * Valor_Sigma, "b-.", linewidth=1)
        plt.plot(reta['x'], reta['y'] / Valor_Sigma, "b-.", linewidth=1, label = f'+/- \u03C3: {Valor_Sigma:.2f}')

    plt.xlabel(eixo_x)
    plt.ylabel(eixo_y)
    plt.xlim(0.0001,1000)
    plt.ylim(0.0001,1000)
    plt.legend(loc="upper left", fontsize=10)

    plt.xscale('log')
    plt.yscale('log')
                          

    if Salvar == True:
        plt.savefig(Pasta_Salvamento + Data + titulo + '.png', format='png')
    plt.show()


def VisualizarPorosidade(Dados, Pasta_Salvamento = None, Modelo = None,
                              N_x = 'Porosidade Gas', N_y = 'Porosidade RMN', porcentagem = 1, N_r = 60,
                              L_x = 'Gas Porosity (%)', L_y = 'NMR Porosity (%)', L_t = f'Gas porosity adjustment with NMR\n to the {Modelo}'
                              Hue = None, Litofacies = None, Salvar = False, Erro = False, densidade = 1200):
    titulo = L_t                                        # Nomes do Gráfico
    eixo_x = L_x
    eixo_y = L_y
    legenda = ['Expected outcome', 'NMR Porosity']


    reta = pd.DataFrame({'x' : np.arange(N_r),                                             # Determinando Reta de ajuste
                         'y' : np.arange(N_r)})

    sns.scatterplot(x = Dados[N_x]*porcentagem,
                    y = Dados[N_y]*porcentagem,
                    hue = Hue,
                    palette = 'Set1')

    sns.lineplot(data = reta,
                x = 'x',
                y = 'y')

    if Erro == True:
      valor_erro = np.sqrt(mean_squared_error(Dados[N_x]*porcentagem, Dados[N_y]*porcentagem))
      plt.plot(reta['x'], reta['y'] + valor_erro, "b-.", linewidth=1)
      plt.plot(reta['x'], reta['y'] - valor_erro, "b-.", linewidth=1, label = f'+/- \u03B5: {valor_erro:.2f}')
      plt.legend(loc="upper left", fontsize=10)

    plt.xlabel(eixo_x)                                                                 # Determinando os nomes
    plt.ylabel(eixo_y)
    plt.xlim(0, N_r)
    plt.ylim(0, N_r)


def VisualizarPoroPer(Dados, Modelo, Pasta_Salvamento = None):
    eixo_x = 'Porosity (%)'
    eixo_y = 'Permeability (md)'

    sns.scatterplot(x = Dados['Porosidade Gas']*100,
                    y = Modelo,
                    hue = Dados['Litofacies'],
                    palette = 'Set1')


    plt.xlabel(eixo_x)                                                                 # Determinando os nomes
    plt.ylabel(eixo_y)
    plt.xlim(0, 35.0)
    plt.ylim(0.0001,5000.0)
    plt.yscale('log')
    plt.show()


def HistogramaPermeabilidade(Dados, permeabilidade):
  sns.histplot(data = Dados, x = permeabilidade, kde=True, bins = 32,
               hue = 'Litofacies', multiple="stack", stat="percent",
               log_scale=True, palette = 'Set1')
  plt.ylabel('Percentage (%)')
  plt.xlabel('Permeability (mD)')
  plt.xlim(0.001, 1000)
  plt.show()


def VisualizarDistribuicaoT2 (Dados, CBW = False, Anotacao = False, Salvar = False, N_porosidade = 'Porosidade_i',
                              N_distribuicao = 'Tempo Distribuicao', N_amostra = 'Amostra',
                              V_arg = 3.2, V_capilar = 92.0, Xticks_list = [0.3, 1, 3, 10, 92,  300, 1000, 5000],
                              Ylim = [0, 1], Xlim = [0.01, 10000], FigSize = (20,6)):

    for i in np.arange(0, (len(Dados)-1), 2):
        amostra1 = Dados[N_amostra][i]
        amostra2 = Dados[N_amostra][i+1]
        titulo1 = 'Sample: ' + amostra1
        titulo2 = 'Sample: ' + amostra2
        eixo_x = 'Relaxation Time (ms)'
        eixo_y = r'$\phi$ (%)'
        xticks_list = Xticks_list
        xlim = Xlim
        ylim = Ylim
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize = FigSize)


        x1 = np.array(list(Dados[N_distribuicao][i]))
        y1 = np.array(list(Dados[N_porosidade][i]))
        ax1.plot(x1,y1)
        ax1.set_xlabel(eixo_x)
        ax1.set_ylabel(eixo_y)
        ax1.set(title = titulo1)
        ax1.set_xscale('log')
        ax1.set_xticks(xticks_list)
        ax1.set_xticklabels([f"{tick}" for tick in xticks_list])
        ax1.set_xlim(xlim)
        ax1.set_ylim(ylim)

        if CBW:

          # Região CBW (x < 3.2)
          ax1.fill_between(x1, y1, where=(x1 < V_arg), alpha=0.3, color='orange')
          ax1.text(0.5, y1[30] / 2, 'CBW', fontsize=12)

          # Região CBP (3.2 ≤ x < 33)
          ax1.fill_between(x1, y1, where=((x1 >= V_arg) & (x1 < V_capilar)), alpha=0.3, color='yellow')
          ax1.text(7, y1[30] / 2, 'CBP', fontsize=12)

          # Região FFI (33 ≤ x ≤ 10000)
          ax1.fill_between(x1, y1, where=((x1 >= V_capilar) & (x1 <= 10000)), alpha=0.3, color='green')
          ax1.text(300, y1[30] / 2, 'FFI', fontsize=12)


        if Anotacao == True:

            ax1.annotate('T2 Geométrico Niumag', xy=(Dados['T2 Geometrico Niumag'][i], 0.5), xycoords=("data", "axes fraction"))
            ax1.axvline(x=Dados['T2 Geometrico Niumag'][i], color='lightgray')
            ax1.annotate('T2 Ponderado Log', xy=(Dados['T2 Ponderado Log'][i], 0.7), xycoords=("data", "axes fraction"))
            ax1.axvline(x=Dados['T2 Ponderado Log'][i], color='lightgray')
            ax1.annotate('T2 Médio Niumag', xy=(Dados['T2 Medio Niumag'][i], 0.9), xycoords=("data", "axes fraction"))
            ax1.axvline(x=Dados['T2 Medio Niumag'][i], color='lightgray')


        x2 = np.array(list(Dados[N_distribuicao][i+1]))
        y2 = np.array(list(Dados[N_porosidade][i+1]))
        ax2.plot(x2, y2)
        ax2.set_xlabel(eixo_x)
        ax2.set_ylabel(eixo_y)
        ax2.set(title = titulo2)
        ax2.set_xscale('log')
        ax2.set_xticks(xticks_list)
        ax2.set_xticklabels([f"{tick}" for tick in xticks_list])
        ax2.set_xlim(xlim)
        ax2.set_ylim(ylim)

        if Anotacao == True:
            ax2.annotate('T2 Geométrico Niumag', xy=(Dados['T2 Geometrico Niumag'][i+1], 0.5), xycoords=("data", "axes fraction"))
            ax2.axvline(x=Dados['T2 Geometrico Niumag'][i+1], color='lightgray')
            ax2.annotate('T2 Ponderado Log', xy=(Dados['T2 Ponderado Log'][i+1], 0.7), xycoords=("data", "axes fraction"))
            ax2.axvline(x=Dados['T2 Ponderado Log'][i+1], color='lightgray')
            ax2.annotate('T2 Médio Niumag', xy=(Dados['T2 Medio Niumag'][i+1], 0.9), xycoords=("data", "axes fraction"))
            ax2.axvline(x=Dados['T2 Medio Niumag'][i+1], color='lightgray')

        if CBW == True:
            ax2.text(0.5,y2[30]/2, 'CBW', fontsize = 12)
            ax2.fill_between(x2, y2, where = x1 < V_arg, alpha = 0.3, color = 'orange')
            ax2.text(7,y2[30]/2, 'CBP', fontsize = 12)
            ax2.fill_between(x2, y2, where = (x1 >= V_arg) & (x1 < V_capilar), alpha = 0.3, color = 'yellow')
            ax2.text(300,y2[30]/2, 'FFI', fontsize = 12)
            ax2.fill_between(x2, y2, where = (x1 >= V_capilar) & (x1 <= 10000), alpha = 0.3, color = 'green')



def VisualizarRQI(dados, regressao, phi, rqi, modelo, ylim = [0.001, 10], xlim = [0.01, 1]):
  eixo_x = 'φz'
  eixo_y = 'RQI'

  reg = regressao.copy().set_index('Litofacies')


  xis = np.arange(0, 2, 0.1)

  y_grai = reg['a']['Grainstone_' + modelo]*xis
  y_dolo = reg['a']['Dolowackstone_' + modelo]*xis
  y_biot = reg['a']['Bioturbated_' + modelo]*xis
  y_brec = reg['a']['Brechado_' + modelo]*xis
  #y_arti = reg['a']['Artificial_' + modelo]*xis
  y_pack = reg['a']['Packstone_' + modelo]*xis

  reta = pd.DataFrame({'x' : xis,
                       'y_grai' : y_grai,
                       'y_dolo' : y_dolo,
                       'y_biot' : y_biot,
                       'y_brec' : y_brec,
                       'y_pack': y_pack})

  sns.scatterplot(dados, x = phi, y = rqi, hue = 'Litofacies', palette = 'Set1')
  sns.lineplot(reta, x = 'x', y = 'y_grai', color = 'red')
  sns.lineplot(reta, x = 'x', y = 'y_dolo', color = 'blue')
  sns.lineplot(reta, x = 'x', y = 'y_brec', color = 'green')
  sns.lineplot(reta, x = 'x', y = 'y_biot', color = 'purple')
  #sns.lineplot(reta, x = 'x', y = 'y_arti', color = 'yellow')
  sns.lineplot(reta, x = 'x', y = 'y_pack', color = 'orange')

  plt.xlabel(eixo_x)                                                                 # Determinando os nomes
  plt.ylabel(eixo_y)
  plt.ylim(ylim)
  plt.xlim(xlim)
  plt.text(s = 'Modelo ' + modelo, x = 0.1, y = 0.003)


  plt.xscale('log')                                                                    # Escala dos eixos
  plt.yscale('log')
  plt.show()


def VisualizarSigmoid(dados, modelo):
  # Coeficientes da Literatura
  A = -3.5916207
  B = 5.06265818
  C = -0.72243226
  D = 0.371324681


  pontos = np.arange(0, 15, 15/len(dados))
  sigmoid = A + B / (1 + (np.exp(-(pontos - C)/D)))
  eixo_x = 'r, Polar_arm_' + modelo
  eixo_y = 'teta, Polar_angle_' + modelo


  sns.lineplot(dados, x = pontos, y = sigmoid)
  sns.scatterplot(dados, x = 'Polar_arm_' + modelo, y = 'Polar_angle_' + modelo, hue = 'Litofacies', palette = 'Set1')

  plt.xlabel(eixo_x)                                                                 # Determinando os nomes
  plt.ylabel(eixo_y)

  plt.xlim(0, 15)
  plt.ylim(0, 2)
  plt.show()


def VisualizarRidgeLinePermeabilidade(dados_totais, dados_ridge, permeabilidade_rf, permeabilidade_gb, permeabilidade_mlp, permeabilidade_sdr, permeabilidade_han, modelo):
    for i in np.arange(len(dados_ridge['Poço'].unique())):
        df_copia = dados_ridge.copy()

        df_dados = dados_totais.loc[dados_totais['Poço'] == dados_totais['Poço'].unique()[i]].reset_index().drop('index', axis = 1)
        dados_plot = df_copia.loc[df_copia['Poço'] == df_copia['Poço'].unique()[i]].reset_index().drop('index', axis = 1)

        array = np.arange(len(dados_plot.Amostra.unique()))
        fig = plt.figure(figsize = (2,10))
        cores = [ 'mediumorchid', 'mediumseagreen', 'cornflowerblue', 'indianred', 'y', 'k', 'w']

        for j in np.arange(len(array)):
            x = dados_plot['Tempo'][j*128:j*128+128]
            y = dados_plot['T2'][j*128:j*128+128]
            cor = dados_plot['Litofacie'][j*128]
            poço = dados_plot['Poço'][0]
            #espacamento = j*-0.02                   # Espaçamento para poços com muitos dados
            espacamento = j*-0.2/len(array)        # Espaçamento para poços com poucos dados

            if j == array[0]:
                ax = str(array[j])
                ax = fig.add_axes([0, espacamento, 1, 0.05])
                ax.fill(x, y, alpha = 0.7, color = cores[cor], linewidth=2)
                ax.set_xscale('log')
                ax.set_frame_on(False)
                ax.get_yaxis().set_visible(False)
                ax.get_xaxis().set_visible(False)
                secx = ax.secondary_xaxis(1.2)
                secx.set_xlabel('ms')
                secx.set_xticks([0.01, 1, 100, 10000])
                plt.title(f'Pseudo-pore size \n distribution \n Well {poço}', fontsize=12, loc = 'center', y = 3)
            else:
                ax = str(array[j])
                ax = fig.add_axes([0, espacamento, 1, 0.05])
                ax.fill(x, y, alpha = 0.7, color = cores[cor], linewidth=2)
                ax.set_xscale('log')
                ax.set_frame_on(False)
                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(False)

        tam = -1*espacamento+0.01
        ax = fig.add_axes([-6, espacamento, 1, tam])          # Permeabilidade Gás
        ax1 = fig.add_axes([-6, espacamento, 1, tam])         # Permeabilidade SDR
        ax2 = fig.add_axes([-4.8, espacamento, 1, tam])       # Permeabilidade Gás
        ax3 = fig.add_axes([-4.8, espacamento, 1, tam])       # Permeabilidade Han
        ax4 = fig.add_axes([-3.6, espacamento, 1, tam])       # Permeabilidade Gás
        ax5 = fig.add_axes([-3.6, espacamento, 1, tam])       # Permeabilidade Random Forest
        ax6 = fig.add_axes([-2.4, espacamento, 1, tam])       # Permeabilidade Gás
        ax7 = fig.add_axes([-2.4, espacamento, 1, tam])       # Permeabilidade Gradient Boosting
        ax8 = fig.add_axes([-1.2, espacamento, 1, tam])       # Permeabilidade Gás
        ax9 = fig.add_axes([-1.2, espacamento, 1, tam])       # Permeabilidade Multi Layer Perceptron

        poço = df_dados['Poço'][0]

        ax.plot(df_dados['Permeabilidade Gas'], df_dados['Amostra'], marker='p', color = 'r')
        ax.set_frame_on(False)
        ax.set_xscale('log')
        ax.invert_yaxis()
        ax.get_yaxis().set_visible(False)
        ax.get_xaxis().set_visible(False)
        secx = ax.secondary_xaxis(1.25)
        ax.set_xlim(0.001, 1000)
        secx.set_xticks([0.001, 0.01, 0.1, 1, 10, 100, 1000])
        secx.set_xlabel('Gas Permeability (mD)', color = 'r')


        ax1.plot(df_dados[permeabilidade_sdr], df_dados['Amostra'], marker='x', color = 'b')
        ax1.set_frame_on(False)
        ax1.set_xscale('log')
        ax1.invert_yaxis()
        ax1.get_yaxis().set_visible(False)
        ax1.get_xaxis().set_visible(False)
        secx1 = ax1.secondary_xaxis(1.55)
        ax1.set_xlim(0.001, 1000)
        secx1.set_xticks([0.001, 0.01, 0.1, 1, 10, 100, 1000])
        secx1.set_xlabel(f'Permeability \n {permeabilidade_sdr} (mD)', color = 'b')
        ax1.set_title(f'Permeability prediction \n\n Well {poço}', fontsize=12, loc = 'center', y = 1.9)


        ax2.plot(df_dados['Permeabilidade Gas'], df_dados['Amostra'], marker='p', color = 'r')
        ax2.set_frame_on(False)
        ax2.set_xscale('log')
        ax2.invert_yaxis()
        ax2.get_yaxis().set_visible(False)
        ax2.get_xaxis().set_visible(False)
        secx2 = ax2.secondary_xaxis(1.25)
        ax2.set_xlim(0.001, 1000)
        secx2.set_xticks([0.001, 0.01, 0.1, 1, 10, 100, 1000])
        secx2.set_xlabel('Gas Permeability (mD)', color = 'r')


        ax3.plot(df_dados[permeabilidade_han], df_dados['Amostra'], marker='x', color = 'b')
        ax3.set_frame_on(False)
        ax3.set_xscale('log')
        ax3.invert_yaxis()
        ax3.get_yaxis().set_visible(False)
        ax3.get_xaxis().set_visible(False)
        secx3 = ax3.secondary_xaxis(1.55)
        ax3.set_xlim(0.001, 1000)
        secx3.set_xticks([0.001, 0.01, 0.1, 1, 10, 100, 1000])
        secx3.set_xlabel(f'Permeability \n {permeabilidade_han} (mD)', color = 'b')
        ax3.set_title(f'Permeability prediction \n\n Well {poço}', fontsize=12, loc = 'center', y = 1.9)

        ax4.plot(df_dados['Permeabilidade Gas'], df_dados['Amostra'], marker='p', color = 'r')
        ax4.set_frame_on(False)
        ax4.set_xscale('log')
        ax4.invert_yaxis()
        ax4.get_yaxis().set_visible(False)
        ax4.get_xaxis().set_visible(False)
        secx4 = ax4.secondary_xaxis(1.25)
        ax4.set_xlim(0.001, 1000)
        secx4.set_xticks([0.001, 0.01, 0.1, 1, 10, 100, 1000])
        secx4.set_xlabel('Gas Permeability (mD)', color = 'r')


        ax5.plot(df_dados[permeabilidade_rf], df_dados['Amostra'], marker='x', color = 'b')
        ax5.set_frame_on(False)
        ax5.set_xscale('log')
        ax5.invert_yaxis()
        ax5.get_yaxis().set_visible(False)
        ax5.get_xaxis().set_visible(False)
        secx5 = ax5.secondary_xaxis(1.55)
        ax5.set_xlim(0.001, 1000)
        secx5.set_xticks([0.001, 0.01, 0.1, 1, 10, 100, 1000])
        secx5.set_xlabel(f'Permeability \n {permeabilidade_rf} (mD)', color = 'b')
        ax5.set_title(f'Permeability prediction \n\n Well {poço}', fontsize=12, loc = 'center', y = 1.9)


        ax6.plot(df_dados['Permeabilidade Gas'], df_dados['Amostra'], marker='p', color = 'r')
        ax6.set_frame_on(False)
        ax6.set_xscale('log')
        ax6.invert_yaxis()
        ax6.get_yaxis().set_visible(False)
        ax6.get_xaxis().set_visible(False)
        secx6 = ax6.secondary_xaxis(1.25)
        ax6.set_xlim(0.001, 1000)
        secx6.set_xticks([0.001, 0.01, 0.1, 1, 10, 100, 1000])
        secx6.set_xlabel('Gas Permeability (mD)', color = 'r')


        ax7.plot(df_dados[permeabilidade_gb], df_dados['Amostra'], marker='x', color = 'b')
        ax7.set_frame_on(False)
        ax7.set_xscale('log')
        ax7.invert_yaxis()
        ax7.get_yaxis().set_visible(False)
        ax7.get_xaxis().set_visible(False)
        secx7 = ax7.secondary_xaxis(1.55)
        ax7.set_xlim(0.001, 1000)
        secx7.set_xticks([0.001, 0.01, 0.1, 1, 10, 100, 1000])
        secx7.set_xlabel(f'Permeability \n {permeabilidade_gb} (mD)', color = 'b')
        ax7.set_title(f'Permeability prediction \n\n Well {poço}', fontsize=12, loc = 'center', y = 1.9)

        ax8.plot(df_dados['Permeabilidade Gas'], df_dados['Amostra'], marker='p', color = 'r')
        ax8.set_frame_on(False)
        ax8.set_xscale('log')
        ax8.invert_yaxis()
        ax8.get_yaxis().set_visible(False)
        ax8.get_xaxis().set_visible(False)
        secx8 = ax8.secondary_xaxis(1.25)
        ax8.set_xlim(0.001, 1000)
        secx8.set_xticks([0.001, 0.01, 0.1, 1, 10, 100, 1000])
        secx8.set_xlabel('Gas Permeability (mD)', color = 'r')
        ax8.set_title(f'Permeability prediction \n\n Well {poço}', fontsize=12, loc = 'center', y = 1.9)

        ax9.plot(df_dados[permeabilidade_mlp], df_dados['Amostra'], marker='x', color = 'b')
        ax9.set_frame_on(False)
        ax9.set_xscale('log')
        ax9.invert_yaxis()
        ax9.get_yaxis().set_visible(False)
        ax9.get_xaxis().set_visible(False)
        secx9 = ax9.secondary_xaxis(1.55)
        ax9.set_xlim(0.001, 1000)
        secx9.set_xticks([0.001, 0.01, 0.1, 1, 10, 100, 1000])
        secx9.set_xlabel(f'Permeability \n {permeabilidade_mlp} (mD)', color = 'b')

        plt.savefig(f'/content/sample_data/Teste-{poço}.png', format='png', dpi = 300)
        plt.show()

def VisualizarContornoSESPI(DataFrame_SESPI):
      # Dados fornecidos
      posicao = pd.Series(DataFrame_SESPI.drop('Time(ms)', axis = 1).columns)
      distribuicao = pd.Series(DataFrame_SESPI['Time(ms)'])
      z = pd.DataFrame(DataFrame_SESPI.set_index('Time(ms)')).T
    
      # Criação da grade de pontos
      xi, yi = np.meshgrid(distribuicao, posicao)
    
      # Somatórias das colunas e linhas
      soma_colunas = z.sum(axis=0)
      soma_linhas= z.sum(axis=1)
    
      # Criação do layout com gridspec
      soma_colunas_normalizada = soma_colunas / soma_colunas.max()
      soma_linhas_normalizada = soma_linhas / soma_linhas.max()
    
      # Criação do layout com gridspec
      fig = plt.figure(figsize=(10, 8))
      gs = gridspec.GridSpec(2, 2, width_ratios=[3, 1], height_ratios=[1, 3], wspace=0.01, hspace=0.01)
    
    
      # Gráfico de barras para a somatória das colunas
      ax0 = plt.subplot(gs[0, 0])
      ax0.plot(distribuicao, soma_colunas_normalizada, linestyle='-', color='skyblue')
      ax0.set_ylabel('S(T2)')
      ax0.set_xscale('log')
      ax0.tick_params(axis='both', which='major', labelsize=8)
    
    
      # Gráfico de barras para a somatória das linhas
      ax1 = plt.subplot(gs[1, 1])
      y_smooth = np.linspace(posicao.min(), posicao.max(), 1000)
      spl = make_interp_spline(posicao, soma_linhas_normalizada, k=3)
      row_sums_smooth = spl(y_smooth)
      #ax1.plot(soma_linhas_normalizada, posicao, color='skyblue')
      ax1.plot(row_sums_smooth, y_smooth, color='skyblue')  # spline suavizada
      ax1.set_xlabel('S(Phi)')
      ax1.tick_params(axis='both', which='major', labelsize=8)
    
      # Mapa de contorno
      ax2 = plt.subplot(gs[1, 0])
      contour = ax2.contourf(xi, yi, z, levels=20, cmap="jet")
      ax2.set_xlabel('T2 (ms)')
      ax2.set_ylabel('Posição (mm)')
      ax2.set_xscale('log')
      ax2.tick_params(axis='both', which='major', labelsize=8)
    
      cbar_ax = fig.add_axes([0.705, 0.69, 0.195, 0.19])  # [esquerda, bottom, width, height]
      cbar = plt.colorbar(contour, cax=cbar_ax)
      cbar.set_ticks([])  # Remove os ticks da barra de cores
    
      plt.tight_layout()
      plt.show()
