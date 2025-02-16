# Sobre a biblioteca Petrophysic
Petrophysic é uma biblioteca Python desenvolvida para auxiliar no processamento de dados de Ressonância Magnética Nuclear (RMN), na obtenção de parâmetros como porosidade, componentes T2 e permeabilidade, além de fornecer ferramentas para visualização dos dados. Este projeto é parte de um trabalho de pós-graduação e visa fornecer ferramentas eficientes para a análise de dados de relaxação.

# About the Petrophysic Library
Petrophysic is a Python library developed to assist in the processing of Nuclear Magnetic Resonance (NMR) data, obtaining parameters such as porosity, T2 components, and permeability, as well as providing tools for data visualization. This project is part of a postgraduate work and aims to provide efficient tools for relaxation data analysis.

# Objetivo
O objetivo deste repositório é fornecer uma série de funções para realizar ajustes de modelos exponenciais múltiplos em dados de relaxação de RMN, bem como visualizar os componentes resultantes. As funções são projetadas para facilitar a obtenção de parâmetros petrofísicos de maneira precisa e eficiente.

# Objective
The objective of this repository is to provide a series of functions to perform multiple exponential model fitting on NMR relaxation data, as well as to visualize the resulting components. The functions are designed to facilitate the precise and efficient obtaining of petrophysical parameters.

# Instalação e acesso das funções
Como costumo utilizar o Colaboratory do Google, pela facilidade de acesso nas salas de aula, o teste de instalação está apenas nessa plataforma.
Para instalar a biblioteca em seu colab recomendo utilizar o seguinte código.


#### !rm -rf /content/Petrophysic_Alpha                                       # Remove a pasta antiga, garantindo que você pode estar sempre recebendo as versões mais atualizadas do código.
#### !git clone https://github.com/mauriciomafra-103/Petrophysic_Alpha.git    # Cria um clone do repositório no google, realizando a instalação das funções.

#### import sys
#### sys.path.append('/content/Petrophysic_Alpha/Petrophysic')                # Pegando as funções dentro da pasta específica, então pode ficar de olho em outras pastas com outras funcionalidades que serão postas em breve.

#### from processamento import *   # Importando as funções que estão na pasta de processamento.py
#### from regressao import *       # Importando as funções que estão na pasta de regressao.py
#### from visualizacoes import *   # Importando as funções que estão na pasta de visualizacoes.py



# Installation and Access to Functions
Since I usually use Google Colaboratory for its ease of access in classrooms, the installation test is only on this platform.
To install the library in your Colab, I recommend using the following code.


#### !rm -rf /content/Petrophysic_Alpha                                       # Remove the old folder, ensuring that you always receive the most up-to-date versions of the code.
#### !git clone https://github.com/mauriciomafra-103/Petrophysic_Alpha.git    # Creates a clone of the repository on Google, installing the functions.

#### import sys
#### sys.path.append('/content/Petrophysic_Alpha/Petrophysic')                # Fetching the functions from the specific folder, so keep an eye on other folders with additional features that will be added soon.

#### from processamento import *   # Importing the functions from the processamento.py folder.
#### from regressao import *       # Importing the functions from the regressao.py folder.
#### from visualizacoes import *   # Importing the functions from the visualizacoes.py folder.
