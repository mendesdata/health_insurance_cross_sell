# Health Insurance Cross-Sell Project
##  Propensão de Compra com Classificação
<img src="img/capa.jpg">

## 1. Questão de Negócio

O cliente deste projeto é uma seguradora de planos de saúde que pretende passar a oferecer aos seus segurados um novo produto: Seguro de automóveis. Para tal, contratou uma pesquisa para que seus clientes respondessem a seguinte pergunta: Você contrataria um seguro de automóvel conosco ?

A pesquisa foi encomendada pela área comercial da Companhia, que solicitou ao setor de Dados um projeto que potencialize o processo de captação dos clientes do plano de saúde para o seguro de automóveis. O setor Financeiro também deseja que o produto de dados seja capaz de realizar previsões de custos operacionais e de receitas para diversos cenários de conversões de clientes. Completando os stakeholders, o setor de Atendimento ao Cliente será responsável pela operação de vendas do novo produto e para tal solicitou a relação dos possíveis interessados.

## 2. Premissas de Negócio

### 2.1 O Conjunto de Dados

Foram entrevistados cerca de 380 mil clientes da Companhia de Seguros. Além da resposta, os seguintes dados dos clientes serão disponibilizados ao time de Dados:

A base de dados utilizada na construção desse projeto pode ser encontrada dentro da plataforma [Kaggle](https://www.kaggle.com/datasets/anmolkumar/health-insurance-cross-sell-prediction).
O conjunto de dados deste projeto está no arquivo **train.csv**

As colunas da base de dados são:

| Nome da Coluna | Descrição da Coluna |
| :------------- | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| id | ID do cliente                            |
| gender | Gênero do cliente |
| age | Idade do cliente|
| driving_license | 0 : Cliente não tem licença para dirigir / 1 : Cliente tem licença para dirigir |
| region_code | Código da região do cliente  |
| previously_insured | 0 : Cliente não possui seguro de automóvel / 1 : Cliente já possui seguro de automóvel|
| vehicle_age | Idade do veículo |
| vehicle_damage | 0 : Veículo já envolvido em aciedente / 1 : Veículo nunca envolvido em acidente |
| annual_premium | Valor anual do seguro pago pelo cliente|
| policy_sales_channel | Código do canal de divulgação ao cliente  |
| vintage | Número de Dias que o cliente é segurado da Companhia |
| response | 0 : Cliente não interessado / 1 : Cliente interessado |

### 2.2 Ferramentas e Métodos Utilizados
- Python 3.11.5
- Jupyter Notebook e VS Code
- CRISP-DS
- Git e GitHub
- Aprendizado Supervisionado - Classificação
- Algoritmos: KNN, Logistic Regression, Extra Trees e LGBM Classifier

### 2.3 Restrições
* ID de restaurantes duplicados serão removidos da base de dados.
* Linhas que possuírem colunas com valor indefinido **(NaN)** serão removidas da base de dados
* Será considerado apenas um tipo de culinária(**coluna cuisines**) por restaurante, sendo o primeiro tipo de culinária da lista
* Os restaurantes serão ranqueados conforme a seguinte ordem: maior nota média de avaliação dos clientes, maior número de avaliações recebidas, maior tempo de cadastro na plataforma.

### 2.4 Perguntas de Negócio
01. Qual número e percentual de clientes interessados em adquirir o seguro automóvel foram apurados a partir da pesquisa ?
02. Qual o percentual de clientes que comprarão o seguro automóvel para cada faixa percentual de 10% do total de clientes ?
03. Em quantas vezes, o modelo do projeto é mais eficiente que um modelo aleatório de escolhas de clientes ?
04. Quais as previsões de custo operacional do modelo para cada faixa de 10% de clientes interessados em adquirir o seguro de automóveis ? Considere um custo médio de U$ 5.00 por cliente contactado. Compare com o modelo aleatório.
05. Quais as previsões de receita do modelo para cada faixa de 10% de clientes contactados ? Considere um ticket médio de U$ 120.00 por seguro vendido. Compare com o modelo aleatório.
06. Faça as mesmas previsões dos dois itens anteriores considerando um percentual específico de clientes que comprarão o seguro de automóveis. Informe o número e o percentual de clientes contactados para o atingimento desta meta. Crie uma listagem que identifique esses clientes.
07. Que insights obtidos pela análise de dados podem contribuir para o negócio da empresa ?

## 3. Estratégia de Resolução
01. **Especificação da Questão de Negócio**: Delimitar o escopo do projeto, quem são os interessados, as premissas, os objetivos.
02. **Coleta e Tratamento de Dados**: Verificar se há dados faltantes, duplicados ou inconsistentes.
03. **Análise Descritiva dos Dados**: Visualizar métricas de tendência central, frequência e distribuição dos dados.
04. **Elaboração de Hipóteses de Negócio**: Criar a partir do dados originais, dados derivados ou conhecimento empírico afirmações (hipóteses) sobre o fenômeno estudado.
05. **Análise Exploratória dos Dados**: Validar ou invalidar as hipóteses levantadas no passo anterior, a partir das análises da relações e correlações das variáveis do modelo. Além disso, gerar insights que possam ser convertidos em retorno financeiro para a Empresa.
06. **Preparação e Seleção dos Atributos**: Utilizar técnicas e ferramentas que permitam que os dados representem o fenômeno estudado.
07. **Geração de Modelos de Machine Learning**: Criar, treinar e validar modelos que permitam responder às perguntas de negócios, utilizando métricas de performance para escolha do melhor modelo
08. **Aprimoramento do Modelo**: Ajustar os parâmetros do modelo selecionado para obtenção de melhor performance.
09. **Comunicação dos Resultados aos Stakeholders**: Apresentar as métricas que comprovam a eficiência da aplicação do modelo,  demonstrar as previsões do modelo e responder às perguntas de negócio. Apresentar também, as lições aprendidas e os próximos passos do projeto.
10. **Deploy do Modelo**: Colocar o modelo em ambiente de produção para que possa ser consumido e abastecido com novos dados.


# 3. Resultado de Negócio
## 3.1. Oportunidades Identificadas

## 3.2. Exibição das Análises

# 4. Lições Aprendidas

# 5. Próximos Passos
