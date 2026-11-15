# Análise e Planejamento do Trabalho 3 – GCC1734 (Inteligência Artificial)

Documento de referência com a análise detalhada da proposta do **Trabalho 3 – Aprendizado de Máquina, Modelos de Linguagem e Agentes Baseados em LLMs** (`GCC1734_2026_2_T3.pdf`), personalizado para a matrícula **2221875BCC**.

---

## 1. Visão Geral do Trabalho

O objetivo do trabalho é construir um **assistente de análise de dados** executável e auditável em um domínio público e de baixo risco, integrando três pilares da disciplina em um único projeto:
1. **Aprendizado de Máquina Supervisionado** (treinamento e avaliação de um modelo preditivo tabular);
2. **Modelagem de Linguagem** (tokenização, modelos $n$-gramas autoregressivos e análise teórica de Transformers e alinhamento);
3. **Agentes Baseados em LLMs** (RAG, ferramentas, padrão ReAct, contrato MCP, controle de fluxo com grafo de estados/LangGraph, guardrails, testes determinísticos e observabilidade).

---

## 2. Definição do Conjunto de Dados (Matrícula: `2221875BCC`)

A escolha do dataset e do domínio base do projeto é definida pela fórmula da **Q1**:
$$\text{ID}(m) = [(\text{parte numérica de } m) \bmod 4] + 1$$

* **Parte numérica da matrícula ($m$):** `2221875`
* **Cálculo do resto:** $2221875 \div 4 = 555468$ com resto **$3$** ($2221875 \bmod 4 = 3$)
* **ID resultante:** $3 + 1 = \mathbf{4}$

> **Dataset atribuído:** **4. California Housing – Regressão** (`sklearn.datasets.fetch_california_housing`)  
> **Domínio do Projeto:** Análise habitacional/imobiliária e socioeconômica censitária da Califórnia (domínio de dados públicos e exploratórios; o sistema deve ser estritamente voltado para análise exploratória/educacional de dados censitários e **não** deve fornecer aconselhamento financeiro ou de investimento).

---

## 3. Passo a Passo Detalhado da Proposta (Q1 a Q6)

### Passo 1: Q1 (1,5 ponto) – Dados, Preparação e Protocolo Experimental
Nesta etapa, será preparada a base de dados **California Housing** para a tarefa de **regressão** (previsão do valor mediano das casas, `MedHouseVal`, a partir de 8 atributos socioeconômicos e geográficos como renda mediana `MedInc`, idade da casa `HouseAge`, média de cômodos `AveRooms`, média de quartos `AveBedrms`, população `Population`, ocupação média `AveOccup`, `Latitude` e `Longitude`).

**O que será feito:**
1. **Descrição do domínio e dados:** Explicar as 8 variáveis de entrada (*features*), a variável-alvo (`MedHouseVal`, expressa em centenas de milhares de dólares), o tipo de tarefa (regressão contínua) e os riscos de interpretação do dataset (ex.: dados históricos do censo de 1990, teto artificial/censura no valor de US$ 500.000, agregação por blocos censitários e não imóveis individuais).
2. **Divisão dos dados:** Separar o dataset em **Treino**, **Validação** e **Teste** (ex.: 70% / 15% / 15% ou 60% / 20% / 20%), justificando as proporções escolhidas e fixando uma semente aleatória (`random_state`, como `2221875`) para reprodutibilidade.
3. **Pipeline de pré-processamento:** Construir um `Pipeline` (ex.: `StandardScaler` do `scikit-learn`) ajustado (`fit`) **exclusivamente no conjunto de treino** e apenas aplicado (`transform`) nos conjuntos de validação e teste.
4. **Análise exploratória:** Reportar cardinalidade (20.640 amostras), dimensões (8 variáveis explicativas + 1 alvo) e estatísticas descritivas (média, desvio-padrão, mínimos, máximos, quartis, valores ausentes/outliers).
5. **Prevenção de vazamento e diagrama:** Explicar como o protocolo evita *data leakage* e incluir no relatório o diagrama do fluxo:
   `dados brutos → divisão → pré-processamento → treinamento → validação → teste final`.

---

### Passo 2: Q2 (1,5 ponto) – Modelos e Aprendizado Supervisionado
Como a tarefa é **regressão**, deverão ser treinados e comparados modelos de regressão supervisionada sobre o California Housing.

**O que será feito:**
1. **Treinamento e Avaliação na Validação:** Treinar pelo menos três famílias de modelos exigidas para regressão:
   * **Regressão Linear** (ou **Ridge**);
   * **Árvore de Decisão** (`DecisionTreeRegressor`);
   * **Random Forest Regressor** (`RandomForestRegressor`).
   Avaliar todos no conjunto de **validação** usando as métricas de regressão exigidas: **MSE** (Erro Quadrático Médio), **RMSE** (Raiz do Erro Quadrático Médio) e **MAE** (Erro Absoluto Médio).
2. **Estudo de Hiperparâmetros:** Escolher um dos modelos (ex.: `RandomForestRegressor` ou `Ridge`) e testar pelo menos **3 configurações de hiperparâmetros** no conjunto de validação (ex.: variando `max_depth` e `n_estimators`, ou o parâmetro `alpha` no Ridge).
3. **Análise Teórica e Diagnóstico:** Comparar o desempenho de treino vs. validação dos modelos discutindo **viés (*bias*)**, **variância (*variance*)**, ***underfitting*** (comum na Regressão Linear simples), ***overfitting*** (comum na Árvore de Decisão sem restrição de profundidade) e **capacidade de generalização**.
4. **Seleção e Teste Final:** Selecionar o melhor modelo com base na validação e avaliá-lo no **conjunto de teste apenas uma única vez ao final** (este modelo será exportado/integrado para ser usado como ferramenta pelo agente na Q4).
5. **Visualização:** Gerar pelo menos um gráfico de diagnóstico adequado para regressão (ex.: gráfico de **resíduos**, valores reais vs. preditos, curva de erro ou **importância das características / *feature importance***) e explicar sua relevância.

---

### Passo 3: Q3 (1,5 ponto) – Modelagem de Linguagem e Alinhamento
Esta etapa une uma implementação prática de modelo de linguagem estatístico autoregressivo ($n$-gramas) com a fundamentação teórica de LLMs modernos.

**O que será feito:**
1. **Construção do Corpus:** Reunir um pequeno corpus textual público relacionado ao domínio habitacional/demográfico da Califórnia e regressão (com **pelo menos 3 fontes públicas**, registrando origem e licença/permissão de uso, que também servirão de base na pasta `base_conhecimento/`), dividido em `train` e `validation`.
2. **Tokenização:** Implementar e documentar o pipeline de tokenização sobre o corpus, informando tamanho do vocabulário e como são tratados pontuação, números, letras maiúsculas/minúsculas e tokens desconhecidos (`<UNK>`).
3. **Modelos Autoregressivos $n$-grama:** Implementar um modelo **Unigram** e um modelo **Bigram** (ou **Trigram**), incluindo tratamento explícito de probabilidades zero (ex.: suavização de Laplace / Add-$k$).
4. **Avaliação Intrínseca:** Calcular a **perda média (entropia cruzada / *cross-entropy loss*)** e a **perplexidade (*perplexity*)** no conjunto de validação.
5. **Comparação e Geração:** Comparar duas configurações (ex.: Unigram vs. Bigram, ou diferentes valores de suavização $\alpha$), analisar exemplos das próximas palavras mais prováveis dado um contexto e explicar a relação entre probabilidade condicional autoregressiva $P(w_t \mid w_{1:t-1})$ e previsão do próximo token.
6. **Discussão Conceitual Obrigatória (no relatório e no vídeo):**
   * Diferença entre **tokenização** e **embedding**;
   * Por que a **máscara causal** (*causal mask*) é necessária em modelos autoregressivos;
   * Papel da **atenção**, **múltiplas cabeças (*multi-head attention*)** e **redes *feed-forward* por posição (*position-wise transformation*)** no Transformer;
   * Diferença entre **Pré-treinamento**, **SFT (*Supervised Fine-Tuning*)**, **RLHF com PPO** e **DPO (*Direct Preference Optimization*)**;
   * Por que nenhuma dessas etapas, isoladamente, garante factualidade, segurança ou ausência de viés.

---

### Passo 4: Q4 (2,0 pontos) – RAG, Ferramentas, ReAct e MCP
Transformação do modelo preditivo da Q2 e dos documentos do domínio em capacidades acionáveis por um **Agente baseado em LLM**.

**O que será feito:**
1. **Base de Conhecimento e Pipeline RAG:**
   * Criar a pasta `base_conhecimento/` com pelo menos **5 documentos públicos** (sobre o dataset California Housing, metodologia do censo da Califórnia, dicionário das variáveis, interpretação das métricas de regressão, limites éticos/técnicos do modelo, etc.) e seus metadados de fonte.
   * Implementar o pipeline RAG com etapas explícitas: **carregamento $\rightarrow$ *chunking* $\rightarrow$ *embeddings* $\rightarrow$ armazenamento $\rightarrow$ recuperação $\rightarrow$ geração**, discutindo no relatório a diferença entre busca lexical (ex.: BM25/TF-IDF) e similaridade semântica (embeddings densos).
   * Avaliar a recuperação com pelo menos **8 consultas**, registrando a evidência esperada, os trechos recuperados e uma métrica de recuperação (*Hit@k*, *Recall@k* ou *Precision@k*).
2. **Especificação e Implementação de Ferramentas (*Tools*):**
   * **Ferramenta 1 – Recuperação RAG:** Busca trechos relevantes na base de conhecimento com seus respectivos identificadores de fonte.
   * **Ferramenta 2 – Previsão Habitacional (`predict_california_housing`):** Consulta o modelo vencedor da Q2. Deve ter contrato estrito (nome, descrição, parâmetros, tipos, validações de limites documentados para `MedInc`, `HouseAge`, `Latitude`, `Longitude`, etc., retorno estruturado e política de erro para argumentos ausentes, inválidos ou fora de faixa). É proibido `eval`/`exec`, comandos de shell ou queries livres.
3. **Execução no Padrão ReAct (*Reason + Act*):**
   * Demonstrar o ciclo observável completo: *Decisão do Agente $\rightarrow$ Chamada de Ferramenta $\rightarrow$ Argumentos $\rightarrow$ Resultado (Observação) $\rightarrow$ Nova Decisão $\rightarrow$ Resposta Final com citação de fontes (ou abstenção caso não haja evidência)*.
   * Explicar como falhas na ferramenta, retornos vazios ou argumentos inválidos impactam o fluxo.
4. **Contrato MCP (*Model Context Protocol*):**
   * Documentar como uma das ferramentas seria exposta via MCP, apresentando exemplos de payloads JSON para `tools/list` e `tools/call` e identificando os papéis de **Host**, **MCP Client** e **MCP Server**.

---

### Passo 5: Q5 (2,0 pontos) – LangGraph, Push-Pull, Testes e Guardrails
Estruturação formal do agente da Q4 como um grafo de estados seguro e testável offline.

**O que será feito:**
1. **Grafo de Estado (`LangGraph` ou implementação equivalente):**
   * Definir um **Estado (`State`)** tipado contendo pelo menos: `pergunta`, `modo_iniciacao` (`pull` ou `push`), `historico_chamadas`, `resultados_recuperados`, `previsao`, `erros`, `iteracoes` e `resposta_final`.
   * Implementar os **6 nós explícitos**:
     1. Validar entrada e verificar escopo da pergunta;
     2. Recuperar contexto e decidir se há evidência suficiente;
     3. Chamar o LLM (ou componente de decisão);
     4. Executar ferramentas autorizadas;
     5. Retornar observações ao próximo ciclo;
     6. Produzir resposta final (ou abstenção) e encerrar.
   * Incluir **aresta condicional** e **ciclo com limite máximo de iterações** (`max_iterations`).
2. **Modos de Operação Pull e Push:**
   * **Modo Pull:** O usuário inicia fazendo uma pergunta e conduzindo a investigação.
   * **Modo Push:** Um evento simulado/agendado (ex.: chegada de um novo registro de bloco censitário da Califórnia ou alerta de um detector de anomalia/preço extremo) dispara automaticamente o grafo para analisar o registro e gerar um laudo/relatório.
   * Explicar no relatório o que muda no gatilho, no estado inicial e na política de repetição entre Push e Pull.
3. **Guardrails (em pelo menos 2 pontos do fluxo):**
   * *Guardrail de Entrada:* Validação de entrada, bloqueio de *prompt injection* e restrição de escopo (rejeitar conselhos financeiros/jurídicos ou temas fora do domínio).
   * *Guardrail de Ferramentas:* Autorização (somente ferramentas *read-only* na *allowlist*) e validação estrita de parâmetros.
   * *Guardrail de Saída:* Verificação da resposta final para impedir alucinações sem citação de fonte ou vazamento de dados internos.
4. **Suíte de Testes Automatizados e Determinísticos (`tests/`):**
   * Implementar um *harness* de testes que rode **100% offline** (usando *mocks*/simuladores determinísticos de LLM, embeddings, recuperador e ferramentas) com **pelo menos 8 casos de teste** cobrindo:
     1. Pergunta respondível vs. pergunta sem evidência suficiente (abstenção);
     2. Pergunta que exige combinar pelo menos 2 trechos;
     3. Chamada correta de ferramenta vs. chamada com argumentos inválidos;
     4. Falha ou retorno vazio de ferramenta;
     5. Entrada fora do escopo;
     6. Instrução maliciosa (*prompt injection*) na entrada ou no documento recuperado;
     7. Respeito ao limite máximo de iterações/chamadas;
     8. Execução em modo **Push** e em modo **Pull**.

---

### Passo 6: Q6 (1,5 ponto) – Observabilidade, Avaliação e Síntese Final
Instrumentação do sistema para inspeção, medição do desempenho global e síntese dos aprendizados da disciplina.

**O que será feito:**
1. **Observabilidade (*Tracing*, *Logging* e Métricas):**
   * Instrumentar o código para gerar um **trace hierárquico** completo de uma execução (mostrando transições de nós do grafo, chamadas ao LLM, recuperação RAG, execução de ferramentas, latência por etapa e resultado).
   * Distinguir conceitualmente e na prática: **Tracing** (trajetória de uma execução), **Logging** (eventos discretos/alertas/erros) e **Métricas** (agregações quantitativas de várias execuções).
2. **Avaliação Sistemática (sobre o conjunto de consultas da Q4):**
   * Reportar pelo menos:
     1. **Métrica de resultado:** taxa de respostas corretas, taxa de abstenção correta ou erro da previsão;
     2. **Métrica de recuperação:** *Hit@k*, *Recall@k* ou precisão;
     3. **Métrica de processo:** número médio de chamadas de ferramenta, chamadas desnecessárias, respeito ao limite de iterações e latência.
   * Realizar a **análise detalhada de 1 execução bem-sucedida e de 1 falha**.
   * Discutir custo, variabilidade, limitações da avaliação e propor **uma melhoria priorizada** (sem depender exclusivamente de *LLM-as-judge*).
3. **Síntese Conceitual Final:**
   * Comparar os três paradigmas construídos no trabalho:
     * **Aprendizado de Máquina:** estado = vetor de características; modelo aprende uma função $f(x) = y$.
     * **Modelagem de Linguagem:** contexto = sequência de tokens; modelo estima distribuição condicional $P(w_t \mid w_{1:t-1})$.
     * **Agente:** estado = contexto + memória + ferramentas + observações + controle de fluxo em grafo.
   * Discutir qual nível oferece maior flexibilidade, qual oferece maior controle experimental e por que **flexibilidade e garantias formais estão em tensão**.

---

## 4. Checklist dos Entregáveis (`GCC1734_T3_SEU_NOME_COMPLETO.zip`)

| Item | Descrição |
| :--- | :--- |
| `projeto.py` | Ponto de entrada principal que executa as etapas do projeto (ML, modelo de linguagem, RAG e agente). |
| Módulos (`ml/`, `linguagem/`, `agente/`, etc.) | Código-fonte modularizado e organizado em pacotes Python. |
| `base_conhecimento/` | Ao menos **5 documentos públicos** sobre o domínio (California Housing / regressão) + metadados das fontes. |
| `tests/` | *Harness* de testes, simuladores determinísticos (sem rede) e os $\ge 8$ casos de teste exigidos na Q5. |
| `README.md` | Objetivo, limites do sistema, requisitos, instalação, preparação da base e comandos para rodar experimento, agente e testes. |
| `requirements.txt` (e `.env.example`) | Dependências Python com versões fixadas (e exemplo de variáveis de ambiente sem chaves reais, se usar LLM remoto). |
| `RESPOSTAS.md` | Relatório técnico completo organizado nas seções **Q1 a Q6**, com tabelas, gráficos, diagramas, arquivos relevantes e a **minutagem exata (`MM:SS`)** de cada item no vídeo. |
| `LEIAME.txt` | Arquivo contendo o link do vídeo explicativo (20 a 30 minutos). |
