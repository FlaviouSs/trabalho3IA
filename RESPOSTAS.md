# Relatório – Trabalho 3 (GCC1734 – Inteligência Artificial)

- **Disciplina:** GCC1734 – Inteligência Artificial (CEFET/RJ)
- **Matrícula:** 2221875BCC
- **Dataset atribuído:** California Housing (ID = 4, Regressão)

---

## Q1 – Dados, Preparação e Protocolo Experimental

### Arquivos relevantes
- `ml/data_pipeline.py` — carregamento, divisão aleatória, pipeline de pré-processamento e resumo estatístico.
- `requirements.txt` — dependências e versões fixadas para reprodutibilidade.



### 1.1 Descrição das entradas, alvo, tarefa e riscos de interpretação
Pela regra da matrícula (`2221875BCC`), $\text{ID}(2221875) = [(2221875 \bmod 4) + 1] = 4$, atribuindo o conjunto **California Housing**. Ele é carregado com `fetch_california_housing(as_frame=True)` na função `load_dataset()`, preservando os dados como DataFrame e Series do *pandas*.

- **Tipo de tarefa:** **Regressão supervisionada**, cujo objetivo é estimar uma variável numérica contínua ($y \in \mathbb{R}$) a partir de 8 atributos numéricos contínuos.
- **Variável-alvo (target):** `MedHouseVal` — valor mediano das residências no grupo de blocos censitários, expresso em centenas de milhares de dólares.
- **Variáveis de entrada (features):**

| Atributo | Descrição |
| :--- | :--- |
| `MedInc` | Renda mediana no grupo de blocos (em dezenas de milhares de US$). |
| `HouseAge` | Idade mediana das residências no grupo de blocos (em anos). |
| `AveRooms` | Número médio de cômodos por domicílio. |
| `AveBedrms` | Número médio de quartos por domicílio. |
| `Population` | População total residente no grupo de blocos. |
| `AveOccup` | Número médio de moradores por domicílio. |
| `Latitude` | Latitude geográfica do grupo de blocos. |
| `Longitude` | Longitude geográfica do grupo de blocos. |

**Riscos de interpretação e limites do domínio:**

1. **Granularidade agregada:** cada registro representa um *block group* censitário, e não um imóvel individual. Não é válido interpretar a previsão como avaliação direta de uma casa específica.
2. **Defasagem temporal:** os dados provêm do Censo dos EUA de 1990, portanto os valores monetários não refletem o mercado imobiliário atual.

### 1.2 Divisão em treino, validação e teste
A divisão é realizada na função `split_dataset()` (`ml/data_pipeline.py`) imediatamente após o carregamento dos dados brutos e **antes** de qualquer transformação ou cálculo estatístico.

- **Proporções adotadas:** **60% treino / 20% validação / 20% teste**.
  - Como `train_test_split` realiza uma partição binária por chamada, a divisão foi feita em duas etapas:
    1. Separação de 20% para teste (`test_size=0.2`), restando 80% para desenvolvimento;
    2. Separação de `val_relative = 0.2 / (1.0 - 0.2) = 0.25` sobre os 80% restantes, resultando em 60% do total para treino e 20% para validação.
  - **Justificativa das proporções:** com 20.640 instâncias, a divisão `60/20/20` mantém 12.384 exemplos para ajuste dos modelos e 4.128 exemplos independentes tanto na validação (para comparação de modelos e hiperparâmetros na Q2) quanto no teste final, oferecendo estabilidade estatística nas métricas de erro sem comprometer o aprendizado.
- **Semente e embaralhamento:** utilizou-se `shuffle=True` (essencial para desfazer ordenação geográfica original dos blocos censitários) e `random_state = SEED = 2221875` (parte numérica da matrícula) em ambas as chamadas, assegurando reprodutibilidade determinística.
- **Verificação de integridade da divisão:** a função valida que os tamanhos são positivos e somam menos que 1.0, e executa `assert` confirmando que os conjuntos de índices de treino, validação e teste são **mutuamente disjuntos** (`isdisjoint`) e que sua soma equivale exatamente ao total original (20.640).

---

### 1.3 Pipeline de pré-processamento
Na função `build_preprocessor()`, foi construído um `Pipeline` do *scikit-learn* contendo a etapa de padronização:

```python
Pipeline(steps=[("scaler", StandardScaler())])
```

Configurado com `.set_output(transform="pandas")` para preservar os nomes das 8 colunas e os índices originais após a transformação.

- **Justificativa da padronização:** as estatísticas descritivas do treino evidenciam forte discrepância de escalas — `Population` atinge 28.566 (média 1.427,44), enquanto `MedInc` tem média 3,8564 e `AveBedrms` tem média 1,0948. A padronização ($z = \frac{x - \mu}{\sigma}$) centraliza cada variável em média 0 e desvio-padrão 1, etapa fundamental para modelos lineares e regularizados.
- **Ausência de imputação:** como a checagem de integridade confirmou 0 valores ausentes no conjunto, não foi necessário incluir `SimpleImputer` no pipeline.
- **Isolamento do ajuste (fit vs. transform):** no fluxo de `ml/data_pipeline.py`, `preprocessor.fit_transform(X_train)` é chamado **exclusivamente** no conjunto de treino, calculando $\mu$ e $\sigma$ apenas sobre as 12.384 amostras de treino. Em seguida, `X_validation` e `X_test` passam **apenas** por `preprocessor.transform(...)`, reutilizando os parâmetros congelados do treino.

**Comprovação numérica da padronização sem ajuste no teste/validação:**

| Atributo | `train_mean` | `train_std` (`ddof=0`) | `val_mean` | `test_mean` |
| :--- | ---: | ---: | ---: | ---: |
| `MedInc` | -0.0000 | 1.0000 | 0.0189 | 0.0190 |
| `HouseAge` | -0.0000 | 1.0000 | 0.0019 | -0.0002 |
| `AveRooms` | 0.0000 | 1.0000 | 0.0195 | 0.0134 |
| `AveBedrms` | 0.0000 | 1.0000 | 0.0074 | 0.0122 |
| `Population` | 0.0000 | 1.0000 | -0.0145 | 0.0056 |
| `AveOccup` | -0.0000 | 1.0000 | -0.0130 | 0.0012 |
| `Latitude` | -0.0000 | 1.0000 | 0.0248 | -0.0111 |
| `Longitude` | 0.0000 | 1.0000 | -0.0281 | 0.0037 |

As médias exatamente nulas (0.0) e desvios unitários (1.0) no treino, contrastados com médias próximas de zero porém não exatas na validação e no teste, comprovam que o `StandardScaler` não foi ajustado com dados de validação ou teste.

---

### 1.4 Cardinalidade, dimensões e estatísticas descritivas
Os resultados abaixo foram obtidos executando `python ml/data_pipeline.py` (funções `compute_summary` e `print_summary`):

- **Cardinalidade e dimensões:**
  - Dataset bruto (features + target): **(20640, 9)**
  - Treino (`X_train`, `y_train`): **(12384, 8)** e **(12384,)** (60%)
  - Validação (`X_validation`, `y_validation`): **(4128, 8)** e **(4128,)** (20%)
  - Teste (`X_test`, `y_test`): **(4128, 8)** e **(4128,)** (20%)
- **Integridade dos dados brutos:**
  - Valores ausentes (`NaN`): **0** em todas as 9 colunas.
  - Linhas duplicadas: **0**.

**Estatísticas descritivas das variáveis de entrada (conjunto de treino, n = 12.384):**

| Variável | `mean` | `std` | `min` | `25%` | `50%` | `75%` | `max` |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `MedInc` | 3.8564 | 1.8773 | 0.4999 | 2.5562 | 3.5344 | 4.7356 | 15.0001 |
| `HouseAge` | 28.6352 | 12.5996 | 2.0000 | 18.0000 | 29.0000 | 37.0000 | 52.0000 |
| `AveRooms` | 5.4134 | 2.3639 | 0.8462 | 4.4410 | 5.2259 | 6.0468 | 132.5333 |
| `AveBedrms` | 1.0948 | 0.4738 | 0.3333 | 1.0058 | 1.0483 | 1.0986 | 34.0667 |
| `Population` | 1427.4365 | 1101.2691 | 3.0000 | 793.0000 | 1172.0000 | 1729.0000 | 28566.0000 |
| `AveOccup` | 3.0993 | 12.0999 | 0.6923 | 2.4290 | 2.8165 | 3.2827 | 1243.3333 |
| `Latitude` | 35.6260 | 2.1415 | 32.5400 | 33.9300 | 34.2600 | 37.7200 | 41.9500 |
| `Longitude` | -119.5599 | 2.0048 | -124.3500 | -121.7800 | -118.4900 | -118.0000 | -114.4900 |

**Distribuição da variável-alvo (`MedHouseVal`) nas três partições:**

| Partição | `count` | `mean` | `std` | `min` | `25%` | `50%` | `75%` | `max` |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `train` | 12384 | 2.0564 | 1.1464 | 0.1500 | 1.1990 | 1.7875 | 2.6165 | 5.0000 |
| `validation` | 4128 | 2.0794 | 1.1711 | 0.2250 | 1.1768 | 1.7915 | 2.6712 | 5.0000 |
| `test` | 4128 | 2.0941 | 1.1591 | 0.1500 | 1.2157 | 1.8330 | 2.7095 | 5.0000 |

As três partições apresentam médias (`~2,06–2,09`), medianas (`~1,79–1,83`) e desvios-padrão (`~1,15–1,17`) consistentes entre si, confirmando que a divisão aleatória preservou a distribuição do alvo.

---

### 1.5 Prevenção de vazamento de dados, reprodutibilidade e fluxo experimental
- **Prevenção de vazamento de dados (*data leakage*):**
  1. A divisão em treino, validação e teste ocorre antes de qualquer estatística ou transformação.
  2. Os índices das três partições são verificados por `assert` (`isdisjoint`), garantindo ausência de duplicação entre conjuntos.
  3. O método `.fit()` do pré-processador é restrito a `X_train`; validação e teste recebem apenas `.transform()`, e o conjunto de teste permanece reservado para avaliação única ao final da Q2.
- **Reprodutibilidade:**
  1. Semente aleatória fixa (`SEED = 2221875`) aplicada na divisão dos dados (e nos modelos estocásticos da Q2).
  2. Versões exatas das bibliotecas registradas em `requirements.txt`.
  3. Fluxo executado de forma determinística via script Python (`ml/data_pipeline.py`).

**Fluxo experimental do projeto:**

![Fluxo experimental do projeto](docs/mermaid-diagram-2026-11-19-151405.png)

---

## Q2 – Modelos e Aprendizado Supervisionado

> *[A preencher após implementação da Q2]*

---

## Q3 – Modelagem de Linguagem e Alinhamento

> *[A preencher após implementação da Q3]*

---

## Q4 – RAG, Ferramentas, ReAct e MCP

> *[A preencher após implementação da Q4]*

---

## Q5 – LangGraph, Push-Pull, Testes e Guardrails

> *[A preencher após implementação da Q5]*

---

## Q6 – Observabilidade, Avaliação e Síntese Final

> *[A preencher após implementação da Q6]*
