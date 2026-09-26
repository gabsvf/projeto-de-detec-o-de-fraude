**O problema**

O objetivo foi identificar transações fraudulentas. O problema é que existem muito mais transações normais do que fraudes, cerca de 99,8% contra 0,17%.
Por isso, não usei a acurácia como principal referência. Um modelo poderia acertar quase tudo simplesmente dizendo que nenhuma transação é fraude. Foquei principalmente em recall, precisão e F1 da classe fraude.

**Preparação dos dados**

Primeiro, analisei os dados e a quantidade de fraudes. Depois, criei uma variável usando o log do valor da transação, organizei os dados e fiz a separação entre treino e teste mantendo a mesma proporção de fraudes.

**Comparação dos modelos**

Comparei três modelos:
    Regressão Logística: precisão 87%, recall 62% e F1 72%.
    Random Forest: precisão 74%, recall 80% e F1 77%.
    Regressão Logística customizada: precisão 85%, recall 68% e F1 76%.
O Random Forest conseguiu encontrar mais fraudes, com 80% de recall, e também teve o maior F1 entre os três.

**Limiar e SHAP**

Também ajustei o limiar de decisão para tentar encontrar um equilíbrio melhor entre identificar fraudes e evitar falsos alarmes.
Com o SHAP, consegui visualizar quais variáveis mais influenciaram o modelo quando ele classificou uma transação como fraude.

**O que mudei em relação à Expert**

Segui a mesma ideia geral da Expert, mas montei o notebook do zero e fiz algumas adaptações. Comparei diferentes modelos, trabalhei com o desbalanceamento, ajustei o limiar de decisão e usei SHAP para entender melhor as decisões do modelo.
No final, meu foco foi não somente detectar fraudes, mas também entender por que o modelo considerou uma transação suspeita.
