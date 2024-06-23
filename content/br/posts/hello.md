---
title: "Escolhendo um modelo de classificação SVM x KNN"
date: 2024-06-18T11:30:03+00:00
tags: ["machine_learning", "KNN", "SVM"]
description: "description"
weight:
slug: ""
draft: false 
comments: true
reward: true 
mermaid: true 
showToc: true 
TocOpen: false 
hidemeta: true 
disableShare: false 
showbreadcrumbs: true 
cover:
    image: "![alt text](image.png)" 
    caption: "" 
    alt: ""
    relative: true
---
# Hello


## sobre o dataset fashion mnist

O dataset fashion mnist e um dataset de imagens em escala 28x28 em tons cinza.

O dataset possui 785 colunas 28x28=784, portanto das 785 colunas , 784 são as representações de pixels da imagem a coluna que sobra é o label que representa aquela imagem

### Categorias de imagens

| Classe | Label |
| --- | --- |
| Camiseta/Top  | 0 |
| Calça | 1 |
| Suéter | 2 |
| Vestido | 3 |
| Colete | 4 |
| Sandália | 5 |
| Camisa | 6 |
| Tênis | 7 |
| bolsa | 8 |
| Botas | 9 |

Exemplos de imagens:

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/641214df-7b93-4272-b430-4373a6f61fea/ea41fd62-4f01-4fa4-a1ea-112731e168a0/Untitled.png)

> [Images fashion_mnist_dataset](https://www.google.com/url?sa=i&url=https%3A%2F%2Fwww.researchgate.net%2Ffigure%2FSample-images-from-Fashion-MNIST-dataset_fig2_342801790&psig=AOvVaw13drb-c6F4dMvfSipxXSuI&ust=1718758996522000&source=images&cd=vfe&opi=89978449&ved=0CBEQjRxqFwoTCKieu63644YDFQAAAAAdAAAAABAJ)
> 

### code

```python
from sklearn.datasets import fetch_openml
df = fetch_openml('Fashion-MNIST', version=1, cache=True)
X = df.data
y = df.target.astype(int)
```

## Pré-processamento dos Dados

No pré-processamento dos dados foi utilizado duas técnicas de pré-processamento

### Normalização dos dados

A normalização dos dados para padronizar os dados deixando a media da coluna igual a zero e o desvio padrão igual 1, para isso foi utilizado o standard scaller

Exemplo antes e após normalizar os dados utilizando  standard scaller

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/641214df-7b93-4272-b430-4373a6f61fea/64241039-c7ee-46b3-8524-7626aca3b076/Untitled.png)

# Redução de dimensionalidade(PCA)

Foi utilizado o PCA para reduzir a dimensionalidade das colunas dado que o treinamento dos modelos estava muito demorado.

Antes do PCA o número de colunas do treinamento eram 784(numero de píxeis). Após aplicar a técnica do PCA para uma variância de 85% dos dados esse numero caiu ~80 colunas.

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/641214df-7b93-4272-b430-4373a6f61fea/3b29fa25-bdd9-493a-9d0e-6ebeace63d0e/Untitled.png)

## code

```python
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import pandas as pd

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
pca = PCA(n_components=0.85, random_state=42)
X_pca = pca.fit_transform(X_scaled)
X_pca_df = pd.DataFrame(X_pca)
```

## Treinamento dos Modelos

Eu escolhi dois modelos para realizar os testes comparativos:

- SVM
- KNN

### Divisão de treino e teste

Para a divisão de treino e teste dos modelos foi verificado o balanceamento dos dados, conjuntos de treino desbalanceados podem levar a modelos com viés. E conjuntos de teste desbalanceados podem levar a métricas de classificação com viés.

Como podem ver abaixo, tanto o conjunto de teste quanto o conjunto de treino estão relativamente balanceadas, com algumas pequenas diferenças entre eles, porem nada que levasse a um viés.