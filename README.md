# SGS Textract NER

Este projeto tem como objetivo realizar a extração de entidade nomeada (NER)
o objetivo do seu desenvolvido foi devido a necessidade de aperfeiçoar a extração de organizações nos registros presentes no BOSS da SGS

## 🛠️ Recursos necessários

* [UV](https://docs.astral.sh/uv/)

## 💻 Como Instalar
```sh
uv install
```

## Como Usar

#TODO

## 🤖 Treinamento o Modelo
Para executar o treinamento do modelo é necessário realizar a preparação dos dados, para isto você precisa tê-los extraídos em `.csv` com as seguintes colunas:
* `GL_LINE_DESCRIPTION`: Descrição do registro por completo (como foi extraído do sistema);
* `NAME`: Nome da empresa que foi extraído do texto acima;

| GL_LINE_DESCRIPTION                              | NAME                |
| ------------------------------------------------ | ------------------- |
| `ABC0000000 / DEZ-24 / AB213123 XPTO COOP LTDA.` | `XPTO COOP LTDA`    |
| `NF  13 XPTH SERVICE LTDA 12/1997`               | `XPTH SERVICE LTDA` |
| `...`                                            |                     |

### Preparando os dados
Para realizar as a preparação dos dados para a forma com qual a [spaCy](https://spacy.io) precisa para realizar o treinamento do modelo, execute o comando abaixo:
```sh
uv run python -m src/prepare_train.py --data INPUT_CSV [--output OUTPUT_CSV]
```
Um arquivo de saída será retornado por padrão em [`data/prepared/train.csv`](./data/prepared/train.csv), você pode informar o argumento `--output` para definir o local de saída.

### Realizando o treino
Para realizar o treinamento do modelo execute o comando abaixo:
```sh
uv run python -m src/train.py --data INPUT_CSV [--epochs MAX_EPOCHS]
```
Você pode passar a quantidade de épocas que deseja, por padrão o valor a ser utilizado é o presente em `training.max_epochs` no arquivo de [configuração](./config.cfg).

## 🎉 Colaboradores
- [Brenda Victoria Prado](https://www.linkedin.com/in/brenda-victoria-prado/)
- [Ryan Cardoso do Prado](https://www.linkedin.com/in/ryan-cardoso-do-prado-17879819b/)