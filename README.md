# BOSS Textract NER

Este projeto tem como objetivo realizar a extração de entidade nomeada (NER)
o objetivo do seu desenvolvido foi devido a necessidade de aperfeiçoar a extração de organizações nos registros presentes no BOSS

### 📕 Sumário
- [BOSS Textract NER](#boss-textract-ner)
    - [📕 Sumário](#-sumário)
  - [🛠️ Softwares necessários](#️-softwares-necessários)
  - [💻 Como Instalar](#-como-instalar)
  - [☄️ Como Usar (Notebook)](#️-como-usar-notebook)
  - [✅ Como Usar (CLI)](#-como-usar-cli)
    - [Argumentos do Comando Predict](#argumentos-do-comando-predict)
  - [🤖 Como treinar?](#-como-treinar)
    - [Preparando os dados](#preparando-os-dados)
    - [Realizando treinamento](#realizando-treinamento)
    - [Argumentos do Comando Train](#argumentos-do-comando-train)
  - [Explicando `max_variation`](#explicando-max_variation)
    - [Problemas de `max_variation`](#problemas-de-max_variation)
  - [🎉 Colaboradores](#-colaboradores)


## 🛠️ Softwares necessários

* [UV](https://docs.astral.sh/uv/)
* 
## 💻 Como Instalar
```sh
git clone https://github.com/RyanPrado/BOSS-Textract-NER.git
cd BOSS-Textract-NER
uv pip install -r pyproject.toml
```
## ☄️ Como Usar (Notebook)
Para utilizar a aplicação através dos notebooks [jupyter](https://jupyter.org), basta acessar o arquivo em [`/notebook/app.ipynb`](./notebook/app.ipynb), basta editar as variáveis apresentadas no documento e utiliza-lo.
## ✅ Como Usar (CLI)
Para utilizar a ferramenta de predição basta utilizar o comando `uv run boss_textract predict -h`
```sh
uv run boss_textract predict -h
```
### Argumentos do Comando Predict
| Argumento       | Tipo      | Padrão  | Obrigatório |
| --------------- | --------- | :-----: | :---------: |
| --data          | `File`    |  *N/A*  |      ✅      |
| --model         | `Folder`  |  *N/A*  |      ✅      |
| --output        | `File`    |  *N/A*  |      ✅      |
| --src_col       | `String`  |  *N/A*  |      ⬜️      |
| --out_col       | `String`  |  *N/A*  |      ⬜️      |
| --start_header  | `Integer` |  *N/A*  |      ⬜️      |
| --sep           | `String`  |  *N/A*  |      ⬜️      |
| --max_variation | `Integer` |  **0**  |      ⬜️      |
| --encoding      | `String`  | *UTF-8* |      ⬜️      |
| --gpu_id        | `Integer` | **-1**  |      ⬜️      |
* `--data*` - Arquivo `.csv` para o modelo fazer predição;
* `--model*` - Diretório do modelo que realizaram a predição;
* `--output*` - Diretório aonde o arquivo de saída será colocado (deve-se colocar no nome do arquivo com a extensão `.csv`);
* `--src_col` - Coluna de origem dentro do arquivo `--data`;
* `--out_col` - Coluna de resposta (será criada a direita da coluna de origem);
* `--start_header` - Em casos que a primeira linha não seja a linha inicial da tabela, aponte em qual linha se inicia o cabeçalho;
* `--sep` - Separador utilizado no arquivo `--data`;
* `--max_variation` - Em casos de tentativa de padronização pela variação, selecione a quantidade maxima de variação para que uma empresa possa sofrer ajuste[(?)](#explicando-max_variation).
* `--encoding` - Tipo de encoding do arquivo `--data` & `--eval`;
* `--gpu_id` - Qual o ID da GPU a ser utilizado `-1` simboliza utilizar o processador;


## 🤖 Como treinar?
Para realizar o treinamento do modelo, uma boa base de dados deve ser acumulada e polida, alguns polimentos são executados pelo próprio scripts, mas outros possa ser necessário serem feitas pelo próprio usuário.
O arquivo de treino deverá ser uma matrix com duas ou mais colunas, sendo elas uma coluna de origem e as outras serão os tipos de [`labels`](https://spacy.io/api/entityrecognizer#add_label), existentes no texto.

### Preparando os dados
Exemplos de dados:
| SOURCE                                           | ORGS                |
| ------------------------------------------------ | ------------------- |
| `ABC0000000 / DEZ-24 / AB213123 XPTO COOP LTDA.` | `XPTO COOP LTDA`    |
| `NF  13 XPTH SERVICE LTDA 12/1997`               | `XPTH SERVICE LTDA` |
| `...`                                            |                     |
Lembre-se que a origens não podem ter repetição, ou seja, certifique-se de que sempre haja uma variação mesmo que minima em cada item semelhante da origem.

### Realizando treinamento
Para utilizar a ferramenta de predição basta utilizar o comando `uv run boss_textract train -h`

```sh
uv run boss_textract train -h
```

### Argumentos do Comando Train
| Argumento        | Tipo                                                            |         Padrão          | Obrigatório |
| ---------------- | --------------------------------------------------------------- | :---------------------: | :---------: |
| --data           | `File`                                                          |          *N/A*          |      ✅      |
| --config         | [`SpaCy Config File`](https://spacy.io/api/data-formats#config) |          *N/A*          |      ✅      |
| --eval           | `File`                                                          |          *N/A*          |      ⬜️      |
| --model          | `Folder`                                                        |          *N/A*          |      ⬜️      |
| --src_col        | `String`                                                        |          *N/A*          |      ⬜️      |
| --res_col        | `String`                                                        |          *N/A*          |      ⬜️      |
| --sep            | `String`                                                        |           *;*           |      ⬜️      |
| --min_samples    | `Integer`                                                       |          **5**          |      ⬜️      |
| --epochs         | `Integer`                                                       |         **10**          |      ⬜️      |
| --train_size     | `Float`                                                         |         **0.8**         |      ⬜️      |
| --dropout        | `Float`                                                         |          *N/A*          |      ⬜️      |
| --eval_frequency | `Integer`                                                       |          *N/A*          |      ⬜️      |
| --output         | `Folder`                                                        | *./models/boss-ner-X-X* |      ⬜️      |
| --encoding       | `String`                                                        |         *UTF-8*         |      ⬜️      |
| --gpu_id         | `Integer`                                                       |         **-1**          |      ⬜️      |

* `--data*` - Arquivo `.csv` para o modelo treinar;
* `--config*` - Arquivo `.cfg` com as configurações do modelo;
* `--eval` - Arquivo para validação customizado (Deve possuir mesmas colunas do arquivo `--data`);
* `--model` - Diretório do modelo (Em caso de Transfer Learning);
* `--src_col` - Coluna de origem dentro do arquivo `--data`;
* `--res_col` - Colunas de resposta dentro do arquivo `--data` (Exemplo de uso "COLUMN2:ORG;COLUMN3:MISC");
* `--sep` - Separador utilizado no arquivo `--data`;
* `--min_samples` - Quantidade minímas de amostra de um respectivo item (para evitar overfitting);
* `--epochs` - Quantidade de épocas a serem treinadas;
* `--train_size` - Em caso de não utilizam de arquivo dedicado para validação, quantos % serão usados do `--data` para validação;
* `--dropout` - Quantos % de dropout para evitar overfitting no modelo;
* `--eval_frequency` - A cada quantos batchs serão feito uma avaliação do modelo (recomenda-se um maior valor para quando o dados de validação forem muitos);
* `--output` - Diretório de saída do modelo gerado;
* `--encoding` - Tipo de encoding do arquivo `--data` & `--eval`;
* `--gpu_id` - Qual o ID da GPU a ser utilizado `-1` simboliza utilizar o processador;

## Explicando `max_variation`
Está função tem como objetivo tentar criar uma padronização nas saídas das entidades nomeadas encontradas, porém deve-se atentar ao seu uso, pois em caso de baixa variação e má configuração deste campo, ele pode prejudicar a qualidade das informações, sua base de funcionamento é, ele tentar identificar um texto dentro de outro, por exemplo:

- **ABC ENTERPRISES LTDA** `#Empresa A`
- **ABC ENTERPRISES**      `#Empresa A`

Ambos os textos reference-se a mesma empresa, porém uma possui a escrita **LTDA** e outra não, afins de criar um padrão, então imagine que o primeiro item `ABC ENTERPRISES LTDA` será consultado na listas de nomes de empresas do respectivo arquivo, se ele encontrar outras empresas que possuam exatamente este texto, então ele adicionaria nas variações, neste nosso exemplo de 2 itens, nenhuma empresa possui exatamente está quantidade e exatos caracteres, visto que o segundo item está faltando LTDA, agora quando `ABC ENTERPRISES` for consultado, o sistema vai identificar que ele possui uma variação, que neste caso é o seu próprio nome com LTDA no final, ou seja, ele vai retornar uma variação, se o campo `max_variation` estiver configurado como ZERO, então o nome se manterá o mesmo sem assumir a sua variação, agora se neste caso estiver maior ou igual a 1, então `ABC ENTERPRISES` irá se transmutar para `ABC ENTERPRISES LTDA`

### Problemas de `max_variation`
Em alguns cenários existem variações genéricas demais, oque pode causar confusão no algorítimo, observe o exemplo a seguir:

- **ABC ENTERPRISES LTDA** `#Empresa A`
- **ABC ENTERPRISES**      `#Empresa A`
- **ABC COOPERATIVE S.A**  `#Empresa B`
- **ABC**                  `#Empresa B`

Observe que as empresas compartilham uma sintaxe de nome parecida, mesmo sendo empresas diferentes, neste cenário o `max_variation` pode se tornar um problema ao invés de uma solução, então utiliza-lo nesta situação é muito relativo, e em geral aumentar o número de variações aceitam pode só acarretar em mais problemas, a empresa com nome `ABC` ficou tão genérica que o `max_variation` será incapaz de dizer se ela pertence ao grupo `A` ou grupo `B`, se o valor de variação for menor que 2 então ela iria se manter com `ABC`, agora se for maior, possivelmente o nome de todas as empresas seria embaralhadas de maneira errônea.



## 🎉 Colaboradores
- [Ryan Cardoso do Prado](https://www.linkedin.com/in/ryan-prado/)
- [Gabriel Silva](https://www.linkedin.com/in/gabriel-silva-276908181/)
- [Brenda Victoria Prado](https://www.linkedin.com/in/brenda-victoria-prado/)