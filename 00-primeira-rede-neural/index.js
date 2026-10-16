import tf from '@tensorflow/tfjs-node';

async function trainModel(inputXs, outputYs) {
    const model = tf.sequential();
    //primeira cada da rede
    // camada de 7 posições (idade normalizada, 3 cores, 3 localizações)
    //80 neurônios (pca base treino)
    //quanto mais, mais complexidade a rede pode aprender e mais processamento vai usar
    //Relu age como filtro: retém só dados interessantes. Quando dado que chega ao neurônio é positiva, passa pra frente. Quando zero ou negativa, não é usada, não serve para nad.
    model.add(tf.layers.dense({ inputShape: [7], units: 80, activation: 'relu' }));
    //saída 3 neurônios; 1 para premium, medium, basic;
    //activation softmax: pega os 3 neurônios e transforma em probabilidade, somando 1.0
    model.add(tf.layers.dense({ units: 3, activation: 'softmax' }));
    //compilando o modelo
    //optimizer: algoritmo que vai ajustar os pesos da rede neural para melhorar a acurácia
    //loss: função de perda, mede o quão ruim está a previsão do modelo. Quanto menor, melhor.
    //adam ajusta os pesos de forma eficiente e inteligente
    //aprende com histórico de erros e acertos
    //loss: 'categoricalCrossentropy' - compara o que modelo acha (scores de cada categoria) com resposta certa
    //metrics: ['accuracy']  - quanto mais distante da previsão do modelo da resposta correta, maior a perda (loss). Quanto mais próximo, menor a perda.
    model.compile({ optimizer: 'adam', loss: 'categoricalCrossentropy', metrics: ['accuracy'] });
    //treinando o modelo
    /**
     * verbose:0 -  desabilita logs de treino e usa so callback
     * epochs: 100 -  qtd vezes rodar no dataset de treino
     * shuffle: true - embaralha dados para evitar que rede aprenda ordem dos dados, evitar viés
     */
    await model.fit(inputXs, outputYs, { verbose:0, epochs: 100, shuffle: true, callbacks: {
        onEpochEnd: (epoch, logs) => {
            console.log(`Epoch ${epoch + 1}: loss = ${logs.loss.toFixed(4)}, accuracy = ${logs.acc.toFixed(4)}`);
        } 
    }
});
    return model;
}   
// Exemplo de pessoas para treino (cada pessoa com idade, cor e localização)
// const pessoas = [
//     { nome: "Erick", idade: 30, cor: "azul", localizacao: "São Paulo" },
//     { nome: "Ana", idade: 25, cor: "vermelho", localizacao: "Rio" },
//     { nome: "Carlos", idade: 40, cor: "verde", localizacao: "Curitiba" }
// ];

// Vetores de entrada com valores já normalizados e one-hot encoded
// Ordem: [idade_normalizada, azul, vermelho, verde, São Paulo, Rio, Curitiba]
// const tensorPessoas = [
//     [0.33, 1, 0, 0, 1, 0, 0], // Erick
//     [0, 0, 1, 0, 0, 1, 0],    // Ana
//     [1, 0, 0, 1, 0, 0, 1]     // Carlos
// ]

// Usamos apenas os dados numéricos, como a rede neural só entende números.
// tensorPessoasNormalizado corresponde ao dataset de entrada do modelo.
const tensorPessoasNormalizado = [
    [0.33, 1, 0, 0, 1, 0, 0], // Erick
    [0, 0, 1, 0, 0, 1, 0],    // Ana
    [1, 0, 0, 1, 0, 0, 1]     // Carlos
]

// Labels das categorias a serem previstas (one-hot encoded)
// [premium, medium, basic]
const labelsNomes = ["premium", "medium", "basic"]; // Ordem dos labels
const tensorLabels = [
    [1, 0, 0], // premium - Erick
    [0, 1, 0], // medium - Ana
    [0, 0, 1]  // basic - Carlos
];

// Criamos tensores de entrada (xs) e saída (ys) para treinar o modelo
const inputXs = tf.tensor2d(tensorPessoasNormalizado)
const outputYs = tf.tensor2d(tensorLabels)

const model = trainModel(inputXs, outputYs);
