import 'https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.22.0/dist/tf.min.js';
import { workerEvents } from '../events/constants.js';
import { recommendationModel } from './recommendationModel.js';

console.log('Model training worker initialized');

async function trainModel({ users }) {
    console.log('Training model with users:', users);

    postMessage({
        type: workerEvents.progressUpdate,
        progress: { progress: 50 }
    });

    await recommendationModel.reload(users);

    postMessage({ type: workerEvents.progressUpdate, progress: { progress: 100 } });
    postMessage({ type: workerEvents.trainingComplete });
}

function recommend(user) {
    if (!recommendationModel.state || !recommendationModel.trainedModel) return; //não treina, não recomenda, pois não há dados para treinamento, nem modelo treinado

    const dt = recommendationModel.state;

    const userVector = recommendationModel
    .encodeUserVector(user, dt.dimentions, dt.users.minAge, dt.users.maxAge, dt.products.categoryCount, dt.products.colorCount)
    .dataSync();

    /*
    Em aplicações reais:
    armazena todos os vetores de produtos em um BD vetorial (Postgre, Neo4j ou Pinecone).
    consulta: encontre 200 produtos mais próximos do vetor do usuário, 
    usando uma métrica de similaridade (cosine similarity, euclidean distance, etc).
    execute model.predict() apenas nesses 200 produtos, para reduzir o custo computacional.

    cria pares de entrada: 
    para cada produto, concatena o vetor do usuário com o vetor codificado do produto, 
    formando um vetor de entrada para o modelo de recomendação.

    por quê? porque o modelo de recomendação foi treinado para receber como entrada
     um vetor que representa o usuário concatenado com o vetor do produto, e então prever 
     a probabilidade de o usuário gostar do produto. O modelo prevê o "score de compatibilidade" 
     para cada par (usuário, produto).
    */
    const inputs = dt.products.list.map((product) => {
        const prdVector = product.vectors.concat.dataSync();
        return [...userVector, ...prdVector];
    });

    //concerte todos esses pares (user, prod) em um único tensor. Formato [numProdutos, inputDim]
    const inputTensor = tf.tensor2d(inputs);
    /*
    roda a rede neural treinada em todos os pares (user,prod) de uma vez, para obter os scores de compatibilidade.
    O resultado é uma pontuação para cada produtos, entre 0 e 1
    Quanto maior, maior a probabilidade de o usuário gostar do produto.

    */
    const predictions = recommendationModel.trainedModel.predict(inputTensor);

    //extrai as pontuações para um Array JS normal.
    const scores = predictions.dataSync();
    
    const recommendations = dt.products.list.map((product, index) => ({
        ...product,
        name: product.name,
        score: scores[index] //previsão do modelo para este produto
    }));

    const sortedRecommendations = recommendations.sort((a, b) => b.score - a.score);

    postMessage({
        type: workerEvents.recommend,
        user,
        recommendations: sortedRecommendations
    });
}

const handlers = {
    [workerEvents.trainModel]: trainModel,
    [workerEvents.recommend]: ({ user }) => recommend(user, {})
};

self.onmessage = (event) => {
    const { action, ...payload } = event.data;
    const handler = handlers[action];

    if (typeof handler === 'function') {
        handler(payload);
    }
};
