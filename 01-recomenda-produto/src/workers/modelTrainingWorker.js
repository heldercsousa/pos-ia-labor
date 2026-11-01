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
    console.log('recommendationModel.state', recommendationModel.state);  
    
    const productVectors = recommendationModel.state.catalog
    .map(product => {
        return {
            name: product.name, 
            meta: {...product},
            vector: product.vectors.concat.dataSync() // Use dataSync() to get the underlying data as a typed array
        };
    }); 

    console.log('productVectors', productVectors);
    
    postMessage({
        type: workerEvents.trainingLog,
        epoch: 1,
        loss: 1,
        accuracy: 1
    });

    setTimeout(() => {
        postMessage({ type: workerEvents.progressUpdate, progress: { progress: 100 } });
        postMessage({ type: workerEvents.trainingComplete });
    }, 1000);
}

function recommend(user, ctx) {
    console.log('will recommend for user:', user);
    // postMessage({
    //     type: workerEvents.recommend,
    //     user,
    //     recommendations: []
    // });
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
