import 'https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.22.0/dist/tf.min.js';
import { workerEvents } from '../events/constants.js';

console.log('Model training worker initialized');
let _globalCtx = {};

let data = {
    minval : -1,
    maxval : -1,
    catalog : [],
    
    users : {
        midAge : -1,
        ageSum : {},
        ageCount : {},
        list : []
    },
    prepare : (catalogList, usrList) => {
        data.users.list = usrList.map((u,i) => {
            index = i,
            age = u.age
        });
        const ages = users.list.map(u => u.age);
        user.midAge = (Math.min(... ages) + Math.max(...ages))  / 2;
        catalog = catalogList.map((p,i) => {
            index = i,
            price = p.price,
            color = p.color,
            category = p.category
        });
        const prices = catalog.map(p => p.price);
        minval = Math.min(...prices);
        maxval = Math.max(...prices);
    },
    
};

async function makeContext(catalog, users) {
    data.prepare(catalog, users);
}

async function trainModel({ users }) {
    console.log('Training model with users:', users)
    postMessage({ type: workerEvents.progressUpdate, progress: { progress: 50 } });
    const catalog = await(await fetch('/data/products.json')).json();
    const context = makeContext(catalog,users);
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
    console.log('will recommend for user:', user)
    // postMessage({
    //     type: workerEvents.recommend,
    //     user,
    //     recommendations: []
    // });
}


const handlers = {
    [workerEvents.trainModel]: trainModel,
    [workerEvents.recommend]: d => recommend(d.user, _globalCtx),
};

self.onmessage = e => {
    const { action, ...data } = e.data;
    if (handlers[action]) handlers[action](data);
};
