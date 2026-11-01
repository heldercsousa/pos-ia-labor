/**
 * Recursively freezes an object, optionally skipping nested properties by key.
 * @param {*} value
 * @param {string[]|null} skip Property names whose values should not be traversed.
 * @returns {*} The original value.
 */
function freezeData(value, skip = null) {
    if (value && typeof value === 'object' && !Object.isFrozen(value)) {
        Object.freeze(value);

        Object.entries(value).forEach(([key, nestedValue]) => {
            if (!skip?.includes(key)) {
                freezeData(nestedValue, skip);
            }
        });
    }

    return value;
}

function normalize(value, minval, maxval) {
    return (value - minval) / ((maxval - minval) || 1);
}


/**
 * Owns the recommendation data and its loading lifecycle.
 */
class RecommendationModel {
    #weightsDefaultValues = {
        category: 0.4,
        color: 0.3,
        price: 0.2,
        age: 0.1
    };

    /**
     * Current weights for the recommendation model. By default, the weights are set to the default values (weightsDefault getter- read only).
     * Dev can change the weights to customize the recommendation model.
     * @returns {object}
     */
    weights = this.weightsDefault;

    #state = null;

    /**
     * return the default weights for the recommendation model.
     * @returns {object}
     */
    get weightsDefault() {
        return {...this.#weightsDefaultValues};
    }

    /**
     * Current model data, or `null` until the first successful reload.
     * The returned object and its nested values are frozen.
     * @returns {object|null}
     */
    get state() {
        return this.#state;
    }

    /**
     * Loads the product catalog and rebuilds the model state for the users.
     * @param {Array<{age: number, purchases: Array<{name: string, price: number}>}>} users
     * @returns {Promise<void>}
     */
    async reload(users) {
        const response = await fetch('/data/products.json');

        if (!response.ok) {
            throw new Error(`Could not load products: ${response.status}`);
        }

        const catalog = await response.json();
        this.#state = freezeData(this.#buildState(catalog, users), ['vectors']);
    }

    #buildState(catalogList, userList) {
        // const users = userList.map((user, index) => ({
        //     index,
        //     age: Number(user.age),
        //     purchases: user.purchases.map((purchase) => ({
        //         name: String(purchase.name),
        //         price: Number(purchase.price)
        //     }))
        // }));

        const staticDefaultValue = { ageSum: 0, count: 0, avgAge: 0, avgAgeNormalized: 0, color: '', category: '' };

        const ages = userList.map((user) => user.age);
        const minAge = ages.length ? Math.min(...ages) : -1;
        const maxAge = ages.length ? Math.max(...ages) : -1;
        const midAge = ages.length ? (minAge + maxAge) / 2 : null;
        const catalog = catalogList.map((product, index) => ({
            index,
            name: product.name,
            price: product.price,
            color: product.color,
            category: product.category,
            statics: {...staticDefaultValue},
            vectors: {
                price: {},
                color: {},
                category: {},
                age: {},
                concat: {}
            }
        }));

        const prices = catalog.map((product) => product.price);
        const minPrice = prices.length ? Math.min(...prices) : null;
        const maxPrice = prices.length ? Math.max(...prices) : null;

        const colors = [...new Set(catalog.map((p) => p.color).filter(Boolean))];
        const categories = [...new Set(catalog.map(p => p.category).filter(Boolean))];

        const staticsByProduct = userList
            .flatMap((user) => user.purchases.map((purchase) => {
                const prod = catalog.find(c => c.name == purchase.name);
                return {
                    product: purchase.name,
                    age: user.age,
                    price: purchase.price,
                    color: prod.color,
                    category: prod.category
                };
            }))
            .reduce((acc, current) => {
                if (!acc[current.product]) {
                    acc[current.product] = {...staticDefaultValue};
                }
                
                acc[current.product].ageSum += current.age;
                acc[current.product].count += 1;
                acc[current.product].category = current.category;
                acc[current.product].color = current.color;
                return acc;
            }, {});

        
        Object.entries(catalog).forEach(([, product]) => {
            let stat = staticsByProduct[product.name];
            if (!stat) {
                stat = {...staticDefaultValue};
            }
            stat.avgAge = stat.ageSum / (stat.count || 1);
            stat.avgAgeNormalized = stat.avgAge > 0 ? normalize(stat.avgAge, minAge, maxAge) : 0.5;
            product.statics = stat;

            product.vectors.price = tf
            .tensor1d([
                normalize(product.price, minPrice, maxPrice) * this.weights.price  //normaliza dados para ficar de 0 a 1, e aplicar o peso da recomendação
            ]);

            product.vectors.age = tf
            .tensor1d([
                stat.avgAgeNormalized * this.weights.age
            ]);
            
            product.vectors.category = tf
            .oneHot(categories.indexOf(product.category), categories.length)
            .cast('float32')
            .mul(this.weights.category);

            product.vectors.color = tf
            .oneHot(colors.indexOf(product.color), colors.length)
            .cast('float32')
            .mul(this.weights.color);

            product.vectors.concat = tf
            .concat1d([product.vectors.price, product.vectors.age, product.vectors.category, product.vectors.color]);
        });

        return {
            minPrice: minPrice,
            maxPrice: maxPrice,
            colorCount: colors.length,
            categoryCount: categories.length,
            catalog,
            users: {
                minAge: minAge,
                maxAge: maxAge,
                midAge: midAge,
                // list: users
            },
            dimentions: 2 + categories.length + colors.length //2 (preço e idade) + categorias + cores
        };
    }
}

/**
 * Shared recommendation model instance used by the worker.
 */
export const recommendationModel = new RecommendationModel();
