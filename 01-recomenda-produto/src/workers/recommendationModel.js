function freezeData(value) {
    if (value && typeof value === 'object' && !Object.isFrozen(value)) {
        Object.freeze(value);

        Object.values(value).forEach(freezeData);
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
    #state = null;

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
        this.#state = freezeData(this.#buildState(catalog, users));
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

        const ages = userList.map((user) => user.age);
        const minAge = ages.length ? Math.min(...ages) : -1;
        const maxAge = ages.length ? Math.max(...ages) : -1;
        const midAge = ages.length
                    ? (minAge + maxAge) / 2
                    : null;
        const catalog = catalogList.map((product, index) => ({
            index,
            price: Number(product.price),
            color: String(product.color),
            category: String(product.category)
        }));

        const prices = catalog.map((product) => product.price);
        const staticsByProduct = userList
            .flatMap((user) => user.purchases.map((purchase) => ({
                product: purchase.name,
                age: user.age,
                price: purchase.price
            })))
            .reduce((acc, current) => {
                if (!acc[current.product]) {
                    acc[current.product] = { ageSum: 0, count: 0,  avgAge: 0, avgAgeNormalized: 0 };
                }

                acc[current.product].ageSum += current.age;
                acc[current.product].count += 1;
                return acc;
            }, {});

        

        Object.entries(catalogList).forEach(([, product]) => {
            if (!staticsByProduct[product.name]) {
                staticsByProduct[product.name] = { ageSum: 0, count: 0, avgAge: 0, avgAgeNormalized: 0 };
            }
            const stat = staticsByProduct[product.name];
            stat.avgAge = stat.ageSum / (stat.count || 1);
            stat.avgAgeNormalized = stat.avgAge > 0 ? normalize(stat.avgAge, minAge, maxAge) : 0;
        });

        const purchasedProducts = catalogList.filter(
            (product) => staticsByProduct[product.name].count > 0
        );

        return {
            minPrice: prices.length ? Math.min(...prices) : null,
            maxPrice: prices.length ? Math.max(...prices) : null,
            colorCount: new Set(purchasedProducts.map((product) => product.color)).size,
            categoryCount: new Set(purchasedProducts.map((product) => product.category)).size,
            catalog,
            users: {
                minAge: minAge,
                maxAge: maxAge,
                midAge: midAge,
                staticsByProduct
                // list: users
            },
            dimentions: 2 + ages.length + catalog.length //não vi sentido nisso
        };
    }
}

/**
 * Shared recommendation model instance used by the worker.
 */
export const recommendationModel = new RecommendationModel();
