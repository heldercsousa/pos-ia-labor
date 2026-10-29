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
        const midAge = ages.length ? (minAge + maxAge) / 2 : null;
        const catalog = catalogList.map((product, index) => ({
            index,
            name: product.name,
            price: product.price,
            color: product.color,
            category: product.category,
            statics: { ageSum: 0, count: 0, avgAge: 0, avgAgeNormalized: 0 }
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

       
        Object.entries(catalog).forEach(([, product]) => {
            let stat = staticsByProduct[product.name];
            if (!stat) {
                stat = { ageSum: 0, count: 0, avgAge: 0, avgAgeNormalized: 0 };
            }
            stat.avgAge = stat.ageSum / (stat.count || 1);
            stat.avgAgeNormalized = stat.avgAge > 0 ? normalize(stat.avgAge, minAge, maxAge) : 0;
            product.statics = stat;
        });

        const purchasedProducts = catalog.filter((product) => staticsByProduct[product.name] && staticsByProduct[product.name].count > 0);
        const colorCount = new Set(
            purchasedProducts.map((product) => product.color).filter(Boolean)
        ).size;
        const categoryCount = new Set(
            purchasedProducts.map((product) => product.category).filter(Boolean)
        ).size;

        return {
            minPrice: prices.length ? Math.min(...prices) : null,
            maxPrice: prices.length ? Math.max(...prices) : null,
            colorCount: colorCount,
            categoryCount: categoryCount,
            catalog,
            users: {
                minAge: minAge,
                maxAge: maxAge,
                midAge: midAge,
                // list: users
            },
            dimentions: 2 + categoryCount + colorCount //2 (preço e idade) + categorias + cores
        };
    }
}

/**
 * Shared recommendation model instance used by the worker.
 */
export const recommendationModel = new RecommendationModel();
