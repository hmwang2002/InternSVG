// svgo.config.mjs
export default {
    plugins: [
        { name: 'inlineStyles', params: { onlyMatchedOnce: false, removeMatchedSelectors: true } },
        'convertStyleToAttrs',

        {
            name: 'preset-default',
            params: {
                overrides: {
                    convertShapeToPath: false,
                    removeViewBox: false,
                    minifyStyles: false,
                },
            },
        },

        { name: 'removeStyleElement' },

        {
            name: 'removeAttrs',
            params: {
                attrs: ['aria-.*'],
                preserveCurrentColor: true,
            },
        },
        {
            name: 'removeEditorsNSData',
            params: { additionalNamespaces: ['http://www.xml-cml.org/schema'] },
        },
    ],
};
