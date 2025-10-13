// svgo.config.js
export default {
    plugins: [
        { name: 'removeStyleElement' },

        {
            name: 'preset-default',
            params: {
                overrides: {
                    convertShapeToPath: false,
                    removeViewBox: false,
                    removeUnknownsAndDefaults: false,
                    removeUselessStrokeAndFill: false,
                    removeHiddenElems: false,
                },
            },
        },

        {
            name: 'removeAttrs',
            params: {
                attrs: ['aria-.*'],
                preserveCurrentColor: true,
            },
        },
        {
            name: 'removeEditorsNSData',
            params: {
                additionalNamespaces: ['http://www.xml-cml.org/schema'],
            },
        },
    ],
};
