// tailwind.config.js
const plugin = require('tailwindcss/plugin');

module.exports = {
  content: ['./src/**/*.{html,ts}'],
  theme: {
    extend: {
      colors: {
        primary: '#F7F7F7', // Light Gray
        secondary: '#D60000', // Red
        accent: '#333333', // Dark Gray
        accent2: '#d8d8cd', // Gray
        accent3: '#2c4143', // Dark Green
        accent4: '#58707b', // teal
        accent5: '#ffce32', // yellow
        accent6: '#61a8bd', // Light Blue
      },
      fontFamily: {
        inter: ['Inter', 'sans-serif'],
      },
      boxShadow: {
        'apple-sm': '0 1px 2px 0 rgba(0, 0, 0, 0.05)',
        'apple-md': '0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06)',
        'apple-lg': '0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05)',
        'apple-xl': '0 20px 25px -5px rgba(0, 0, 0, 0.1), 0 10px 10px -5px rgba(0, 0, 0, 0.04)',
      },
      borderRadius: {
        'apple': '1rem',
      },
      transitionProperty: {
        'height': 'height',
        'spacing': 'margin, padding',
      },
      transitionTimingFunction: {
        'apple': 'cubic-bezier(0.25, 0.1, 0.25, 1)',
      },
      transitionDuration: {
        '400': '400ms',
      },
      keyframes: {
        'apple-bounce': {
          '0%, 100%': { transform: 'translateY(-5%)' },
          '50%': { transform: 'translateY(0)' },
        },
        'apple-fade-in': {
          '0%': { opacity: '0' },
          '100%': { opacity: '1' },
        },
        'apple-scale': {
          '0%': { transform: 'scale(0.95)' },
          '100%': { transform: 'scale(1)' },
        },
      },
      animation: {
        'apple-bounce': 'apple-bounce 1s ease-in-out infinite',
        'apple-fade-in': 'apple-fade-in 0.5s ease-out',
        'apple-scale': 'apple-scale 0.3s ease-in-out',
      },
    },
  },
  variants: {
    extend: {
      opacity: ['disabled'],
      cursor: ['disabled'],
      backgroundColor: ['active'],
      textColor: ['active'],
    },
  },
  plugins: [
    plugin(({ addUtilities, theme, variants }) => {
      const newUtilities = {
        '.apple-focus-ring': {
          boxShadow: `0 0 0 4px ${theme('colors.accent6')}`,
        },
        '.apple-press-effect': {
          transform: 'scale(0.95)',
        },
        '.apple-hover-lift': {
          transition: 'transform 0.3s ease',
          '&:hover': {
            transform: 'translateY(-3px)',
          },
        },
        '.apple-glass': {
          backgroundColor: 'rgba(255, 255, 255, 0.7)',
          backdropFilter: 'blur(10px)',
        },
      };

      addUtilities(newUtilities, variants('appleEffects'));
    }),
  ],
};
