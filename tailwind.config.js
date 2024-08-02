// tailwind.config.js
const plugin = require('tailwindcss/plugin');

module.exports = {
  content: ['./src/**/*.{html,ts}'],
  darkMode: 'class', // Enable class-based dark mode
  theme: {
    extend: {
      colors: {
        primary: {
          DEFAULT: '#F7F7F7',
          100: '#FFFFFF',
          200: '#FAFAFA',
          300: '#F7F7F7',
          400: '#F0F0F0',
          500: '#E8E8E8',
        },
        secondary: {
          DEFAULT: '#D60000',
          100: '#FF8080',
          200: '#FF4040',
          300: '#FF0000',
          400: '#D60000',
          500: '#AD0000',
        },
        accent: {
          DEFAULT: '#333333',
          100: '#666666',
          200: '#4D4D4D',
          300: '#333333',
          400: '#1A1A1A',
          500: '#000000',
        },
        accent2: '#d8d8cd',
        accent3: '#2c4143',
        accent4: '#58707b',
        accent5: '#ffce32',
        accent6: '#61a8bd',
        'background-dark': '#1A1A1A',
        'text-dark': '#F7F7F7',
      },
      fontFamily: {
        inter: ['Inter', 'sans-serif'],
      },
      fontSize: {
        '2xs': ['0.625rem', { lineHeight: '1rem' }],
        '3xs': ['0.5rem', { lineHeight: '0.75rem' }],
      },
      spacing: {
        '1/2': '50%',
        '1/3': '33.333333%',
        '2/3': '66.666667%',
        '1/4': '25%',
      },
      gridTemplateColumns: {
        'layout-mobile': '1fr',
        'layout-desktop': '200px 1fr',
      },
      boxShadow: {
        'apple-sm': '0 1px 2px 0 rgba(0, 0, 0, 0.05)',
        'apple-md': '0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06)',
        'apple-lg': '0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05)',
        'apple-xl': '0 20px 25px -5px rgba(0, 0, 0, 0.1), 0 10px 10px -5px rgba(0, 0, 0, 0.04)',
        'apple-2xl': '0 25px 50px -12px rgba(0, 0, 0, 0.25)',
        'apple-inner': 'inset 0 2px 4px 0 rgba(0, 0, 0, 0.06)',
      },
      borderRadius: {
        'apple-sm': '0.375rem',
        'apple': '1rem',
        'apple-lg': '1.5rem',
        'apple-full': '9999px',
      },
      transitionProperty: {
        'height': 'height',
        'spacing': 'margin, padding',
        'opacity': 'opacity',
        'colors': 'color, background-color, border-color',
      },
      transitionTimingFunction: {
        'apple': 'cubic-bezier(0.25, 0.1, 0.25, 1)',
        'apple-in': 'cubic-bezier(0.42, 0, 1, 1)',
        'apple-out': 'cubic-bezier(0, 0, 0.58, 1)',
        'apple-in-out': 'cubic-bezier(0.42, 0, 0.58, 1)',
        'apple-ease-out': 'cubic-bezier(0.16, 1, 0.3, 1)',
      },
      transitionDuration: {
        '250': '250ms',
        '400': '400ms',
        '600': '600ms',
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
        'apple-spin': {
          '0%': { transform: 'rotate(0deg)' },
          '100%': { transform: 'rotate(360deg)' },
        },
        'apple-pulse': {
          '0%, 100%': { opacity: '1' },
          '50%': { opacity: '0.5' },
        },
      },
      animation: {
        'apple-bounce': 'apple-bounce 1s ease-in-out infinite',
        'apple-fade-in': 'apple-fade-in 0.5s ease-out',
        'apple-scale': 'apple-scale 0.3s ease-in-out',
        'apple-spin': 'apple-spin 1s linear infinite',
        'apple-pulse': 'apple-pulse 2s cubic-bezier(0.4, 0, 0.6, 1) infinite',
      },
      screens: {
        'xs': '475px',
        '3xl': '1600px',
      },
    },
  },
  variants: {
    extend: {
      opacity: ['disabled', 'hover', 'focus', 'dark'],
      cursor: ['disabled', 'hover'],
      backgroundColor: ['active', 'disabled', 'dark'],
      textColor: ['active', 'disabled', 'dark'],
      borderColor: ['focus', 'hover', 'disabled', 'dark'],
      ringColor: ['focus', 'hover'],
      ringOpacity: ['focus', 'hover'],
      scale: ['active', 'group-hover'],
    },
  },
  plugins: [
    plugin(({ addUtilities, theme, variants }) => {
      const newUtilities = {
        '.apple-focus-ring': {
          boxShadow: `0 0 0 4px ${theme('colors.accent6')}`,
        },
        '.apple-press-effect': {
          transform: 'scale(0.98)',
          transition: 'transform 0.1s ease-in-out',
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
        '.apple-text-shadow': {
          textShadow: '0 1px 2px rgba(0, 0, 0, 0.1)',
        },
        '.apple-inset-shadow': {
          boxShadow: 'inset 0 1px 3px rgba(0, 0, 0, 0.1)',
        },
        '.apple-card': {
          backgroundColor: theme('colors.primary.DEFAULT'),
          borderRadius: theme('borderRadius.apple'),
          boxShadow: theme('boxShadow.apple-md'),
          padding: theme('spacing.4'),
        },
        '.apple-button': {
          backgroundColor: theme('colors.secondary.DEFAULT'),
          color: theme('colors.primary.DEFAULT'),
          padding: `${theme('spacing.2')} ${theme('spacing.4')}`,
          borderRadius: theme('borderRadius.apple'),
          fontWeight: theme('fontWeight.semibold'),
          transition: 'all 0.3s ease',
          '&:hover': {
            backgroundColor: theme('colors.secondary.300'),
          },
          '&:active': {
            transform: 'scale(0.98)',
          },
        },
      };

      addUtilities(newUtilities, variants('appleEffects'));
    }),
    require('@tailwindcss/typography'),
    require('@tailwindcss/aspect-ratio'),
  ],
};
