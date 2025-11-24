# g2-forge Documentation Site

This directory contains the GitHub Pages documentation site for g2-forge.

## Structure

```
docs/
├── index.html          # Main landing page
├── _config.yml         # Jekyll configuration
├── assets/
│   ├── css/
│   │   └── style.css   # Main stylesheet
│   ├── js/
│   │   └── main.js     # Interactive features
│   └── images/         # Images and logos
└── README.md           # This file
```

## Local Development

To test the site locally:

### Option 1: Using Jekyll (recommended)

```bash
# Install Jekyll (if not already installed)
gem install jekyll bundler

# Navigate to docs directory
cd docs/

# Serve the site locally
jekyll serve

# Visit http://localhost:4000/g2-forge/
```

### Option 2: Using Python's HTTP server

```bash
# Navigate to docs directory
cd docs/

# Start simple HTTP server
python -m http.server 8000

# Visit http://localhost:8000/
```

### Option 3: Using Node.js http-server

```bash
# Install http-server globally
npm install -g http-server

# Navigate to docs directory
cd docs/

# Start server
http-server

# Visit http://localhost:8080/
```

## Deployment

The site is automatically deployed to GitHub Pages when changes are pushed to the main branch.

**URL**: https://gift-framework.github.io/g2-forge/

## Customization

### Updating Content

- **Main page**: Edit `index.html`
- **Styles**: Edit `assets/css/style.css`
- **JavaScript**: Edit `assets/js/main.js`
- **Configuration**: Edit `_config.yml`

### Adding New Pages

To add new documentation pages:

1. Create a new HTML or Markdown file in `docs/`
2. Add navigation links in `index.html` if needed
3. Update `_config.yml` if using Jekyll collections

### Updating Styles

The site uses CSS custom properties (variables) for easy theming. Main colors and styles can be updated in the `:root` section of `assets/css/style.css`:

```css
:root {
    --primary-color: #6366f1;
    --secondary-color: #06b6d4;
    --accent-color: #f59e0b;
    /* ... more variables */
}
```

## Features

### Current Features

- ✅ Responsive design (mobile, tablet, desktop)
- ✅ Smooth scrolling navigation
- ✅ Animated sections on scroll
- ✅ Syntax highlighting for code blocks
- ✅ Copy buttons for code snippets
- ✅ Modern gradient design
- ✅ SEO optimized

### Planned Features

- [ ] Dark mode toggle
- [ ] Interactive topology visualizer
- [ ] API documentation
- [ ] Tutorial pages
- [ ] Blog for project updates
- [ ] Search functionality

## Browser Support

- Chrome/Edge (latest 2 versions)
- Firefox (latest 2 versions)
- Safari (latest 2 versions)
- Mobile browsers (iOS Safari, Chrome Mobile)

## Contributing

To contribute to the documentation site:

1. Make changes in the `docs/` directory
2. Test locally using one of the methods above
3. Commit and push changes
4. Create a pull request

## License

Same as the main g2-forge project (MIT License).

## Contact

- **Email**: brieuc@bdelaf.com
- **GitHub Issues**: https://github.com/gift-framework/g2-forge/issues
- **Project**: https://github.com/gift-framework/g2-forge
