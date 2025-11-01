# GitHub Pages Deployment Instructions

## Quick Deploy to GitHub Pages (Demo Mode)

Your `index.html` is ready to deploy to GitHub Pages **right now**! It includes a demo mode that works without a backend server.

### Step 1: Update Your Repository

```bash
# Commit the new web files
git add web/ api.py deploy.sh
git commit -m "Add web interface for ocular disease classification"
git push origin main
```

### Step 2: Enable GitHub Pages

1. Go to your repository: https://github.com/fdbadmin/ODR
2. Click **Settings** (top menu)
3. Click **Pages** (left sidebar)
4. Under **Source**, select:
   - Branch: `main`
   - Folder: `/web` or `root` (depending on structure)
5. Click **Save**
6. Wait 1-2 minutes for deployment

### Step 3: Access Your Site

Your site will be available at:
```
https://fdbadmin.github.io/ODR/
```

or if you selected `/web`:
```
https://fdbadmin.github.io/ODR/web/
```

## Demo Mode Features

When hosted on GitHub Pages, the site automatically runs in **demo mode**:
- ✅ Beautiful interface works perfectly
- ✅ Image upload and preview
- ✅ Simulated prediction results (for demonstration)
- ⚠️ Not connected to real model (GitHub Pages is static HTML)

The demo mode is perfect for:
- Showcasing your project
- Portfolio demonstrations
- User interface testing
- Sharing with others

## For Real Predictions

To enable **real predictions**, you need to deploy the backend API separately:

### Option A: Use Heroku (Free Tier Available)

```bash
# Install Heroku CLI
brew install heroku  # macOS

# Login
heroku login

# Create app
heroku create odir-classifier

# Add Procfile (already created if you use deploy.sh)
echo "web: gunicorn api:app" > Procfile

# Deploy
git push heroku main

# Your API URL: https://odir-classifier.herokuapp.com
```

Then update `index.html` line ~460:
```javascript
const API_URL = 'https://odir-classifier.herokuapp.com/predict';
```

### Option B: Use Railway.app (Recommended - Easy)

1. Go to https://railway.app
2. Sign in with GitHub
3. Click "New Project" → "Deploy from GitHub repo"
4. Select your `ODR` repository
5. Railway auto-detects Python and deploys
6. Get your URL (e.g., `https://odr-api.railway.app`)
7. Update `API_URL` in `index.html`

### Option C: Use Render.com (Free Tier)

1. Go to https://render.com
2. New → Web Service
3. Connect your GitHub repo
4. Build command: `pip install -r requirements.txt`
5. Start command: `python api.py`
6. Get your URL
7. Update `API_URL` in `index.html`

## Architecture

```
┌─────────────────────────────────────┐
│   GitHub Pages (Static Frontend)   │
│                                     │
│  - index.html (User Interface)     │
│  - Beautiful UI                    │
│  - Image upload                    │
│  - Results display                 │
└──────────────┬──────────────────────┘
               │
               │ AJAX Request
               │
               ▼
┌─────────────────────────────────────┐
│   Cloud API (Your Choice)          │
│                                     │
│  - Flask API (api.py)              │
│  - PyTorch Model                   │
│  - Image preprocessing             │
│  - Disease classification          │
└─────────────────────────────────────┘
```

## Cost Analysis

| Option | Cost | Pros | Cons |
|--------|------|------|------|
| GitHub Pages Only | **FREE** | Simple, fast | Demo mode only |
| GitHub Pages + Railway | **$5/month** | Easy, reliable | Cost after free tier |
| GitHub Pages + Render | **FREE tier** | Good free tier | Cold starts |
| GitHub Pages + Heroku | **$5-7/month** | Reliable | No free tier anymore |
| TensorFlow.js (Future) | **FREE** | No backend needed | Model conversion required |

## Monitoring Your Site

### Check GitHub Pages Status
- Go to Settings → Pages
- See deployment status and URL

### Check API Health
- Visit: `https://your-api-url.com/health`
- Should return: `{"status": "healthy", "model_loaded": true}`

### Test Prediction API
```bash
curl -X POST https://your-api-url.com/predict \
  -F "image=@test_image.jpg"
```

## Troubleshooting

### Site Not Loading
1. Check Settings → Pages is enabled
2. Verify branch and folder are correct
3. Wait 2-3 minutes after enabling
4. Check for any GitHub Actions errors

### Demo Mode Not Working
- Clear browser cache
- Check browser console (F12) for errors
- Verify `index.html` is in the correct location

### API Connection Errors
1. Check API is deployed and running
2. Verify `API_URL` in `index.html` is correct
3. Check CORS is enabled (`flask-cors` installed)
4. Verify API endpoint with `curl` test

### CORS Errors
```javascript
// In index.html, the API should be:
const API_URL = 'https://your-actual-api-url.com/predict';  // NOT localhost
```

## Next Steps After Deployment

1. **Test the Site**: Visit your GitHub Pages URL
2. **Share**: Add the link to your GitHub README
3. **Deploy API**: Choose a backend option for real predictions
4. **Custom Domain** (Optional): Add custom domain in Settings
5. **Analytics** (Optional): Add Google Analytics
6. **SEO** (Optional): Add meta tags for better search results

## Example README Badge

Add this to your main README.md:

```markdown
## 🌐 Live Demo

Try the web interface: [https://fdbadmin.github.io/ODR/](https://fdbadmin.github.io/ODR/)

[![Website](https://img.shields.io/website?url=https%3A%2F%2Ffdbadmin.github.io%2FODR%2F)](https://fdbadmin.github.io/ODR/)
```

## Security Notes for Production

⚠️ **Before going live with real users:**

1. Add rate limiting to API
2. Implement authentication if needed
3. Add input validation
4. Use HTTPS only
5. Monitor API usage
6. Add error logging
7. Set up backups

## Questions?

- Check `web/README.md` for detailed deployment options
- Test locally first with `./deploy.sh`
- Open an issue on GitHub for help

---

**Ready to deploy?** Just run:
```bash
git add web/ api.py
git commit -m "Add web interface"
git push origin main
```

Then enable GitHub Pages in Settings! 🚀
