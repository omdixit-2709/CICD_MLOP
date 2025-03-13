# GitHub Repository Setup for CI/CD

Follow these steps to set up your GitHub repository with the necessary configuration for CI/CD:

## 1. Create a GitHub Repository

1. Go to [GitHub](https://github.com/) and sign in to your account
2. Click on the "+" icon in the top right corner and select "New repository"
3. Name your repository (e.g., "sentiment-analysis-mlops")
4. Choose whether the repository should be public or private
5. Do not initialize the repository with a README, .gitignore, or license (since we're importing an existing repository)
6. Click "Create repository"

## 2. Push Your Local Repository to GitHub

Run these commands in your terminal, replacing `YOUR_USERNAME` with your GitHub username and `YOUR_REPO_NAME` with your repository name:

```bash
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
git branch -M main
git push -u origin main
```

## 3. Set Up Required GitHub Secrets

For the CI/CD workflow to function properly, you need to add these secrets to your GitHub repository:

1. Go to your GitHub repository
2. Click on "Settings" tab
3. In the left sidebar, click on "Secrets and variables" > "Actions"
4. Click "New repository secret" and add each of the following:

### Docker Hub Secrets
- **DOCKER_HUB_USERNAME**: Your Docker Hub username
- **DOCKER_HUB_ACCESS_TOKEN**: Your Docker Hub access token (create one in Docker Hub account settings)

### Heroku Secrets
- **HEROKU_API_KEY**: Your Heroku API key (find in your Heroku account settings)
- **HEROKU_APP_NAME**: Your Heroku app name (e.g., "sentiment-analysis-api-app")
- **HEROKU_EMAIL**: Your Heroku email address

## 4. First CI/CD Workflow Run

The CI/CD workflow will automatically run when:
- You push changes to the `main` or `master` branch
- You create a pull request to the `main` or `master` branch
- You manually trigger the workflow from the "Actions" tab

## 5. Monitoring the Workflow

1. Go to the "Actions" tab in your GitHub repository
2. You will see your workflows listed there
3. Click on a workflow run to see details and logs

## Troubleshooting

If your workflow fails, check:
1. That all required secrets are set correctly
2. Logs in the "Actions" tab to identify specific errors
3. That your repository contains all necessary files, especially the Dockerfiles and API code 