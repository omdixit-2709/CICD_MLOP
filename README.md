# Sentiment Analysis MLOps Project

A complete MLOps pipeline for sentiment analysis using the Sentiment140 dataset, with a FastAPI service, Docker containerization, CI/CD with GitHub Actions, and deployment to Heroku.

## Project Report

For a comprehensive explanation of the project, including architecture details, implementation specifics, CI/CD pipeline, challenges, and future work, please refer to our [detailed project report](PROJECT_REPORT.md).

## Project Structure

```
sentiment-analysis/
├── data/
│   ├── raw/                # Raw data storage
│   │   └── sentiment140.csv (download separately)
│   └── processed/          # Processed data
│       └── vectorizer.joblib
├── models/
│   └── saved_models/       # Trained model files
│       ├── sentiment_model.joblib
│       └── docker_model.joblib
├── tests/                  # Test files
│   └── test_api.py
├── .github/
│   └── workflows/          # CI/CD workflows
│       └── ci-cd.yml
├── logs/                   # Application logs
├── direct_api.py           # Main API implementation
├── direct_api_compatible.py # API with compatible model
├── test_direct_api.py      # API test script
├── Dockerfile              # Docker configuration
├── Dockerfile.heroku       # Heroku Docker configuration
├── heroku.yml              # Heroku deployment configuration
├── Procfile                # Heroku process configuration
├── requirements.txt        # Project dependencies
├── requirements_docker.txt # Docker-specific dependencies
├── docker-compose.yml      # Docker Compose configuration
├── prometheus.yml          # Prometheus configuration
└── README.md               # Project documentation
```

## Features

- Sentiment analysis model with 76.12% accuracy
- RESTful API built with FastAPI
- Authentication with HTTP Basic Auth
- Comprehensive logging with loguru
- Performance monitoring with Prometheus
- Docker containerization
- CI/CD pipeline with GitHub Actions
- Automated deployment to Heroku
- Unit and integration tests with pytest

## Setup and Installation

### Prerequisites

- Python 3.9+
- Docker Desktop
- Git
- Heroku CLI (for Heroku deployment)

### Dataset Download

The Sentiment140 dataset is not included in this repository due to its large size. You need to download it separately:

1. Download the dataset from [Kaggle](https://www.kaggle.com/datasets/kazanova/sentiment140) or the [official website](http://help.sentiment140.com/for-students/)
2. Place the CSV file in the `data/raw/` directory as `sentiment140.csv`

### Local Development

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd sentiment-analysis
   ```

2. Download the dataset as described above

3. Create a virtual environment and install dependencies:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

4. Run the API locally:
   ```bash
   python direct_api.py
   ```

5. Test the API:
   ```bash
   python test_direct_api.py
   ```

### Docker Deployment

1. Build the Docker image:
   ```bash
   docker build -t sentiment-analysis-api -f Dockerfile.heroku .
   ```

2. Run the container:
   ```bash
   docker run -p 8004:8004 -e PORT=8004 sentiment-analysis-api
   ```

3. Alternatively, use Docker Compose to run both the API and Prometheus:
   ```bash
   docker-compose up -d
   ```

### Heroku Deployment

1. Login to Heroku:
   ```bash
   heroku login
   ```

2. Create a new Heroku app:
   ```bash
   heroku create <app-name>
   ```

3. Set the Heroku remote:
   ```bash
   heroku git:remote -a <app-name>
   ```

4. Deploy using the heroku.yml configuration:
   ```bash
   git push heroku main
   ```

5. Alternatively, deploy using the Container Registry:
   ```bash
   heroku container:login
   heroku container:push web
   heroku container:release web
   ```

## API Endpoints

- `GET /`: Root endpoint, returns a welcome message
- `GET /test`: Test endpoint, returns a test message with timestamp
- `GET /health`: Health check endpoint
- `POST /predict`: Sentiment prediction endpoint (requires authentication)

### Authentication

The API uses HTTP Basic Authentication with these demo credentials:
- Username: admin
- Password: password

### Example Request

```bash
curl -X POST http://localhost:8004/predict \
  -H "Content-Type: application/json" \
  -H "Authorization: Basic YWRtaW46cGFzc3dvcmQ=" \
  -d '{"text": "I love this product!"}'
```

## CI/CD Pipeline

The project includes a GitHub Actions workflow that:
1. Runs tests
2. Builds and tests the Docker image
3. Pushes the image to Docker Hub (on main branch)
4. Deploys the application to Heroku (on main branch)

### GitHub Secrets Required

For the CI/CD pipeline to work, add these secrets to your GitHub repository:

- `DOCKER_HUB_USERNAME`: Your Docker Hub username
- `DOCKER_HUB_ACCESS_TOKEN`: Your Docker Hub access token
- `HEROKU_API_KEY`: Your Heroku API key
- `HEROKU_APP_NAME`: Your Heroku app name
- `HEROKU_EMAIL`: Your Heroku email address

## Monitoring

1. Access Prometheus dashboard at http://localhost:9090 when using Docker Compose
2. API metrics are available at http://localhost:8004/metrics

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests
5. Submit a pull request

## License

[MIT License](LICENSE) 