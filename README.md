# Wave Feature Extractor for Medical Imaging

A sophisticated signal processing application for analyzing wave-based signals with applications in medical imaging. This project demonstrates advanced signal processing techniques including time-domain analysis, frequency analysis, time-frequency analysis, and wavelet transforms, combined with machine learning for tissue classification.

![Image](https://github.com/user-attachments/assets/502952fc-6313-40a0-9074-c71502950529)
![Image](https://github.com/user-attachments/assets/0b5fc6ca-b8d7-4a06-9a9b-96ad77df06c1)

## Project Overview

The Wave Feature Extractor is a comprehensive tool designed to:

1. **Generate simulated medical wave signals** for different tissue types
2. **Analyze signals** in multiple domains (time, frequency, wavelet)
3. **Extract meaningful features** for pattern recognition
4. **Classify tissue types** using machine learning models
5. **Visualize results** through an intuitive, futuristic UI

This application serves as both an educational tool for understanding signal processing techniques in medical imaging and as a demonstration of how these methods can be applied to identify different tissue types based on their characteristic wave patterns.

## Features

### Signal Generation
- Simulate waves for different tissue types (normal, abnormal, tumor, cyst)
- Customize abnormality levels, duration, and sampling rate
- Apply advanced modifications to noise, frequency, and amplitude
- Save and catalog generated signals for later use

### Multi-domain Analysis
- **Time Domain**: Examine raw signal characteristics over time
- **Frequency Domain**: Identify frequency components using Fast Fourier Transform (FFT)
- **Time-Frequency Domain**: Visualize frequency changes over time with Short-Time Fourier Transform (STFT)
- **Wavelet Domain**: Perform multi-resolution analysis using wavelet transforms

### Feature Extraction
- Extract statistical features from time domain
- Extract spectral features from frequency domain
- Extract multi-resolution features from wavelet domain
- Visualize feature importance and distributions

### Classification
- Classify tissue types using multiple model types:
  - Convolutional Neural Networks (CNN)
  - Random Forest
  - Support Vector Machines (SVM)
- Visualize classification results and confidence levels
- Compare performance across different model types

### Interactive UI
- Modern, responsive Streamlit interface with dark mode
- Interactive visualizations with 2D and 3D plots
- Multi-page application for organized workflow
- Futuristic aesthetic with glowing elements and glass-effect cards

## Technology Stack

- **Python**: Core programming language
- **NumPy/SciPy**: Numerical computing and signal processing
- **PyWavelets**: Wavelet transforms and analysis
- **PyTorch**: Deep learning models
- **scikit-learn**: Machine learning algorithms
- **Pandas**: Data manipulation and analysis
- **Matplotlib/Plotly**: Data visualization
- **Streamlit**: Web application framework

## Project Structure

```
signal_processing_project/
│
├── app.py                      # Main Streamlit application entry point
│
├── modules/                    # Core functionality modules
│   ├── __init__.py             # Package initialization
│   ├── wave_extractor.py       # Signal processing algorithms
│   ├── wave_classifier.py      # ML models for wave classification
│   ├── visualizations.py       # Visualization functions
│   └── utils.py                # Helper functions and utilities
│
├── pages/                      # Multi-page Streamlit app sections
│   ├── home.py                 # Home/dashboard page
│   ├── analyzer.py             # Signal analysis page
│   ├── simulator.py            # Wave simulation page
│   └── classifier.py           # Tissue classification page
│
├── tests/                      # Unit tests
│   ├── test_extractor.py       # Tests for wave feature extraction
│   └── test_classifier.py      # Tests for classification models
│
├── models/                     # Pre-trained model storage
│   └── pretrained/             # Default pre-trained models
│
├── assets/                     # Static assets
│   ├── css/                    # Custom CSS styles
│   ├── images/                 # Static images for UI
│   └── sample_data/            # Sample wave data for demos
│
├── requirements.txt            # Project dependencies
├── setup.py                    # Package configuration
└── README.md                   # Project documentation
```

## Development Life Cycle

### 1. Planning and Design
- Requirements gathering and feasibility analysis
- UI/UX design with focus on futuristic aesthetic
- Architecture design using modular components
- Selection of signal processing algorithms and ML models

### 2. Development
- Implementation of core signal processing modules
- Development of classification models and training pipelines
- Creation of visualization components
- Integration with Streamlit for interactive UI
- Unit testing and validation

### 3. Testing
- Component testing for each module
- Integration testing for the full application
- Performance testing and optimization
- User experience testing

### 4. Deployment
- Packaging of application for distribution
- Documentation of installation and usage
- CI/CD pipeline setup for automated deployment
- Version control and release management

### 5. Maintenance
- Bug fixes and issue resolution
- Feature enhancements based on user feedback
- Performance optimizations
- Documentation updates

## Deployment Pipeline

### Local Development Environment
1. **Setup**:
   ```bash
   # Clone repository
   git clone https://github.com/yourusername/signal-processing-project.git
   cd signal-processing-project
   
   # Create and activate virtual environment
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   
   # Install in development mode
   pip install -e .
   ```

2. **Run Development Server**:
   ```bash
   streamlit run app.py
   ```

3. **Testing**:
   ```bash
   # Run unit tests
   python -m unittest discover tests
   
   # Run with coverage
   coverage run -m unittest discover tests
   coverage report -m
   ```

### Continuous Integration
- GitHub Actions workflow for automated testing
- Code quality checks with flake8 and black
- Test coverage reports with codecov

### Deployment Options

#### 1. Streamlit Cloud Deployment
```yaml
# .streamlit/config.toml
[theme]
primaryColor = "#4c83ff"
backgroundColor = "#0e1117"
secondaryBackgroundColor = "#1f2937"
textColor = "#ffffff"
font = "sans serif"

[server]
enableCORS = false
```

Streamlit Cloud automatically deploys from the connected GitHub repository.

#### 2. Docker Deployment
```Dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "app.py"]
```

Build and run:
```bash
docker build -t wave-feature-extractor .
docker run -p 8501:8501 wave-feature-extractor
```

#### 3. Kubernetes Deployment
```yaml
# kubernetes/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: wave-feature-extractor
spec:
  replicas: 3
  selector:
    matchLabels:
      app: wave-feature-extractor
  template:
    metadata:
      labels:
        app: wave-feature-extractor
    spec:
      containers:
      - name: wave-feature-extractor
        image: wave-feature-extractor:latest
        ports:
        - containerPort: 8501
        resources:
          limits:
            cpu: "1"
            memory: "2Gi"
          requests:
            cpu: "500m"
            memory: "1Gi"
---
apiVersion: v1
kind: Service
metadata:
  name: wave-feature-extractor-service
spec:
  selector:
    app: wave-feature-extractor
  ports:
  - port: 80
    targetPort: 8501
  type: LoadBalancer
```

Apply the configuration:
```bash
kubectl apply -f kubernetes/deployment.yaml
```

## CI/CD Pipeline

We use GitHub Actions for our CI/CD pipeline, which automatically tests, builds, and deploys the application.

```yaml
# .github/workflows/ci-cd.yml
name: CI/CD Pipeline

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v2
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: '3.9'
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -e ".[dev]"
    - name: Lint with flake8
      run: |
        flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
    - name: Test with pytest
      run: |
        pytest
    - name: Upload coverage report
      uses: codecov/codecov-action@v1

  build:
    needs: test
    runs-on: ubuntu-latest
    if: github.event_name == 'push' && github.ref == 'refs/heads/main'
    steps:
    - uses: actions/checkout@v2
    - name: Build Docker image
      run: |
        docker build -t wave-feature-extractor:latest .
        docker tag wave-feature-extractor:latest yourusername/wave-feature-extractor:latest
    - name: Push to Docker Hub
      run: |
        echo "${{ secrets.DOCKER_PASSWORD }}" | docker login -u "${{ secrets.DOCKER_USERNAME }}" --password-stdin
        docker push yourusername/wave-feature-extractor:latest

  deploy:
    needs: build
    runs-on: ubuntu-latest
    if: github.event_name == 'push' && github.ref == 'refs/heads/main'
    steps:
    - uses: actions/checkout@v2
    - name: Deploy to Streamlit Cloud
      uses: streamlit/cloud-deploy-action@v1
      with:
        app-name: wave-feature-extractor
        api-key: ${{ secrets.STREAMLIT_API_KEY }}
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/signal-processing-project.git
cd signal-processing-project
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -e .
```

## Usage

Run the Streamlit application:
```bash
streamlit run app.py
```

The application will be available at http://localhost:8501

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

Distributed under the MIT License. See `LICENSE` for more information.

## Acknowledgements

- PyWavelets for wavelet transform implementation
- Streamlit for the interactive web application framework
- The medical imaging research community for inspiration and methodologies
