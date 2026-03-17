group "default" {
  targets = ["api", "ui", "mlflow", "training", "evaluation", "security"]
}

target "api" {
  context    = "."
  dockerfile = "docker/Dockerfile.api"
  tags       = ["mlops-api:buildx"]
}

target "ui" {
  context    = "."
  dockerfile = "docker/Dockerfile.ui"
  tags       = ["mlops-ui:buildx"]
}

target "mlflow" {
  context    = "."
  dockerfile = "docker/Dockerfile.mlflow"
  tags       = ["mlops-mlflow:buildx"]
}

target "training" {
  context    = "."
  dockerfile = "docker/Dockerfile.training"
  tags       = ["mlops-training:buildx"]
}

target "evaluation" {
  context    = "."
  dockerfile = "docker/Dockerfile.evaluation"
  tags       = ["mlops-evaluation:buildx"]
}

target "security" {
  context    = "."
  dockerfile = "docker/Dockerfile.security"
  tags       = ["mlops-security:buildx"]
}
