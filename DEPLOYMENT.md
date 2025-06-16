# Postic ML Deployment Guide

Это руководство содержит все команды для развертывания и управления Postic ML.

## Содержание

- [Требования](#требования)
- [Развертывание в Minikube](#развертывание-в-minikube)
- [Публикация образов](#публикация-образов)
- [Настройка GitHub Container Registry](#настройка-github-container-registry)
- [Полезные команды](#полезные-команды)

## Требования

Убедитесь, что у вас установлены следующие инструменты:

```bash
# Проверка наличия необходимых инструментов
minikube version
kubectl version --client
docker version
git version
```

### Установка инструментов (Windows)

```powershell
# Установка через Chocolatey
choco install minikube kubectl docker-desktop git
```

## Развертывание в Minikube

### 1. Развертывание с GPU поддержкой

```powershell
# Остановка существующего кластера (если есть)
minikube delete --all

# Запуск Minikube с GPU поддержкой
minikube start --driver=docker --container-runtime=docker --gpus=all --memory=8192 --cpus=4 --disk-size=30g --kubernetes-version=v1.33.1

# Включение дополнений
minikube addons enable ingress
minikube addons enable metrics-server

# Настройка Docker environment
minikube docker-env | Invoke-Expression

# Сборка образа в Minikube
docker build -t postic-ml:latest .

# Развертывание приложения
kubectl apply -k k8s/overlays/minikube/

# Проверка статуса
kubectl get pods -n postic-ml
kubectl get services -n postic-ml

# Получение URL для доступа
minikube service postic-ml-service -n postic-ml --url
```

### 2. Развертывание только на CPU

```powershell
# Остановка существующего кластера
minikube delete --all

# Запуск Minikube без GPU
minikube start --driver=docker --container-runtime=docker --memory=6144 --cpus=4 --disk-size=30g

# Включение дополнений
minikube addons enable ingress
minikube addons enable metrics-server

# Настройка Docker environment
minikube docker-env | Invoke-Expression

# Сборка образа
docker build -t postic-ml:latest .

# Развертывание с CPU-конфигурацией
kubectl apply -k k8s/overlays/minikube/

# Проверка статуса
kubectl get pods -n postic-ml
minikube service postic-ml-service -n postic-ml --url
```

### 3. Развертывание из GitHub Container Registry

```powershell
# Настройка переменных
$GITHUB_USERNAME = "blackhatred"
$IMAGE_NAME = "postic-ml"
$REGISTRY = "ghcr.io"
$IMAGE_TAG = "$REGISTRY/$GITHUB_USERNAME/$IMAGE_NAME:k8s-latest"

# Остановка существующего кластера
minikube delete --all

# Запуск Minikube
minikube start --driver=docker --container-runtime=docker --gpus=all --memory=8192 --cpus=4 --disk-size=30g --kubernetes-version=v1.33.1

# Включение дополнений
minikube addons enable ingress
minikube addons enable metrics-server

# Создание секрета для доступа к registry (если нужно)
kubectl create secret docker-registry ghcr-secret `
  --docker-server=ghcr.io `
  --docker-username=$GITHUB_USERNAME `
  --docker-password=$env:GITHUB_TOKEN `
  --namespace=postic-ml

# Развертывание из registry
kubectl apply -k k8s/overlays/registry/

# Проверка статуса
kubectl get pods -n postic-ml
kubectl get services -n postic-ml

# Получение URL
minikube service postic-ml-service -n postic-ml --url
```

## Публикация образов

### Сборка и публикация в GitHub Container Registry

```powershell
# Настройка переменных
$GITHUB_USERNAME = "blackhatred"
$IMAGE_NAME = "postic-ml"
$REGISTRY = "ghcr.io"

# Проверка авторизации в GitHub Container Registry
docker info | Select-String $REGISTRY

# Если не авторизованы, выполните вход:
# echo $env:GITHUB_TOKEN | docker login ghcr.io -u $GITHUB_USERNAME --password-stdin

# Получение commit hash для тегирования
$GIT_HASH = git rev-parse --short HEAD

# Сборка образа
Write-Host "🏗️ Building Docker image..." -ForegroundColor Green
docker build -t "${REGISTRY}/${GITHUB_USERNAME}/${IMAGE_NAME}:latest" .
docker build -t "${REGISTRY}/${GITHUB_USERNAME}/${IMAGE_NAME}:${GIT_HASH}" .
docker build -t "${REGISTRY}/${GITHUB_USERNAME}/${IMAGE_NAME}:k8s-latest" .

# Публикация образов
Write-Host "📤 Pushing images to registry..." -ForegroundColor Green
docker push "${REGISTRY}/${GITHUB_USERNAME}/${IMAGE_NAME}:latest"
docker push "${REGISTRY}/${GITHUB_USERNAME}/${IMAGE_NAME}:${GIT_HASH}"
docker push "${REGISTRY}/${GITHUB_USERNAME}/${IMAGE_NAME}:k8s-latest"

Write-Host "✅ Images successfully published!" -ForegroundColor Green
Write-Host "📦 Image: ${REGISTRY}/${GITHUB_USERNAME}/${IMAGE_NAME}:latest"
Write-Host "📦 Image: ${REGISTRY}/${GITHUB_USERNAME}/${IMAGE_NAME}:${GIT_HASH}"
Write-Host "📦 Image: ${REGISTRY}/${GITHUB_USERNAME}/${IMAGE_NAME}:k8s-latest"
```

## Настройка GitHub Container Registry

### Создание Personal Access Token

1. Перейдите на https://github.com/settings/tokens
2. Нажмите "Generate new token" → "Generate new token (classic)"
3. Выберите срок действия (рекомендуется 90 дней или больше)
4. Выберите области доступа:
   - ✅ `write:packages` (загрузка пакетов)
   - ✅ `read:packages` (скачивание пакетов)
   - ✅ `delete:packages` (удаление пакетов - опционально)
5. Нажмите "Generate token"
6. Скопируйте токен (вы не увидите его снова!)

### Настройка аутентификации

```powershell
# Настройка переменных среды
$env:GITHUB_USERNAME = "your-username"
$env:GITHUB_TOKEN = "your-personal-access-token"

# Вход в GitHub Container Registry
echo $env:GITHUB_TOKEN | docker login ghcr.io -u $env:GITHUB_USERNAME --password-stdin

# Проверка успешной авторизации
docker info | Select-String "ghcr.io"

# Настройка Git (если нужно)
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"
```

## Полезные команды

### Мониторинг и отладка

```powershell
# Просмотр логов приложения
kubectl logs -f deployment/postic-ml-app -n postic-ml

# Просмотр статуса подов
kubectl get pods -n postic-ml -w

# Описание пода для отладки
kubectl describe pod <pod-name> -n postic-ml

# Доступ к shell внутри пода
kubectl exec -it <pod-name> -n postic-ml -- /bin/bash

# Просмотр событий в namespace
kubectl get events -n postic-ml --sort-by='.lastTimestamp'

# Проверка ресурсов
kubectl top pods -n postic-ml
kubectl top nodes
```

### Управление Minikube

```powershell
# Статус Minikube
minikube status

# Доступ к dashboard
minikube dashboard

# Просмотр IP адреса
minikube ip

# Остановка Minikube
minikube stop

# Удаление кластера
minikube delete

# Очистка всех кластеров
minikube delete --all

# Просмотр профилей
minikube profile list
```

### Docker команды

```powershell
# Просмотр образов
docker images | Select-String "postic-ml"

# Удаление старых образов
docker image prune -f

# Просмотр контейнеров
docker ps -a

# Очистка системы Docker
docker system prune -a
```

### Kubernetes команды

```powershell
# Применение конфигураций
kubectl apply -k k8s/overlays/minikube/

# Удаление ресурсов
kubectl delete -k k8s/overlays/minikube/

# Просмотр всех ресурсов в namespace
kubectl get all -n postic-ml

# Проверка конфигурации
kubectl config current-context
kubectl config get-contexts

# Переключение namespace по умолчанию
kubectl config set-context --current --namespace=postic-ml
```

## Устранение неполадок

### Типичные проблемы

1. **Minikube не запускается**
   ```powershell
   minikube delete --all
   minikube start --driver=docker
   ```

2. **Образ не найден**
   ```powershell
   # Убедитесь, что используете правильный Docker environment
   minikube docker-env | Invoke-Expression
   docker images
   ```

3. **Проблемы с сетью**
   ```powershell
   # Перезапуск сетевых компонентов
   minikube addons disable ingress
   minikube addons enable ingress
   ```

4. **Недостаточно ресурсов**
   ```powershell
   # Увеличение ресурсов
   minikube config set memory 8192
   minikube config set cpus 4
   minikube delete
   minikube start
   ```
