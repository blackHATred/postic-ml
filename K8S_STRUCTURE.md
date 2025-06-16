# Kubernetes Configuration Structure

Структура конфигурации Kubernetes для Postic ML.

## 📁 Организация файлов

```
k8s/
├── base/                          # Базовые конфигурации
│   ├── app.yaml                   # Основное приложение (Deployment + Service)
│   ├── configmap.yaml             # Базовая конфигурация приложения
│   ├── kustomization.yaml         # Базовая kustomization
│   ├── namespace.yaml             # Namespace для проекта
│   ├── ollama.yaml                # LLM сервис (с GPU поддержкой)
│   ├── persistent-volumes.yaml    # Хранилища для данных
│   ├── qdrant.yaml               # Векторная база данных
│   └── redis.yaml                # Кеш и очереди
└── overlays/                      # Окружения
    ├── minikube/                  # Локальная разработка
    │   ├── app-deployment-patch.yaml    # Уменьшенные ресурсы для локальной разработки
    │   ├── app-service-patch.yaml       # NodePort для доступа в Minikube
    │   ├── kustomization.yaml           # Конфигурация для Minikube
    │   └── ollama-minikube-patch.yaml   # Настройки Ollama для Minikube
    └── registry/                  # Продакшн с GitHub Container Registry
        ├── app-deployment-patch.yaml    # Продакшн ресурсы и образ из registry
        ├── app-service-patch.yaml       # LoadBalancer для продакшна
        ├── kustomization.yaml           # Конфигурация для продакшна
        └── ollama-minikube-patch.yaml   # Настройки Ollama для продакшна
```

## 🎯 Назначение файлов

### Base (базовые конфигурации)

- **app.yaml** - Основное приложение Postic ML
  - Deployment с контейнером FastAPI
  - Service для внутреннего доступа
  - Базовые настройки ресурсов

- **ollama.yaml** - LLM сервис
  - GPU поддержка (nvidia.com/gpu: 1)
  - Автозагрузка модели gemma2:2b
  - Persistent Volume для кеширования моделей

- **qdrant.yaml** - Векторная база данных
  - Для семантического поиска
  - Persistent storage для векторов

- **redis.yaml** - Кеш и очереди
  - Для быстрого доступа к данным
  - Кеширование результатов поиска

### Overlays (окружения)

#### Minikube (локальная разработка)
- **Уменьшенные ресурсы** - для работы на локальной машине
- **NodePort сервисы** - для доступа через `minikube service`
- **Локальные образы** - используется `imagePullPolicy: Never`

#### Registry (продакшн)
- **Увеличенные ресурсы** - для продакшенной нагрузки
- **LoadBalancer сервисы** - для внешнего доступа
- **Образы из registry** - использует GitHub Container Registry

## 🚀 Использование

### Локальная разработка
```bash
kubectl apply -k k8s/overlays/minikube/
```

### Продакшн развертывание
```bash
kubectl apply -k k8s/overlays/registry/
```

### Удаление ресурсов
```bash
kubectl delete -k k8s/overlays/minikube/
# или
kubectl delete -k k8s/overlays/registry/
```

## 🔧 Настройка

### Ресурсы
Ресурсы настраиваются в patch файлах:
- **Minikube**: 512Mi RAM, 250m CPU (экономично)
- **Registry**: 2Gi RAM, 1 CPU (продакшн)

### Образы
- **Minikube**: использует локально собранные образы
- **Registry**: загружает из `ghcr.io/blackhatred/postic-ml`

### Сеть
- **Minikube**: NodePort на порту 30080
- **Registry**: LoadBalancer (внешний IP)

## 📋 Проверка конфигурации

```bash
# Проверка синтаксиса
kubectl kustomize k8s/overlays/minikube/
kubectl kustomize k8s/overlays/registry/

# Сухой прогон
kubectl apply -k k8s/overlays/minikube/ --dry-run=client

# Валидация ресурсов
kubectl apply -k k8s/overlays/minikube/ --validate=true
```
