# Руководство по развертыванию с GitHub Container Registry

## 🚀 Быстрый старт

### 1. Настройка GitHub Container Registry
```bash
.\setup-github-registry.bat
```

### 2. Сборка и публикация образа
```bash
.\publish-image.bat
```

### 3. Развертывание из Registry
```bash
.\deploy-minikube-registry.bat
```

## 📦 Доступные скрипты

| Скрипт | Назначение |
|--------|------------|
| `setup-github-registry.bat` | Однократная настройка аутентификации GHCR |
| `publish-image.bat` | Сборка и публикация образа в GHCR |
| `deploy-minikube-registry.bat` | Развертывание с использованием опубликованного образа |

## 🔐 Настройка аутентификации

1. **Создание Personal Access Token:**
   - Перейдите на https://github.com/settings/tokens
   - Создайте токен с правами `write:packages` и `read:packages`

2. **Вход в Registry:**
   ```bash
   echo YOUR_PAT | docker login ghcr.io -u YOUR_USERNAME --password-stdin
   ```

## 📋 Теги образов

Опубликованные образы используют следующие теги:
- `ghcr.io/blackHATred/postic-ml:latest` - Последняя стабильная версия
- `ghcr.io/blackHATred/postic-ml:k8s-latest` - Версия для развертывания в Kubernetes  
- `ghcr.io/blackHATred/postic-ml:COMMIT_HASH` - Версия конкретного коммита

## 🔄 CI/CD с GitHub Actions

Репозиторий включает автоматические сборки при:
- ✅ Push в ветки main/develop
- ✅ Pull requests
- ✅ Git теги (v*)

Подробности см. в `.github/workflows/docker-publish.yml`.

## 🏗️ Варианты развертывания

### Локальная разработка
```bash
.\deploy-minikube.bat  # Сборка локально
```

### Продакшен/Registry
```bash
.\deploy-minikube-registry.bat  # Использование опубликованного образа
```

## 🛠️ Устранение неполадок

### Проблемы с аутентификацией
- Проверьте, что PAT имеет корректные права
- Проверьте вход: `docker info | findstr ghcr.io`
- Перезапустите настройку: `.\setup-github-registry.bat`

### Проблемы с загрузкой образа
- Убедитесь, что образ существует: https://github.com/blackHATred?tab=packages
- Проверьте видимость образа (публичный vs приватный)
- Проверьте секрет Kubernetes: `kubectl get secret ghcr-secret -n postic-ml`

### Конфликты Registry vs Локальные образы
- Очистите локальные образы: `docker image prune -a`
- Используйте конкретные теги для избежания конфликтов
- Проверьте, какой образ используется: `kubectl describe pod -n postic-ml`
