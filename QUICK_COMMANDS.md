# Quick Commands Reference

Быстрый справочник команд для ежедневной работы с Postic ML.

## 🚀 Быстрый запуск

### Локальная разработка
```bash
# Запуск приложения
python main.py

# Запуск с отладкой
python main.py --debug

# Проверка зависимостей
pip install -r requirements.txt
```

### Minikube (одной командой)
```powershell
# Полное развертывание
minikube start --driver=docker --gpus=all --memory=8192 --cpus=4; minikube addons enable ingress; kubectl apply -k k8s/overlays/minikube/

# Получить URL
minikube service postic-ml-service -n postic-ml --url

# Перезапуск с пересборкой
kubectl delete -k k8s/overlays/minikube/; minikube docker-env | Invoke-Expression; docker build -t postic-ml:latest .; kubectl apply -k k8s/overlays/minikube/
```

## 🔧 Разработка

### Docker команды
```bash
# Сборка образа
docker build -t postic-ml:latest .

# Запуск локально
docker run -p 8000:8000 postic-ml:latest

# Удаление всех старых образов
docker image prune -a -f
```

### Kubernetes команды
```bash
# Применить изменения
kubectl apply -k k8s/overlays/minikube/

# Перезапустить deployment
kubectl rollout restart deployment/postic-ml-app -n postic-ml

# Посмотреть логи
kubectl logs -f deployment/postic-ml-app -n postic-ml

# Статус подов
kubectl get pods -n postic-ml

# Удалить все ресурсы
kubectl delete -k k8s/overlays/minikube/
```

## 📊 Мониторинг

### Просмотр логов
```bash
# Все логи
kubectl logs -f deployment/postic-ml-app -n postic-ml

# Последние 100 строк
kubectl logs --tail=100 deployment/postic-ml-app -n postic-ml

# Логи с временными метками
kubectl logs -f deployment/postic-ml-app -n postic-ml --timestamps
```

### Проверка статуса
```bash
# Статус всех ресурсов
kubectl get all -n postic-ml

# Использование ресурсов
kubectl top pods -n postic-ml
kubectl top nodes

# События
kubectl get events -n postic-ml --sort-by='.lastTimestamp'
```

## 🐛 Отладка

### Доступ к контейнеру
```bash
# Shell в поде
kubectl exec -it deployment/postic-ml-app -n postic-ml -- /bin/bash

# Выполнить команду
kubectl exec deployment/postic-ml-app -n postic-ml -- python -c "import sys; print(sys.version)"
```

### Сеть и сервисы
```bash
# Проверка сервисов
kubectl get svc -n postic-ml

# Port-forward для локального доступа
kubectl port-forward svc/postic-ml-service 8000:80 -n postic-ml

# Описание сервиса
kubectl describe svc postic-ml-service -n postic-ml
```

## 🔄 CI/CD

### Публикация образа
```powershell
# Логин в GitHub Container Registry
echo $env:GITHUB_TOKEN | docker login ghcr.io -u $env:GITHUB_USERNAME --password-stdin

# Сборка и публикация
$GIT_HASH = git rev-parse --short HEAD
docker build -t "ghcr.io/blackhatred/postic-ml:$GIT_HASH" .
docker push "ghcr.io/blackhatred/postic-ml:$GIT_HASH"

# Обновление latest тега
docker tag "ghcr.io/blackhatred/postic-ml:$GIT_HASH" "ghcr.io/blackhatred/postic-ml:latest"
docker push "ghcr.io/blackhatred/postic-ml:latest"
```

### Развертывание из registry
```bash
# Использование образа из registry
kubectl apply -k k8s/overlays/registry/

# Обновление образа
kubectl set image deployment/postic-ml-app postic-ml=ghcr.io/blackhatred/postic-ml:latest -n postic-ml
```

## 🧹 Очистка

### Полная очистка Minikube
```bash
# Остановка и удаление
minikube stop
minikube delete --all

# Очистка Docker
docker system prune -a -f
```

### Частичная очистка
```bash
# Удаление только приложения
kubectl delete -k k8s/overlays/minikube/

# Удаление namespace
kubectl delete namespace postic-ml

# Перезапуск Minikube
minikube stop
minikube start --driver=docker --gpus=all --memory=8192 --cpus=4
```

## 🔍 Полезные алиасы

Добавьте в ваш PowerShell профиль (`$PROFILE`):

```powershell
# Minikube алиасы
function mk-start { minikube start --driver=docker --gpus=all --memory=8192 --cpus=4 }
function mk-stop { minikube stop }
function mk-delete { minikube delete --all }
function mk-dashboard { minikube dashboard }
function mk-ip { minikube ip }

# Kubectl алиасы
function k { kubectl $args }
function kgp { kubectl get pods -n postic-ml }
function kgs { kubectl get svc -n postic-ml }
function kga { kubectl get all -n postic-ml }
function klogs { kubectl logs -f deployment/postic-ml-app -n postic-ml }
function kdesc { kubectl describe $args -n postic-ml }

# Docker алиасы
function d { docker $args }
function dps { docker ps }
function di { docker images }
function dprune { docker system prune -a -f }

# Postic ML алиасы
function postic-deploy { kubectl apply -k k8s/overlays/minikube/ }
function postic-delete { kubectl delete -k k8s/overlays/minikube/ }
function postic-restart { kubectl rollout restart deployment/postic-ml-app -n postic-ml }
function postic-url { minikube service postic-ml-service -n postic-ml --url }
function postic-logs { kubectl logs -f deployment/postic-ml-app -n postic-ml }
```

## 📚 Дополнительные ресурсы

- [Подробное руководство по развертыванию](DEPLOYMENT.md)
- [Основной README](README.md)
- [Структура проекта](STRUCTURE.md)
- [Registry setup](REGISTRY.md)
