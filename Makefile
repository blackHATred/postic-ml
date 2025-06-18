k8s-dashboard:
	@echo "Проксируем подключение к Kubernetes Dashboard"
	@echo "URL: https://localhost:8443"
	kubectl -n kubernetes-dashboard port-forward svc/kubernetes-dashboard-kong-proxy 8443:443

push-app:
	docker build -t "ghcr.io/blackhatred/postic-ml:latest" .; docker push "ghcr.io/blackhatred/postic-ml:latest"