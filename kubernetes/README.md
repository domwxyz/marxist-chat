# Marxist Chat Kubernetes Deployment

This directory contains Kubernetes configuration files for deploying the Marxist Chat application.

## Prerequisites

- Kubernetes cluster (e.g., minikube, GKE, EKS, AKS)
- kubectl CLI tool installed and configured
- Docker for building the container image

## Build and Push the Docker Image

Before deploying to Kubernetes, you need to build and push the Docker image:

```bash
# Build the Docker image
docker build -t marxist-chat:latest .

# Tag the image for your registry
docker tag marxist-chat:latest your-registry/marxist-chat:latest

# Push to your registry
docker push your-registry/marxist-chat:latest
```

Update the `deployment.yaml` file to use your registry's image path.

## Deploy to Kubernetes

Apply the manifests in the following order:

1. Create the persistent volume claims:
   ```bash
   kubectl apply -f kubernetes/persistent-volumes.yaml
   ```

2. Create the ConfigMap:
   ```bash
   kubectl apply -f kubernetes/configmap.yaml
   ```

3. Deploy the application:
   ```bash
   kubectl apply -f kubernetes/deployment.yaml
   ```

4. Create the service:
   ```bash
   kubectl apply -f kubernetes/service.yaml
   ```

5. Set up the ingress (requires an ingress controller like nginx-ingress):
   ```bash
   kubectl apply -f kubernetes/ingress.yaml
   ```

6. Apply the horizontal pod autoscaler:
   ```bash
   kubectl apply -f kubernetes/horizontal-pod-autoscaler.yaml
   ```

## TLS Configuration

Before applying the ingress configuration, you need to create a TLS secret:

```bash
kubectl create secret tls chat-tls-secret --key /path/to/private.key --cert /path/to/certificate.crt
```

Alternatively, you can use cert-manager to automatically provision and manage TLS certificates.

## Monitoring

Check the status of your deployment:

```bash
kubectl get pods
kubectl get services
kubectl get ingress
kubectl describe hpa marxist-chat-hpa
```

View logs:

```bash
kubectl logs -f deployment/marxist-chat
```

## Data Persistence

The application uses three persistent volume claims:
- `posts-cache-pvc`: Stores cached RSS feed data
- `vector-store-pvc`: Stores the vector database
- `logs-pvc`: Stores application logs

Make sure to back up these volumes regularly or consider using a storage solution with built-in redundancy.

## Scaling Considerations

The HorizontalPodAutoscaler will automatically scale the deployment based on CPU and memory usage, but keep in mind:

1. Each pod requires significant memory (2-4GB) due to the LLM model.
2. Consider using node selectors or taints/tolerations to ensure pods are scheduled on nodes with sufficient resources.
3. For larger deployments, implement a shared vector store (e.g., Pinecone, Qdrant, or a shared Chroma database).

## Production Recommendations

For production environments:
- Set resource requests and limits appropriate for your workload
- Implement proper monitoring and alerting
- Configure backups for persistent volumes
- Set up a CI/CD pipeline for automated deployments
- Use proper RBAC permissions
- Restrict network access using Network Policies
