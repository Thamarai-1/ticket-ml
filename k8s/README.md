# Kubernetes Manifests for ML Service (FastAPI)

This directory contains production-ready Kubernetes manifests and network policies for deploying the Python FastAPI ML microservice to Kubernetes using `kubectl` and `kustomize`.

## Overview

- **Namespace**: `ticket-assitant`
- **Internal Service URL**: `http://ml-service.ticket-assitant.svc.cluster.local:8000`
- **Port**: `8000`
- **Components**:
  - `deployment.yaml`: Deployment running FastAPI with SentenceTransformer (`all-MiniLM-L6-v2`) and health probes (`/health`).
  - `service.yaml`: ClusterIP service exposing port 8000 internally.
  - `hpa.yaml`: HorizontalPodAutoscaler scaling between 1 and 3 replicas based on CPU load.
  - `pdb.yaml`: PodDisruptionBudget ensuring minimum availability during maintenance.
  - `networkpolicy.yaml`: Zero-Trust policy restricting incoming requests to backend pods only (`app: ticket-backend`).

## Manual Deployment via kubectl

```bash
kubectl apply -k ml-service/k8s/
```

Verify rollout:
```bash
kubectl rollout status deployment/ml-service -n ticket-assitant
kubectl get pods -n ticket-assitant -l app=ml-service
```
