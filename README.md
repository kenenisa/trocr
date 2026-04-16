# TrOCR Service (Local OCR, Small Model)

This service runs local OCR using `microsoft/trocr-small-printed` and exposes an HTTP API for image text extraction.

It is designed to be built as a Docker image and deployed to k3s.

## API

- `GET /healthz` - health check
- `POST /ocr` - OCR endpoint (`multipart/form-data`, field: `file`)

Response example:

```json
{
  "text": "https://transactioninfo.ethiotelecom.et/receipt/DDF6VR6DRC",
  "tx_id": "DDF6VR6DRC",
  "model": "microsoft/trocr-small-printed",
  "device": "cpu"
}
```

## Local Run

```bash
cd trocr-service
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8080
```

Test:

```bash
curl -X POST "http://localhost:8080/ocr" \
  -F "file=@/path/to/screenshot.jpg"
```

## Docker Build

```bash
cd trocr-service
docker build -t trocr-service:local .
docker run --rm -p 8080:8080 trocr-service:local
```

## k3s Manifests

Files:

- `k8s/deployment.yaml`
- `k8s/service.yaml`
- `k8s/hpa.yaml`

Update image in `k8s/deployment.yaml`:

```yaml
image: ghcr.io/<your-org-or-user>/trocr-service:<tag>
```

Apply:

```bash
kubectl apply -f trocr-service/k8s/deployment.yaml
kubectl apply -f trocr-service/k8s/service.yaml
kubectl apply -f trocr-service/k8s/hpa.yaml
```

## GitHub Actions

Workflow: `.github/workflows/trocr-service.yml`

It does:

1. Build Docker image from `trocr-service/Dockerfile`
2. Push to GHCR:
   - `ghcr.io/<owner>/trocr-service:<sha>`
   - `ghcr.io/<owner>/trocr-service:latest`
3. Deploy to k8s (only if `KUBECONFIG_B64` secret exists)

### Required Secrets

- `KUBECONFIG_B64` (base64-encoded kubeconfig for target cluster)
- `K8S_NAMESPACE` (optional, defaults to `default`)

Notes:

- `GITHUB_TOKEN` is used automatically for GHCR push.
- If `KUBECONFIG_B64` is not set, deploy job is skipped and only image build/push runs.

## Runtime Configuration

Environment variables:

- `TROCR_MODEL_NAME` (default: `microsoft/trocr-small-printed`)
- `MAX_UPLOAD_BYTES` (default: `10485760`)

## Resource Starting Point (CPU)

- requests: `500m CPU`, `2Gi RAM`
- limits: `2 CPU`, `4Gi RAM`
- replicas: `1` (with HPA up to `3`)
