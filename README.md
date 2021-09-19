# Taxi Tips

This project trains and runs a model for  NYC Taxi tip prediction.

## Quick Start

Sign up at [neu.ro](https://neu.ro) and setup your local machine according to [instructions](https://docs.neu.ro/).

Then run:

```shell
# Initial setup
pip install -U neuro-cli neuro-flow
neuro login

# Build docker images
neuro-flow build myimage
neuro-flow build locust

# Download the dataset to the storage
neuro-flow run download_data
```

Update config files:

Replace `alexeynaiden` in `endpoint_url` for `locust` to your username in `.neuro/live.yml`:

```yaml
  locust:
    image: $[[ images.locust.ref ]]
    preset: cpu-small
    multi: true
    http_port: 8080
    http_auth: False
    life_span: 1d
    detach: True
    browse: True
    params:
      endpoint_url: https://taxi-tips-lb--alexeynaiden.jobs.default.org.neu.ro/predict
```

Do the same to `backend inference` job names in `config/haproxy.cfg`:

```config
backend inference
  server server-a taxi-tips-inference-a--alexeynaiden.platform-jobs:8000
  server server-b taxi-tips-inference-b--alexeynaiden.platform-jobs:8000
```

After setup is done, you're free to:

```shell
# Run training and save model and scaler
neuro-flow run train

# Run filebrowser and navigate to the results/ folder in the browser to see the results
neuro-flow run filebrowser

# Run two inference instances
neuro-flow run inference -s a
neuro-flow run inference -s b

# Run the load balancer
neuro-flow run lb

# Run locust and start test in the web ui
# Note you can run multiple locust instances to increase throughput
# The example belows runs two, but more can be started
# Two inference servers can survive 3 locust instances with 250 users each with a total rps of ~500
neuro-flow run locust -s a
neuro-flow run locust -s b
```

## Monitoring

Inference jobs expose Prometheus metrics on the `/metrics` endpoint.

## API

Inference jobs - directly and via the load balancer - expose a REST API + Swagged docs (the `/docs` endpoint). The model accepts JSON in the following format:

```json
{
  "trip_distance": 3.5,
  "payment_type": 1.0,
  "fare_amount": 12.2,
  "total_amount": 14.5,
  "pickup_month": 3,
  "pickup_hour": 22,
  "pickup_day_of_week": 2
}
```

And returns response in JSON format as well:

```json
{
  "tip": 123
}
```
