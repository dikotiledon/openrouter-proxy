#!/usr/bin/env python3
"""
OpenRouter API Proxy
Proxies requests to OpenRouter API and rotates API keys to bypass rate limits.
"""

import uvicorn
from fastapi import FastAPI

from config import config, logger
from routes import router, lifespan
from utils import get_local_ip

# Disable docs in production by default
docs_url = "/docs" if config.get("server", {}).get("enable_docs", False) else None
app = FastAPI(
    title="OpenRouter OpenAI-Compatible Proxy",
    description="Proxies OpenAI-compatible client requests to OpenRouter API and rotates API keys to bypass rate limits",
    version="1.0.0",
    lifespan=lifespan,
    docs_url=docs_url,
    redoc_url=None,
    openapi_url="/openapi.json" if docs_url else None,
)

# Include routes
app.include_router(router)

# Entry point
if __name__ == "__main__":
    host = config["server"]["host"]
    port = config["server"]["port"]

    # If host is 0.0.0.0, use actual local IP for display
    display_host = get_local_ip() if host == "0.0.0.0" else host

    logger.warning("Starting OpenRouter OpenAI-Compatible Proxy on %s:%s", host, port)
    logger.warning("OpenAI-compatible API URL: http://%s:%s/v1", display_host, port)
    logger.warning("OpenRouter-style API URL: http://%s:%s/api/v1", display_host, port)
    logger.info("Health check: http://%s:%s/health", display_host, port)

    # Configure log level for HTTP access logs
    log_config = uvicorn.config.LOGGING_CONFIG
    http_log_level = config["server"].get("http_log_level", "INFO").upper()
    log_config["loggers"]["uvicorn.access"]["level"] = http_log_level
    logger.info("HTTP access log level set to %s", http_log_level)

    uvicorn.run(app, host=host, port=port, log_config=log_config, timeout_graceful_shutdown=30)
