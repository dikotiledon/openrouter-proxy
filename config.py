#!/usr/bin/env python3
"""
Configuration module for OpenRouter API Proxy.
Loads settings from a YAML file and initializes logging.
"""

import logging
import sys
from typing import Dict, Any

import yaml

from constants import CONFIG_FILE


def load_config() -> Dict[str, Any]:
    """Load configuration from YAML file."""
    try:
        with open(CONFIG_FILE, encoding="utf-8") as file:
            return yaml.safe_load(file)
    except FileNotFoundError:
        print(f"Configuration file {CONFIG_FILE} not found. "
              "Please create it based on config.yml.example.")
        sys.exit(1)
    except yaml.YAMLError as e:
        print(f"Error parsing configuration file: {e}")
        sys.exit(1)


def setup_logging(config_: Dict[str, Any]) -> logging.Logger:
    """Configure logging based on configuration."""
    log_level_str = config_.get("server", {}).get("log_level", "INFO")
    log_level = getattr(logging, log_level_str.upper(), logging.INFO)

    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    logger_ = logging.getLogger("openrouter-proxy")
    logger_.info("Logging level set to %s", log_level_str)

    return logger_


def normalize_and_validate_config(config_data: Dict[str, Any]):
    """
    Normalizes the configuration by adding defaults for missing keys
    and validates the structure and types, logging warnings/errors.
    Modifies the config_data dictionary in place.
    """
    # --- Server Section ---
    if not isinstance(config_data.get("server"), dict):
        logger.warning("'server' section missing or invalid in config.yml. Using defaults.")
        config_data["server"] = {}
    server_config = config_data["server"]

    default_host = "0.0.0.0"
    if not isinstance(server_config.get("host"), str):
        logger.warning(
            "'server.host' missing or invalid in config.yml. Using default: '%s'",
            default_host,
        )
        server_config["host"] = default_host

    default_port = 5555
    port = server_config.get("port")
    if not isinstance(port, int):
        logger.warning(
            "'server.port' missing or invalid in config.yml. Using default: %d",
            default_port,
        )
        server_config["port"] = default_port
    elif port < 1 or port > 65535:
        logger.warning(
            "'server.port' out of range (1-65535) in config.yml. Using default: %d",
            default_port,
        )
        server_config["port"] = default_port

    default_access_key = ""
    access_key = server_config.get("access_key")
    if not isinstance(access_key, str) or not access_key:
        logger.warning(
            "'server.access_key' missing or empty in config.yml. "
            "Proxy will reject all authenticated requests.",
        )
        server_config["access_key"] = default_access_key
    elif access_key.lower() in {
        "your_local_access_key_here",
        "your_access_key_here",
        "changeme",
        "replace_me",
        "your_key_here",
        "example_key",
    }:
        logger.warning(
            "'server.access_key' appears to be a placeholder ('%s'). "
            "Please set a real access key before deploying.",
            access_key,
        )

    providers_config = config_data.get("providers")
    use_provider_registry = isinstance(providers_config, dict) and bool(providers_config)

    if not use_provider_registry:
        # --- Legacy OpenRouter Section ---
        if not isinstance(config_data.get("openrouter"), dict):
            logger.warning("'openrouter' section missing or invalid in config.yml. Using defaults.")
            config_data["openrouter"] = {}
        openrouter_config = config_data["openrouter"]

        default_base_url = "https://openrouter.ai/api/v1"
        if not isinstance(openrouter_config.get("base_url"), str):
            logger.warning(
                "'openrouter.base_url' missing or invalid in config.yml. Using default: %s",
                default_base_url
            )
            openrouter_config["base_url"] = default_base_url
        # Remove trailing slash if present
        openrouter_config["base_url"] = openrouter_config["base_url"].rstrip("/")

        default_public_endpoints = ["/api/v1/models"]
        if "public_endpoints" not in openrouter_config:
            openrouter_config["public_endpoints"] = default_public_endpoints
        elif openrouter_config["public_endpoints"] is None:
            openrouter_config["public_endpoints"] = []
        if not isinstance(openrouter_config["public_endpoints"], list):
            logger.warning(
                "'openrouter.public_endpoints' missing or invalid in config.yml. "
                "Using default: %s",
                default_public_endpoints
            )
            openrouter_config["public_endpoints"] = default_public_endpoints
        else:
            validated_endpoints = []
            for i, endpoint in enumerate(openrouter_config["public_endpoints"]):
                if not isinstance(endpoint, str):
                    logger.warning("Item %d in 'openrouter.public_endpoints' is not a string. Skipping.", i)
                    continue
                if not endpoint:
                    logger.warning("Item %d in 'openrouter.public_endpoints' is empty. Skipping.", i)
                    continue
                # Ensure leading slash
                if not endpoint.startswith("/"):
                    validated_endpoints.append("/" + endpoint)
                else:
                    validated_endpoints.append(endpoint)
            openrouter_config["public_endpoints"] = validated_endpoints

        if not isinstance(openrouter_config.get("keys"), list):
            logger.warning("'openrouter.keys' missing or invalid in config.yml. Using empty list.")
            openrouter_config["keys"] = []
        if not openrouter_config["keys"]:
            logger.warning(
                "'openrouter.keys' list is empty in config.yml. "
                "Proxy will not work for authenticated endpoints."
            )

        if "allowed_models" in openrouter_config and openrouter_config["allowed_models"] is None:
            openrouter_config["allowed_models"] = []
        if not isinstance(openrouter_config.get("allowed_models"), list):
            logger.warning("'openrouter.allowed_models' missing or invalid in config.yml. Using empty list.")
            openrouter_config["allowed_models"] = []
        else:
            validated_models = []
            for i, model_name in enumerate(openrouter_config["allowed_models"]):
                if not isinstance(model_name, str):
                    logger.warning("Item %d in 'openrouter.allowed_models' is not a string. Skipping.", i)
                    continue
                normalized_model = model_name.strip()
                if not normalized_model:
                    logger.warning("Item %d in 'openrouter.allowed_models' is empty. Skipping.", i)
                    continue
                if normalized_model not in validated_models:
                    validated_models.append(normalized_model)
            openrouter_config["allowed_models"] = validated_models

        def_key_selection_strategy = "round-robin"
        if (not isinstance(key_selection_strategy := openrouter_config.get("key_selection_strategy"), str) or
                key_selection_strategy not in ["round-robin", "first", "random"]):
            logger.warning(
                "'openrouter.key_selection_strategy' is unknown: '%s', set '%s'",
                str(key_selection_strategy), def_key_selection_strategy
            )
            openrouter_config["key_selection_strategy"] = def_key_selection_strategy

        if not isinstance(openrouter_config.get("key_selection_opts"), list):
            logger.warning("'openrouter.key_selection_opts' missing or invalid in config.yml. Using empty list.")
            openrouter_config["key_selection_opts"] = []

        default_free_only = False
        if not isinstance(openrouter_config.get("free_only"), bool):
             logger.warning(
                 "'openrouter.free_only' missing or invalid in config.yml. Using default: %s",
                 default_free_only
             )
             openrouter_config["free_only"] = default_free_only

        default_global_rate_delay = 0
        if not isinstance(openrouter_config.get("global_rate_delay"), (int, float)):
             logger.warning(
                 "'openrouter.global_rate_delay' missing or invalid in config.yml. "
                 "Using default: %s",
                 default_global_rate_delay
             )
             openrouter_config["global_rate_delay"] = default_global_rate_delay
        elif openrouter_config.get("global_rate_delay") < 0:
             logger.warning(
                 "'openrouter.global_rate_delay' cannot be negative. Using default: %s",
                 default_global_rate_delay
             )
             openrouter_config["global_rate_delay"] = default_global_rate_delay
    elif not isinstance(config_data.get("openrouter"), dict):
        # Keep an empty legacy section available for compatibility with code
        # that still reads config['openrouter'] directly.
        config_data["openrouter"] = {}

    # --- Request Proxy Section ---
    if not isinstance(config_data.get("requestProxy"), dict):
        logger.warning("'requestProxy' section missing or invalid in config.yml. Using defaults.")
        config_data["requestProxy"] = {}
    proxy_config = config_data["requestProxy"]

    default_proxy_enabled = False
    if not isinstance(proxy_config.get("enabled"), bool):
        logger.warning(
            "'requestProxy.enabled' missing or invalid in config.yml. Using default: %s",
            default_proxy_enabled
        )
        proxy_config["enabled"] = default_proxy_enabled

    default_proxy_url = ""
    if not isinstance(proxy_config.get("url"), str):
        logger.warning(
            "'requestProxy.url' missing or invalid in config.yml. Using default: '%s'",
            default_proxy_url
        )
        proxy_config["url"] = default_proxy_url


# Load configuration
config = load_config()

# Initialize logging
logger = setup_logging(config)

# Normalize and validate configuration (modifies config in place)
normalize_and_validate_config(config)
